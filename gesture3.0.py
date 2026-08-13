"""
ASL 指拼字母实时识别 (Python 版)
================================
架构与网页版一致，分三层:
  1. MediaPipe Hands 提取 21 个手部关键点 (本地摄像头, 实时)
  2. 几何特征 -> 手写参考模板做类高斯似然打分 (softmax 后验概率) -> 逐帧候选字母
  3. 稳定确认后写入原始字母序列; 停顿 / 按空格时调用 Groq API 按英文单词的
     字母语境概率做纠错解码 (处理 A/E/M/N/S/T 等手形相近字母的混淆)

新增功能: 单词结束后不直接写入最终文本, 而是进入"待选择"状态, 由你用手势
决定用哪个版本:
    握拳 + 拇指朝上   -> 使用语境校正后的文本
    握拳 + 拇指朝下   -> 使用校正前的原始拼接文本
待选择期间画面会提示两个候选文本, 正常的字母识别会暂停, 直到你选择完成。
(也保留了 y / n 两个键作为手势不好用时的备用手动选择方式: y=用校正版, n=用原始版)

安装依赖:
    pip install opencv-python mediapipe groq

运行:
    export GROQ_API_KEY=gsk_...   # 语境校正需要, 不设置也能跑, 只是不会校正
    python gesture.py

按键:
    space  = 确认当前单词 (触发语境校正, 进入待选择状态)
    b      = 退格 (删除最后一个原始字母, 若序列为空则删除最后一个已确认单词)
    c      = 清空全部 (含待选择状态)
    y      = [待选择状态下] 手动选择使用校正版
    n      = [待选择状态下] 手动选择使用原始版
    q      = 退出

已支持字母 (几何模板较可靠): B C D F G H I K L O P Q R U V W X Y
近似支持 (依赖语境校正兜底): A E M N S T
暂不支持: J, Z (需要连续运动轨迹建模, 当前只做静态手形)
"""

import math
import os
import time
import json
import collections

import cv2
import mediapipe as mp
import numpy as np

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


# ----------------------------------------------------------------------
# 1. 关键点索引 (与 MediaPipe Hands 21 点模型一致)
# ----------------------------------------------------------------------
class L:
    WRIST = 0
    T_CMC, T_MCP, T_IP, T_TIP = 1, 2, 3, 4
    I_MCP, I_PIP, I_DIP, I_TIP = 5, 6, 7, 8
    M_MCP, M_PIP, M_DIP, M_TIP = 9, 10, 11, 12
    R_MCP, R_PIP, R_DIP, R_TIP = 13, 14, 15, 16
    P_MCP, P_PIP, P_DIP, P_TIP = 17, 18, 19, 20


def dist(a, b):
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def sigmoid(x, k=6.0):
    return 1.0 / (1.0 + math.exp(-k * x))


# ----------------------------------------------------------------------
# 2. 几何特征 -> 候选字母 + 后验概率 (对应网页版 mp())
#    每个分支相当于一个手写高斯模板的"均值条件", margin 越大表示离决策
#    边界越远、"似然"越高, 用 sigmoid 把 margin 映射成 0.4~0.95 的伪后验概率,
#    再把剩余概率按备选字母分摊 -> 得到一个简化的朴素贝叶斯式后验分布。
# ----------------------------------------------------------------------
def predict_letter(lm):
    """lm: 长度21的关键点列表, 每个元素有 .x .y .z (归一化坐标)
    返回: {"letter": str, "confidence": float, "candidates": [(letter, p), ...]}
    """
    scale = max(dist(lm[L.WRIST], lm[L.M_MCP]), 1e-4)

    def curl(tip_idx, mcp_idx):
        return dist(lm[tip_idx], lm[L.WRIST]) / max(dist(lm[mcp_idx], lm[L.WRIST]), 1e-4)

    cI = curl(L.I_TIP, L.I_MCP)
    cM = curl(L.M_TIP, L.M_MCP)
    cR = curl(L.R_TIP, L.R_MCP)
    cP = curl(L.P_TIP, L.P_MCP)

    extI, extM, extR, extP = cI > 1.15, cM > 1.15, cR > 1.15, cP > 1.15
    partI = 0.85 < cI <= 1.15

    thumb_out = dist(lm[L.T_TIP], lm[L.I_MCP]) / scale
    thumb_extended = thumb_out > 0.85

    t_tip_to_i_tip = dist(lm[L.T_TIP], lm[L.I_TIP]) / scale
    t_tip_to_m_tip = dist(lm[L.T_TIP], lm[L.M_TIP]) / scale
    t_tip_to_i_pip = dist(lm[L.T_TIP], lm[L.I_PIP]) / scale
    t_tip_to_m_pip = dist(lm[L.T_TIP], lm[L.M_PIP]) / scale
    t_tip_to_r_mcp = dist(lm[L.T_TIP], lm[L.R_MCP]) / scale
    t_tip_to_p_mcp = dist(lm[L.T_TIP], lm[L.P_MCP]) / scale

    spread_im = dist(lm[L.I_TIP], lm[L.M_TIP]) / scale
    horiz_i = abs(lm[L.I_TIP].x - lm[L.I_MCP].x)
    vert_i = abs(lm[L.I_TIP].y - lm[L.I_MCP].y)
    sideways = horiz_i > vert_i * 1.3

    letter, margin, alts = "?", 0.05, []

    if extI and extM and extR and extP:
        letter, margin, alts = "B", min(cI, cM, cR, cP) - 1.15, ["5"]

    elif extI and extM and extR and not extP:
        letter, margin, alts = "W", cR - 1.15, ["B"]

    elif extI and extM and not extR and not extP:
        if thumb_extended and thumb_out > 1.15:
            letter, margin, alts = "L", thumb_out - 1.15, ["G", "Q"]
        elif sideways:
            letter, margin, alts = "H", 0.3, ["U"]
        elif thumb_out > 0.6 and t_tip_to_m_pip < 0.55:
            letter, margin, alts = "K", 0.4, ["P", "V"]
        elif spread_im > 0.45:
            letter, margin, alts = "V", spread_im - 0.45, ["U", "R"]
        else:
            letter, margin, alts = "U", 0.45 - spread_im, ["V", "R", "H"]

    elif not extI and extM and not extR and not extP:
        letter, margin, alts = "D", 0.2, []

    elif extI and not extM and not extR and not extP:
        if partI:
            letter, margin, alts = "X", 0.15, ["D"]
        elif thumb_extended:
            letter, margin, alts = "L", thumb_out - 0.85, ["G", "Q"]
        else:
            letter, margin, alts = "D", 0.15, ["X"]

    elif not extI and not extM and not extR and extP:
        if thumb_extended:
            letter, margin, alts = "Y", thumb_out - 0.85, ["I"]
        else:
            letter, margin, alts = "I", 0.85 - thumb_out, ["Y"]

    elif not extI and not extM and extR and extP:
        letter, margin, alts = "R", 0.2, ["U"]

    elif not extI and extM and extR and extP:
        letter, margin, alts = "W", 0.2, ["B"]

    elif not extI and not extM and not extR and not extP:
        # A / E / M / N / S / T / O / C 高混淆簇, 靠拇指相对位置区分
        curl_avg = (cI + cM + cR + cP) / 4
        if curl_avg > 0.85:
            if t_tip_to_i_tip < 0.35 and t_tip_to_m_tip < 0.4:
                letter, margin, alts = "O", 0.35 - t_tip_to_i_tip, ["C"]
            else:
                letter, margin, alts = "C", t_tip_to_i_tip - 0.35, ["O"]
        elif t_tip_to_i_tip < 0.3:
            letter, margin, alts = "E", 0.3 - t_tip_to_i_tip, ["A", "S"]
        elif t_tip_to_i_pip < 0.4 and t_tip_to_m_pip < 0.4:
            letter, margin, alts = "T", 0.4 - t_tip_to_i_pip, ["N", "M"]
        elif t_tip_to_p_mcp < 0.55:
            letter, margin, alts = "M", 0.55 - t_tip_to_p_mcp, ["N", "S"]
        elif t_tip_to_r_mcp < 0.5:
            letter, margin, alts = "N", 0.5 - t_tip_to_r_mcp, ["M", "T"]
        elif thumb_out > 0.7:
            letter, margin, alts = "A", thumb_out - 0.7, ["S", "E"]
        else:
            letter, margin, alts = "S", 0.15, ["A", "M"]

    elif not extI and extM and extR and not extP:
        letter, margin, alts = "F", 0.2, ["O"]

    p1 = 0.4 + 0.55 * sigmoid(margin, 8.0)
    remaining = 1.0 - p1
    n_alts = max(len(alts), 1)
    cands = [(letter, p1)]
    for i, a in enumerate(alts[:3]):
        cands.append((a, remaining / n_alts * (1 - i * 0.25)))
    s = sum(p for _, p in cands)
    cands = sorted([(l, p / s) for l, p in cands], key=lambda x: -x[1])

    return {"letter": cands[0][0], "confidence": cands[0][1], "candidates": cands[:4]}


# ----------------------------------------------------------------------
# 2.5 拳头 + 拇指朝上/朝下 手势检测 (用于校正版/原始版选择)
#     条件: 四指都弯曲成拳 (不是伸直), 且拇指明显伸出拳头外 (不是贴着掌心的S)。
#     方向: 用拇指指尖相对手腕在竖直方向(归一化y, 越小越靠画面上方)的偏移来判断。
# ----------------------------------------------------------------------
def detect_thumb_gesture(lm):
    """返回 "up" / "down" / None"""
    scale = max(dist(lm[L.WRIST], lm[L.M_MCP]), 1e-4)

    def curl(tip_idx, mcp_idx):
        return dist(lm[tip_idx], lm[L.WRIST]) / max(dist(lm[mcp_idx], lm[L.WRIST]), 1e-4)

    cI = curl(L.I_TIP, L.I_MCP)
    cM = curl(L.M_TIP, L.M_MCP)
    cR = curl(L.R_TIP, L.R_MCP)
    cP = curl(L.P_TIP, L.P_MCP)

    # 四指握拳: 都没有伸直 (阈值比 predict_letter 里的 1.15 略宽松一点)
    fist = cI < 1.05 and cM < 1.05 and cR < 1.05 and cP < 1.05

    thumb_out = dist(lm[L.T_TIP], lm[L.I_MCP]) / scale
    thumb_extended = thumb_out > 0.5  # 拇指从拳头里伸出来, 不是贴着掌心 (那样是S)

    if not (fist and thumb_extended):
        return None

    # 归一化坐标 y 向下为正: 拇指指尖 y 比手腕 y 小很多 = 指向画面上方 = 朝上
    dy = (lm[L.T_TIP].y - lm[L.WRIST].y) / scale

    if dy < -0.35:
        return "up"
    elif dy > 0.35:
        return "down"
    return None


# ----------------------------------------------------------------------
# 3. 语境校正: 使用 Groq (免费) 纠正字母序列
# ----------------------------------------------------------------------
try:
    from groq import Groq
    _HAS_GROQ = True
except ImportError:
    _HAS_GROQ = False

_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY")) if (_HAS_GROQ and os.environ.get("GROQ_API_KEY")) else None


def correct_word(raw_seq):
    """raw_seq: [{"letter": str, "alts": [str, ...]}, ...] -> 返回猜测的单词字符串"""
    fallback = "".join(item["letter"] for item in raw_seq)
    if _groq_client is None or not raw_seq:
        return fallback

    payload = "".join(item["letter"] for item in raw_seq)
    prompt = f"ASL手指拼写识别到了字母序列: {payload}。这个序列可能有识别错误。请纠正成最可能的英文单词，只输出这个单词，不要任何其他文字。"

    try:
        response = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Groq 免费模型
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0,
        )
        word = response.choices[0].message.content.strip().split()[0] if response.choices else fallback
        return word
    except Exception as e:
        print(f"[语境校正调用失败, 回退为原始拼接] {e}")
        return fallback


# ----------------------------------------------------------------------
# 4. 主循环: 摄像头 -> MediaPipe -> mp -> 稳定确认 -> 校正 -> 手势选择版本
# ----------------------------------------------------------------------
def main():
    mp_hands_solution = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    CONFIRM_TIME = 2.0
    CONF_THRESHOLD = 0.55
    NO_HAND_COMMIT_FRAMES = 28
    THUMB_CONFIRM_TIME = 0.8  # 拳头+拇指手势需要保持多久才算确认选择

    raw_seq = collections.deque()
    final_words = []
    stable_letter = None
    stable_start_time = None
    last_committed = None
    no_hand_frames = 0

    # ---- 待选择状态: 单词已经算好校正版/原始版, 等你用手势选一个 ----
    pending_choice = None  # {"raw": str, "corrected": str} 或 None
    thumb_state = None
    thumb_state_start = None

    def enter_pending_choice(seq):
        """把当前 raw_seq 变成待选择状态, 并清空 raw_seq / 重置字母确认状态"""
        nonlocal pending_choice, thumb_state, thumb_state_start
        nonlocal stable_letter, stable_start_time, last_committed
        raw_text = "".join(i["letter"] for i in seq)
        corrected_text = correct_word(list(seq))
        pending_choice = {"raw": raw_text, "corrected": corrected_text}
        thumb_state, thumb_state_start = None, None
        stable_letter, stable_start_time, last_committed = None, None, None
        print(f"[待确认] 校正版: {corrected_text}   原始版: {raw_text}   "
              f"-> 拳头+拇指朝上=用校正 / 拳头+拇指朝下=用原始 (或按 y/n)")

    def resolve_pending_choice(use_corrected):
        """根据选择把 pending_choice 写入 final_words 并清空待选择状态"""
        nonlocal pending_choice, thumb_state, thumb_state_start
        chosen = pending_choice["corrected"] if use_corrected else pending_choice["raw"]
        final_words.append(chosen)
        print(f"[已选择] {'校正版' if use_corrected else '原始版'} -> {chosen}   "
              f"完整文本: {' '.join(final_words)}")
        pending_choice = None
        thumb_state, thumb_state_start = None, None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    if _groq_client is None:
        print("[提示] 未检测到 GROQ_API_KEY 或未安装 groq 包, "
              "语境校正会直接回退为原始字母拼接。")

    with mp_hands_solution.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 镜像, 自拍视角
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            cur_letter, cur_candidates = None, []

            if results.multi_hand_landmarks:
                no_hand_frames = 0
                hand_lm = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands_solution.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                current_time = time.time()

                if pending_choice is not None:
                    # ---- 待选择状态: 只做拳头+拇指方向识别, 不做字母识别 ----
                    gesture = detect_thumb_gesture(hand_lm.landmark)

                    if gesture == thumb_state:
                        if thumb_state_start is None:
                            thumb_state_start = current_time
                        elapsed = current_time - thumb_state_start
                        if gesture is not None:
                            progress = min(elapsed / THUMB_CONFIRM_TIME, 1.0)
                            label = "校正版" if gesture == "up" else "原始版"
                            cv2.putText(frame, f"选择中: {label} {progress*100:.0f}%",
                                        (frame.shape[1] - 260, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            if elapsed >= THUMB_CONFIRM_TIME:
                                resolve_pending_choice(use_corrected=(gesture == "up"))
                    else:
                        thumb_state = gesture
                        thumb_state_start = current_time
                else:
                    # ---- 正常字母识别流程 ----
                    out = predict_letter(hand_lm.landmark)
                    cur_letter, cur_candidates = out["letter"], out["candidates"]

                    if cur_letter == stable_letter:
                        if stable_start_time is None:
                            stable_start_time = current_time
                        elapsed = current_time - stable_start_time

                        # 显示停留进度
                        progress = min(elapsed / CONFIRM_TIME, 1.0)
                        cv2.putText(frame, f"稳定: {progress*100:.0f}%",
                                    (frame.shape[1] - 150, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        if (elapsed >= CONFIRM_TIME
                                and out["confidence"] >= CONF_THRESHOLD
                                and cur_letter != "?"
                                and cur_letter != last_committed):
                            alts = [c for c, _ in out["candidates"][1:]]
                            raw_seq.append({"letter": cur_letter, "alts": alts})
                            last_committed = cur_letter
                            stable_start_time = None  # 重置计时器
                            print(f"[确认字母] {cur_letter}  当前序列: "
                                  f"{''.join(i['letter'] for i in raw_seq)}")
                    else:
                        stable_letter = cur_letter
                        stable_start_time = current_time
            else:
                stable_letter = None
                stable_start_time = None
                last_committed = None
                thumb_state, thumb_state_start = None, None  # 手离开画面时重置手势状态, 不影响 pending_choice 本身
                no_hand_frames += 1
                if pending_choice is None and no_hand_frames == NO_HAND_COMMIT_FRAMES and raw_seq:
                    enter_pending_choice(list(raw_seq))
                    raw_seq.clear()

            # ---- 叠加显示 ----
            y = 30
            for letter, p in cur_candidates:
                cv2.putText(frame, f"{letter}: {p*100:.0f}%", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (94, 234, 212), 2)
                y += 26

            if stable_letter and stable_start_time:
                elapsed = time.time() - stable_start_time
                cv2.putText(frame, f"当前: {stable_letter} ({elapsed:.1f}s/{CONFIRM_TIME}s)",
                            (10, frame.shape[0] - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if pending_choice is not None:
                cv2.putText(frame, f"校正版: {pending_choice['corrected']}   原始版: {pending_choice['raw']}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.putText(frame, "拳头+拇指朝上=用校正 / 拳头+拇指朝下=用原始 (或按 y/n)",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.putText(frame, "raw: " + "".join(i["letter"] for i in raw_seq),
                        (10, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (244, 183, 64), 2)
            cv2.putText(frame, "text: " + " ".join(final_words),
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (231, 236, 239), 2)
            cv2.imshow("ASL Fingerspelling Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                if pending_choice is None and raw_seq:
                    enter_pending_choice(list(raw_seq))
                    raw_seq.clear()
            elif key == ord("y"):
                if pending_choice is not None:
                    resolve_pending_choice(use_corrected=True)
            elif key == ord("n"):
                if pending_choice is not None:
                    resolve_pending_choice(use_corrected=False)
            elif key == ord("b"):
                if pending_choice is not None:
                    pass  # 待选择状态下不响应退格, 先选完再说
                elif raw_seq:
                    raw_seq.pop()
                elif final_words:
                    final_words.pop()
            elif key == ord("c"):
                raw_seq.clear()
                final_words.clear()
                pending_choice = None
                thumb_state, thumb_state_start = None, None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()