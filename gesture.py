"""
ASL 指拼字母实时识别 (Python 版)
================================
架构与网页版一致，分三层:
  1. MediaPipe Hands 提取 21 个手部关键点 (本地摄像头, 实时)
  2. 几何特征 -> 手写参考模板做类高斯似然打分 (softmax 后验概率) -> 逐帧候选字母
  3. 稳定确认后写入原始字母序列; 停顿 / 按空格时调用 Claude API 按英文单词的
     字母语境概率做纠错解码 (处理 A/E/M/N/S/T 等手形相近字母的混淆)

安装依赖:
    pip install opencv-python mediapipe anthropic

运行:
    export ANTHROPIC_API_KEY=sk-ant-...   # 语境校正需要, 不设置也能跑, 只是不会校正
    python asl_fingerspell.py

按键:
    space  = 确认当前单词 (触发语境校正)
    b      = 退格 (删除最后一个原始字母, 若序列为空则删除最后一个已确认单词)
    c      = 清空全部
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
# 3. 语境校正: 把原始字母序列 (含每位备选) 发给 Claude, 按语境猜单词
# ----------------------------------------------------------------------
_client = anthropic.Anthropic() if (_HAS_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY")) else None


def correct_word(raw_seq):
    """raw_seq: [{"letter": str, "alts": [str, ...]}, ...] -> 返回猜测的单词字符串"""
    fallback = "".join(item["letter"] for item in raw_seq)
    if _client is None or not raw_seq:
        return fallback

    payload = " ".join(
        f"{item['letter']}[{'/'.join(item['alts']) or '-'}]" for item in raw_seq
    )
    prompt = (
        "你在做ASL(美国手语)指拼字母的实时纠错。下面是本地手势分类器按摄像头逐帧识别出的一串原始字母，"
        "每个字母后方括号里是这一位置容易被手势分类器混淆的备选字母（因为手形相近，比如 A/E/M/N/S/T 之间、"
        "K/P/V 之间容易混）。\n"
        f"原始序列: {payload}\n"
        "请判断说话者最可能想要拼写的一个英文单词（如果整体更像多个词或不完整也给出最合理的猜测）。"
        '只输出严格JSON，不要markdown代码块，不要任何多余文字，格式:\n'
        '{"word":"最可能的单词","confidence":0到1之间的小数}'
    )

    try:
        resp = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return parsed.get("word", fallback)
    except Exception as e:
        print(f"[语境校正调用失败, 回退为原始拼接] {e}")
        return fallback


# ----------------------------------------------------------------------
# 4. 主循环: 摄像头 -> MediaPipe -> mp -> 稳定确认 -> 校正
# ----------------------------------------------------------------------
def main():
    mp_hands_solution = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # STABLE_FRAMES = 5
    CONFIRM_TIME = 2.0
    CONF_THRESHOLD = 0.55
    NO_HAND_COMMIT_FRAMES = 28

    raw_seq = collections.deque()
    final_words = []
    stable_letter = None
    stable_start_time = None
    last_committed = None
    no_hand_frames = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    if _client is None:
        print("[提示] 未检测到 ANTHROPIC_API_KEY 或未安装 anthropic 包, "
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
                out = predict_letter(hand_lm.landmark)
                cur_letter, cur_candidates = out["letter"], out["candidates"]
                current_time = time.time()

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
                no_hand_frames += 1
                if no_hand_frames == NO_HAND_COMMIT_FRAMES and raw_seq:
                    word = correct_word(list(raw_seq))
                    final_words.append(word)
                    print(f"[语境校正] -> {word}   完整文本: {' '.join(final_words)}")
                    raw_seq.clear()

            # ---- 叠加显示 ----
            y = 30
            for letter, p in cur_candidates:
                cv2.putText(frame, f"{letter}: {p*100:.0f}%", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (94, 234, 212), 2)
                y += 26
            
            # ===== 插入这段 =====
            if stable_letter and stable_start_time:
                elapsed = time.time() - stable_start_time
                cv2.putText(frame, f"当前: {stable_letter} ({elapsed:.1f}s/{CONFIRM_TIME}s)", 
                            (10, frame.shape[0] - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # ===== 插入结束 =====
            
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
                if raw_seq:
                    word = correct_word(list(raw_seq))
                    final_words.append(word)
                    print(f"[手动语境校正] -> {word}")
                    raw_seq.clear()
            elif key == ord("b"):
                if raw_seq:
                    raw_seq.pop()
                elif final_words:
                    final_words.pop()
            elif key == ord("c"):
                raw_seq.clear()
                final_words.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()