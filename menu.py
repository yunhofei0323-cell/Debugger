"""
menu.py

简单的菜单页面 (OpenCV 窗口)，用于在学习模式 (learning.py)、测试模式 (text.py)、翻译模式 (translate.py) 之间切换。
该文件负责前端页面的设置与渲染，按键操作:
    1 -> 学习模式
    2 -> 测试模式
    3 -> 翻译模式
    s -> 设置 (占位，目前支持调整字体大小)
    q -> 退出

调用对应模块的 main() 函数（会阻塞直到该模式退出），然后返回菜单。

依赖: OpenCV, gesture.py 中定义的识别逻辑会被各模式引用。
"""

import cv2
import time
import importlib

MENU_W = 640
MENU_H = 360
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (230, 230, 230)
HIGHLIGHT = (94, 234, 212)

# 简单的可配置项
settings = {
    "font_scale": 1.0,
    "thickness": 2,
}


def draw_menu(win_name="ASL Menu"):
    img = 255 * (None)  # placeholder will be overwritten
    img = (np_full := (255 * 0.0) + 0).astype('uint8') if False else None
    # We build image by NumPy to be robust even if no assets.
    import numpy as np
    img = np.zeros((MENU_H, MENU_W, 3), dtype=np.uint8)
    img[:] = BG_COLOR

    title = "ASL 学习/测试/翻译 菜单"
    cv2.putText(img, title, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.0 * settings["font_scale"], TEXT_COLOR, settings["thickness"] + 1)

    lines = [
        "1 - 学习模式 (learning)",
        "2 - 测试模式 (text)",
        "3 - 翻译模式 (translate)",
        "s - 设置字体大小",
        "q - 退出",
    ]

    y = 110
    for i, ln in enumerate(lines):
        color = TEXT_COLOR
        if i < 3:
            color = HIGHLIGHT
        cv2.putText(img, ln, (40, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9 * settings["font_scale"], color, settings["thickness"])
        y += 44

    return img


def run_mode(module_name):
    # 动态导入并运行该模式的 main() 函数
    try:
        mod = importlib.import_module(module_name)
        importlib.reload(mod)
        if hasattr(mod, "main"):
            mod.main()
        else:
            print(f"模块 {module_name} 中未找到 main()，返回菜单。")
    except Exception as e:
        print(f"启动模式 {module_name} 失败: {e}")


def settings_menu():
    # 仅支持调整字体大小
    print("进入设置: 使用上下键调整字体大小，按 Enter 返回")
    val = settings["font_scale"]
    while True:
        print(f"当前 font_scale = {val:.2f}")
        key = input("按 u 增加, d 减少, r 重置, 回车退出: ").strip().lower()
        if key == "u":
            val += 0.1
        elif key == "d":
            val = max(0.3, val - 0.1)
        elif key == "r":
            val = 1.0
        else:
            break
    settings["font_scale"] = val


def main():
    win = "ASL Menu"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, MENU_W, MENU_H)

    while True:
        img = draw_menu(win)
        cv2.imshow(win, img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("1"):
            cv2.destroyWindow(win)
            run_mode("learning")
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, MENU_W, MENU_H)
        elif key == ord("2"):
            cv2.destroyWindow(win)
            run_mode("text")
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, MENU_W, MENU_H)
        elif key == ord("3"):
            cv2.destroyWindow(win)
            run_mode("translate")
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, MENU_W, MENU_H)
        elif key == ord("s"):
            cv2.destroyWindow(win)
            settings_menu()
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, MENU_W, MENU_H)
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("打开菜单: 1 学习, 2 测试, 3 翻译, q 退出")
    main()
