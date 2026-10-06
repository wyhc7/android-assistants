# -*- coding: utf-8 -*-
"""标注复核工具: 展示自动预标注, 人工确认或修正。

用法: python annotate.py
操作:
  Enter : 接受当前标注, 写入已确认标记并跳下一张
  p     : 重标棋子 -> 之后左键点棋子底部中心
  t     : 重标目标 -> 之后左键依次点顶面框的两个对角点
  s     : 跳过    q : 退出
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parent
SRC, ANN = ROOT / "screenshots", ROOT / "annotations"
CLASSES = ["piece", "target"]

state = {"mode": None, "clicks": [], "rec": None}


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN or state["mode"] is None:
        return
    state["clicks"].append((x, y))
    if state["mode"] == "p":
        state["rec"]["piece"] = [float(x), float(y)]
        state["mode"] = None
    elif state["mode"] == "t" and len(state["clicks"]) == 2:
        (x1, y1), (x2, y2) = state["clicks"]
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        state["rec"]["target_box"] = [x, y, w, h]
        state["rec"]["target"] = [x + w / 2, y + h / 2]
        state["mode"] = None
        state["clicks"] = []
    elif state["mode"] == "t":
        state["clicks"] = state["clicks"][-1:]


def draw(img, rec):
    v = img.copy()
    px, py = [int(p) for p in rec["piece"]]
    cv2.circle(v, (px, py), 12, (0, 0, 255), -1)
    x, y, w, h = rec["target_box"]
    cv2.rectangle(v, (x, y), (x + w, y + h), (0, 255, 0), 4)
    tx, ty = [int(p) for p in rec["target"]]
    cv2.drawMarker(v, (tx, ty), (255, 0, 255), cv2.MARKER_CROSS, 40, 4)
    return v


def main():
    cv2.namedWindow("review", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("review", on_mouse)
    files = sorted(ANN.glob("*.json"))
    if not files:
        sys.exit("annotations/ 为空, 先运行 auto_annotate.py")
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("confirmed"):
            continue
        state["rec"] = rec
        state["mode"] = None
        state["clicks"] = []
        img = cv2.imread(str(SRC / rec["image"]))
        while True:
            v = draw(img, rec)
            hint = {"p": "点击棋子底部中心", "t": "点击顶面两对角",
                    None: "Enter=接受 p=改棋子 t=改目标 s=跳过"}[state["mode"]]
            cv2.putText(v, hint, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)
            cv2.imshow("review", v)
            k = cv2.waitKey(30) & 0xFF
            if k == 13:  # Enter
                rec["confirmed"] = True
                f.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                             encoding="utf-8")
                break
            if k == ord("p"):
                state["mode"] = "p"
            elif k == ord("t"):
                state["mode"] = "t"
                state["clicks"] = []
            elif k == ord("s"):
                break
            elif k == ord("q"):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()
    print("复核完成")


if __name__ == "__main__":
    main()
