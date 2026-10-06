# -*- coding: utf-8 -*-
"""换皮肤后的数据采集器: 能跳就跳, 跳崩了就自动重开, 每帧都存。

为什么要单独写: 换皮肤后棋子类分数会塌到 0.02~0.5(旧皮肤 0.9), 通用脚本一遇到
"棋子没检出"就停; 这里改成自动识别结算画面并点"再玩一局"重开, 保证持续采集。

用法: python collect_skin.py <模型目录> <输出前缀> [目标帧数] [k]
输出: emulator/<前缀>_NNN.png + emulator/<前缀>.jsonl(每帧的检测结果)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from detect import Detector

ADB = r"D:\Program Files\Netease\MuMu\nx_device\15.0\shell\adb.exe"
DEV = "127.0.0.1:16384"
ROOT = Path(__file__).parent
EMU = ROOT / "emulator"

MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "jump_pose-3" / "weights" / "best_ncnn_model"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "skin"
WANT = int(sys.argv[3]) if len(sys.argv) > 3 else 120
K = float(sys.argv[4]) if len(sys.argv) > 4 else 1.38
PIECE_CONF = 0.008          # 换皮肤后棋子类分数塌陷, 用极低阈值换取"能定位"

det = Detector(MODEL, (PIECE_CONF, 0.35))


def sh(*args):
    return subprocess.run([ADB, "-s", DEV, *args], capture_output=True, timeout=30)


def capture(tag):
    remote = "/sdcard/_cs.png"
    sh("shell", "screencap", "-p", remote)
    local = EMU / ("%s.png" % tag)
    if local.exists():
        local.unlink()
    sh("pull", remote, str(local))
    return cv2.imread(str(local))


def press(x, y, ms):
    sh("shell", "input", "swipe", str(int(x)), str(int(y)), str(int(x)), str(int(y)), str(int(ms)))


def find_restart_button(img):
    """结算画面里找白色胶囊按钮: 下半屏里足够宽的高亮连通域。"""
    h, w = img.shape[:2]
    sub = img[int(h * 0.45):, :]
    bright = ((sub[:, :, 0] > 225) & (sub[:, :, 1] > 225) & (sub[:, :, 2] > 225)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bright, 8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw > w * 0.3 and 40 < bh < 260 and area > 20000:
            if best is None or area > best[0]:
                best = (area, cent[i][0], cent[i][1] + int(h * 0.45))
    return None if best is None else (best[1], best[2])


def main():
    rows = []
    saved = 0
    restarts = 0
    tries = 0
    while saved < WANT and tries < WANT * 4:
        tries += 1
        img = capture("%s_%03d" % (PREFIX, saved))
        if img is None:
            time.sleep(0.5)
            continue
        r = det.infer(img)
        if r[0] is None or r[1] is None or r[1]["kpt"][1] >= r[0]["kpt"][1]:
            btn = find_restart_button(img)
            if btn is not None:
                press(btn[0], btn[1], 60)
                restarts += 1
                print("重开第 %d 次 (按钮 %.0f,%.0f)" % (restarts, btn[0], btn[1]))
                time.sleep(6.0)
            else:
                time.sleep(1.0)
            (EMU / ("%s_%03d.png" % (PREFIX, saved))).unlink(missing_ok=True)
            continue
        p, t = r[0]["kpt"], r[1]["kpt"]
        d = ((t[0] - p[0]) ** 2 + (p[1] - t[1]) ** 2) ** 0.5
        rows.append(dict(frame="%s_%03d.png" % (PREFIX, saved), width=img.shape[1], height=img.shape[0],
                         piece=[round(p[0], 2), round(p[1], 2)], target=[round(t[0], 2), round(t[1], 2)],
                         target_box=[round(float(v), 2) for v in r[1]["box"]],
                         conf_piece=round(r[0]["conf"], 4), conf_target=round(r[1]["conf"], 3),
                         dist=round(d, 1)))
        saved += 1
        print("采集 %d/%d  conf=%.3f/%.2f 跳距 %.0fpx" % (saved, WANT, r[0]["conf"], r[1]["conf"], d))
        press(p[0], p[1], d * K)
        time.sleep(1.9)

    out = EMU / ("%s.jsonl" % PREFIX)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("完成: %d 帧, 重开 %d 次 -> %s" % (saved, restarts, out))


if __name__ == "__main__":
    main()
