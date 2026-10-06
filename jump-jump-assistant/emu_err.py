# -*- coding: utf-8 -*-
"""从落地后的帧反推落点偏差, 用于校准按压系数 k。

原理: 落地后棋子站在"刚跳上去的那块方块"顶面上。该方块顶面是菱形/圆,
其左右顶点的连线过中心, 且该行就是顶面最宽的一行。
因此: 在棋子底部所在行附近找方块轮廓最宽的一行 -> 中点即顶面中心,
      与棋子底部位置的差 = 落点偏差(沿跳跃方向的投影即控制误差)。

用法: python emu_err.py [模型目录]
"""
import glob
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

import auto_annotate2 as A
from detect import Detector

ROOT = Path(__file__).parent
MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "jump_pose-3" / "weights" / "best_ncnn_model"


def block_center(mask, px, py, span=160):
    """在 mask 中找包含 (px, py+25) 的连通域, 返回顶面最宽行的中点与行号。"""
    h, w = mask.shape
    sy = int(min(max(py + 25, 0), h - 1))
    sx = int(min(max(px, 0), w - 1))
    n, lab = cv2.connectedComponents(mask)
    idx = lab[sy, sx]
    if idx == 0:
        return None
    comp = (lab == idx)
    best = None
    y0, y1 = int(max(py - span, 0)), int(min(py + span, h - 1))
    for y in range(y0, y1 + 1):
        xs = np.nonzero(comp[y])[0]
        if xs.size < 20:
            continue
        width = xs[-1] - xs[0]
        if best is None or width > best[0]:
            best = (width, (xs[0] + xs[-1]) / 2.0, y)
    return best


def main():
    det = Detector(MODEL)
    files = sorted(ROOT.glob("emulator/a*_post.png"))
    errs = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        res = det.infer(img)
        if res[0] is None:
            continue
        px, py = res[0]["kpt"][0], res[0]["kpt"][1]
        pmask, _ = A.piece_mask(img)
        mask = A.fg_mask(img, pmask)
        b = block_center(mask, px, py)
        if b is None:
            continue
        width, cx, cy = b
        errs.append((cx - px, cy - py, width, p.name))
    if not errs:
        print("无有效样本")
        return
    ex = [e[0] for e in errs]
    ey = [e[1] for e in errs]
    mag = [(e[0] ** 2 + e[1] ** 2) ** 0.5 for e in errs]
    print("样本 %d 帧" % len(errs))
    print("横向偏差: 均值 %+.1fpx 中位 %+.1fpx 标准差 %.1f" % (sum(ex) / len(ex), statistics.median(ex), statistics.pstdev(ex)))
    print("纵向偏差: 均值 %+.1fpx 中位 %+.1fpx 标准差 %.1f" % (sum(ey) / len(ey), statistics.median(ey), statistics.pstdev(ey)))
    print("偏差幅值: 均值 %.1fpx 中位 %.1fpx 最大 %.1fpx" % (sum(mag) / len(mag), statistics.median(mag), max(mag)))
    big = sorted(zip(mag, [e[3] for e in errs]), reverse=True)[:8]
    print("偏差最大 8 帧:", [(round(m), n[:12]) for m, n in big])


if __name__ == "__main__":
    main()
