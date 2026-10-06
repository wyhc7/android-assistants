# -*- coding: utf-8 -*-
"""降目标阈值到底会不会改变选择: 在真实对局帧上, 对比 0.35 与 0.20/0.10 选出的落点。

只看两件事:
  新增可跳 = 0.35 判成"无目标"、低阈值能给出落点 —— 这是收益(真机上正是这种情况)
  改选别的 = 两者都有落点, 但落点位移 > 5px        —— 这是风险(可能指到别的方块)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from calib import merge_ensemble
from detect import Detector

A = "runs/jump_pose_hsv/weights/best_ncnn_model"
B = "runs/jump_pose-3/weights/best_ncnn_model"
LO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 400

frames = []
for pat in ("emulator/*/a*_pre.png", "emulator/*/*_pre.png", "emulator_skin/*.png", "screenshots/*.png"):
    for p in sorted(Path(".").glob(pat)):
        frames.append(p)
frames = frames[:LIMIT]
print("真实帧 %d 张" % len(frames))

hi_a, hi_b = Detector(A, (0.15, 0.35)), Detector(B, (0.15, 0.35))
lo_a, lo_b = Detector(A, (0.15, LO)), Detector(B, (0.15, LO))

gain = risk = same = both_miss = 0
changed = []
for p in frames:
    img = cv2.imread(str(p))
    if img is None:
        continue
    rh = merge_ensemble((hi_a.infer(img), hi_b.infer(img)))
    rl = merge_ensemble((lo_a.infer(img), lo_b.infer(img)))
    th = rh[1] is not None and rh[0] is not None
    tl = rl[1] is not None and rl[0] is not None
    if not th and tl:
        gain += 1
        changed.append((p, "新增可跳", rl[1]["conf"]))
    elif not th and not tl:
        both_miss += 1
    elif th and tl:
        d = float(np.hypot(rh[1]["kpt"][0] - rl[1]["kpt"][0], rh[1]["kpt"][1] - rl[1]["kpt"][1]))
        if d > 5:
            risk += 1
            changed.append((p, "改选别的 %.1fpx" % d, rl[1]["conf"]))
        else:
            same += 1
    else:
        changed.append((p, "高阈值能跳低阈值不能(不该发生)", 0))

print("一致 %d | 新增可跳 %d | 改选别的 %d | 都检不出 %d" % (same, gain, risk, both_miss))
for p, why, c in changed[:15]:
    print("   %s  %s conf=%.3f" % (p, why, c))
