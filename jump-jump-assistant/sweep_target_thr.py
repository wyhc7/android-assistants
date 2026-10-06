# -*- coding: utf-8 -*-
"""目标类阈值扫描: 目标类阈值一直贴着 0.35, 药瓶类目标本身只有 ~0.30, 真机上基线模型退化后
就会整帧判成"无目标"。这里在人工精标上把阈值从 0.35 往下扫, 看漏检与误检(落点跑偏)各怎么变。

判据:
  漏检   = 棋子或目标有一个没检出来
  违规   = 落点 Y 不在棋子上方(几何规则本来就会挡掉)
  跑偏   = 落点与人工标的距离 > 15px(相当于指到了别的方块)
"""
import json
import sys

import cv2
import numpy as np

from calib import merge_ensemble
from detect import Detector
from eval_skin import HUMAN, SCREENS

A = sys.argv[1] if len(sys.argv) > 1 else "runs/jump_pose_hsv/weights/best_ncnn_model"
B = sys.argv[2] if len(sys.argv) > 2 else "runs/jump_pose-3/weights/best_ncnn_model"
THRESHOLDS = [0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

frames = []
for f in sorted(HUMAN.glob("*.json")):
    rec = json.loads(f.read_text(encoding="utf-8"))
    img = cv2.imread(str(SCREENS / rec["image"]))
    if img is not None:
        frames.append((img, rec))
print("人工精标 %d 帧" % len(frames))

print(" 目标阈值 | 漏检 | 违规 | 落点跑偏>15px | 落点平均误差")
for thr in THRESHOLDS:
    da, db = Detector(A, (0.15, thr)), Detector(B, (0.15, thr))
    miss = viol = off = 0
    errs = []
    for img, rec in frames:
        r = merge_ensemble((da.infer(img), db.infer(img)))
        if r[0] is None or r[1] is None:
            miss += 1
            continue
        err = float(np.hypot(r[1]["kpt"][0] - rec["target"][0], r[1]["kpt"][1] - rec["target"][1]))
        errs.append(err)
        if err > 15:
            off += 1
        if r[1]["kpt"][1] >= r[0]["kpt"][1]:
            viol += 1
    print(" %8.2f | %4d | %4d | %12d | %8.2fpx"
          % (thr, miss, viol, off, np.mean(errs) if errs else 0))
