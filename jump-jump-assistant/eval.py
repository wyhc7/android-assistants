# -*- coding: utf-8 -*-
"""评估: 关键点像素误差 (PCK) 与检测 mAP。

用法: python eval.py [weights]   默认 runs/jump_pose/weights/best.pt
对 val 集逐张推理, 与 annotations/ 真值对比, 输出每类关键点平均/中位像素误差。
"""
import json
import statistics
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent
weights = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "jump_pose" / "weights" / "best.pt"
model = YOLO(str(weights))
errors = {"piece": [], "target": []}

for img_path in sorted((ROOT / "dataset" / "val" / "images").glob("*")):
    gt_path = ROOT / "annotations" / (img_path.stem + ".json")
    if not gt_path.exists():
        continue
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    res = model.predict(str(img_path), verbose=False)[0]
    if res.keypoints is None or len(res.boxes) == 0:
        print(f"[漏检] {img_path.name}")
        continue
    # 按类别取置信度最高的框对应关键点
    best = {}
    for box, kpt in zip(res.boxes, res.keypoints.xy):
        cls = int(box.cls)
        if cls not in best or box.conf > best[cls][1]:
            best[cls] = (kpt[0].tolist(), float(box.conf))
    for cls, name in ((0, "piece"), (1, "target")):
        if cls in best:
            px, py = best[cls][0]
            gx, gy = gt[name]
            err = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
            errors[name].append(err)
            print(f"{img_path.name} {name}: {err:.1f}px")

for name, errs in errors.items():
    if errs:
        pck5 = sum(e <= 5 for e in errs) / len(errs)
        print(f"{name}: n={len(errs)} mean={statistics.mean(errs):.2f}px "
              f"median={statistics.median(errs):.2f}px PCK@5px={pck5:.1%}")
