# -*- coding: utf-8 -*-
"""对比人工标注与模型预测, 输出裁剪可视化。用法: python cmp_label.py <图名> [y0 y1]"""
import json
import sys
from pathlib import Path

import cv2

import detect as D

ROOT = Path(__file__).parent
name = sys.argv[1]
y0, y1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 2372)

img = cv2.imread(str(ROOT / "screenshots" / name))
ann = json.loads((ROOT / "annotations" / (Path(name).stem + ".json")).read_text(encoding="utf-8"))
det = D.Detector(ROOT / "runs" / "jump_pose-2" / "weights" / "best_ncnn_model")
res = det.infer(img)

crop = img[y0:y1].copy()
px, py = int(ann["piece"][0]), int(ann["piece"][1]) - y0
cv2.circle(crop, (px, py), 9, (0, 0, 255), -1)
cv2.drawMarker(crop, (px, py), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 3)
tx, ty = int(ann["target"][0]), int(ann["target"][1]) - y0
cv2.drawMarker(crop, (tx, ty), (255, 0, 255), cv2.MARKER_CROSS, 40, 3)
cv2.putText(crop, "HUMAN target", (tx - 150, ty - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 3)

for c, col in ((0, (255, 128, 0)), (1, (0, 255, 255))):
    if not res[c]:
        continue
    x1, yy1, x2, yy2 = [int(v) for v in res[c]["box"]]
    cv2.rectangle(crop, (x1, yy1 - y0), (x2, yy2 - y0), col, 3)
    kx, ky = int(res[c]["kpt"][0]), int(res[c]["kpt"][1]) - y0
    cv2.drawMarker(crop, (kx, ky), col, cv2.MARKER_TILTED_CROSS, 40, 3)
    cv2.putText(crop, "MODEL %.2f" % res[c]["conf"], (x1, max(30, yy1 - y0 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)

out = ROOT / "preview" / ("cmp_" + Path(name).stem[:12] + ".png")
cv2.imwrite(str(out), crop)
print(out, crop.shape)
