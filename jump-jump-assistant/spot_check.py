# -*- coding: utf-8 -*-
"""抽查: 围绕目标框裁全尺寸局部并拼图。"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
SRC, ANN = ROOT / "screenshots", ROOT / "annotations"
files = sorted(ANN.glob("*.json"))
random.seed(0)
picks = random.sample(files, min(12, len(files)))
crops = []
for f in picks:
    rec = json.loads(f.read_text(encoding="utf-8"))
    img = cv2.imread(str(SRC / rec["image"]))
    px, py = [int(v) for v in rec["piece"]]
    x, y, w, h = rec["target_box"]
    cv2.circle(img, (px, py), 12, (0, 0, 255), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    tx, ty = [int(v) for v in rec["target"]]
    cv2.drawMarker(img, (tx, ty), (255, 0, 255), cv2.MARKER_CROSS, 40, 4)
    # 裁剪覆盖棋子与目标的区域
    x0 = max(0, min(px, x) - 150)
    y0 = max(0, min(py, y) - 300)
    x1 = min(img.shape[1], max(px, x + w) + 150)
    y1 = min(img.shape[0], max(py, y + h) + 150)
    crop = img[y0:y1, x0:x1]
    crop = cv2.resize(crop, (640, int(crop.shape[0] * 640 / crop.shape[1])))
    cv2.putText(crop, f.stem[-12:], (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)
    crops.append(crop)
mh = max(c.shape[0] for c in crops)
crops = [cv2.copyMakeBorder(c, 0, mh - c.shape[0], 0, 0,
                            cv2.BORDER_CONSTANT, value=(32, 32, 32))
         for c in crops]
rows = [np.hstack(crops[i:i + 3]) for i in range(0, len(crops), 3)]
mw = max(r.shape[1] for r in rows)
rows = [cv2.copyMakeBorder(r, 0, 0, 0, mw - r.shape[1],
                           cv2.BORDER_CONSTANT, value=(32, 32, 32))
        for r in rows]
sheet = np.vstack(rows)
s = 2000 / sheet.shape[1]
cv2.imwrite(str(ROOT / "preview" / "spot.png"),
            cv2.resize(sheet, (2000, int(sheet.shape[0] * s))))
print("preview/spot.png")
