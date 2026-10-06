# -*- coding: utf-8 -*-
"""生成标注叠加预览图(缩略拼图), 便于快速人工复核。"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
SRC, ANN = ROOT / "screenshots", ROOT / "annotations"
OUT = ROOT / "preview"
OUT.mkdir(exist_ok=True)
THUMB_W, COLS = 270, 5

thumbs = []
for f in sorted(ANN.glob("*.json")):
    rec = json.loads(f.read_text(encoding="utf-8"))
    img = cv2.imread(str(SRC / rec["image"]))
    if img is None:
        continue
    px, py = [int(v) for v in rec["piece"]]
    x, y, w, h = rec["target_box"]
    cv2.circle(img, (px, py), 10, (0, 0, 255), -1)          # 棋子红点
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)  # 目标绿框
    tx, ty = [int(v) for v in rec["target"]]
    cv2.drawMarker(img, (tx, ty), (255, 0, 255),
                   cv2.MARKER_CROSS, 30, 3)                  # 落点十字
    scale = THUMB_W / img.shape[1]
    thumbs.append(cv2.resize(img, (THUMB_W, int(img.shape[0] * scale))))

if thumbs:
    th, tw = thumbs[0].shape[:2]
    rows = (len(thumbs) + COLS - 1) // COLS
    sheet = np.full((rows * th, COLS * tw, 3), 32, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, COLS)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    cv2.imwrite(str(OUT / "sheet.png"), sheet)
    print(f"preview/sheet.png  {len(thumbs)} 张")
