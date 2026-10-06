# -*- coding: utf-8 -*-
"""把全部标注结果渲染成分批对照图(缩略图宽度可辨), 便于逐张自查。"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
SRC, ANN, OUT = ROOT / "screenshots", ROOT / "annotations", ROOT / "preview"
OUT.mkdir(exist_ok=True)
THUMB_W = int(sys.argv[1]) if len(sys.argv) > 1 else 360
COLS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
PER = int(sys.argv[3]) if len(sys.argv) > 3 else 24

files = sorted(ANN.glob("*.json"))
for si in range(0, len(files), PER):
    batch = files[si:si + PER]
    thumbs = []
    for f in batch:
        rec = json.loads(f.read_text(encoding="utf-8"))
        img = cv2.imread(str(SRC / rec["image"]))
        if img is None:
            continue
        px, py = [int(v) for v in rec["piece"]]
        x, y, w, h = rec["target_box"]
        tx, ty = [int(v) for v in rec["target"]]
        cv2.circle(img, (px, py), 14, (0, 0, 255), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 6)
        cv2.drawMarker(img, (tx, ty), (255, 0, 255), cv2.MARKER_CROSS, 50, 6)
        cv2.putText(img, f"{si + len(thumbs) + 1}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 10)
        cv2.putText(img, f"{si + len(thumbs) + 1}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 4)
        scale = THUMB_W / img.shape[1]
        thumbs.append(cv2.resize(img, (THUMB_W, int(img.shape[0] * scale))))
    if not thumbs:
        continue
    th, tw = thumbs[0].shape[:2]
    rows = math.ceil(len(thumbs) / COLS)
    sheet = np.full((rows * th, COLS * tw, 3), 24, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, COLS)
        sheet[r * th:r * th + t.shape[0], c * tw:(c + 1) * tw] = t
    p = OUT / f"sheet_{si // PER + 1}.png"
    cv2.imwrite(str(p), sheet)
    print(p, sheet.shape)
