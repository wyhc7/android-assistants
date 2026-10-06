# -*- coding: utf-8 -*-
"""把模型预测叠加图拼成分批对照表。用法: python sheet_pred.py <目录> [缩略宽] [列数] [每表张数]"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "preview" / "detect_final"
TW = int(sys.argv[2]) if len(sys.argv) > 2 else 300
COLS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
PER = int(sys.argv[4]) if len(sys.argv) > 4 else 36
OUT = ROOT / "preview"

files = sorted(p for p in SRC.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
for si in range(0, len(files), PER):
    thumbs = []
    for p in files[si:si + PER]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        thumbs.append(cv2.resize(img, (TW, int(img.shape[0] * TW / img.shape[1]))))
    if not thumbs:
        continue
    th, tw = thumbs[0].shape[:2]
    rows = math.ceil(len(thumbs) / COLS)
    sheet = np.full((rows * th, COLS * tw, 3), 24, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, COLS)
        sheet[r * th:r * th + t.shape[0], c * tw:(c + 1) * tw] = t
    out = OUT / ("pred_%d.png" % (si // PER + 1))
    cv2.imwrite(str(out), sheet)
    print(out, sheet.shape)
