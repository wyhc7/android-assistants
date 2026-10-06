# -*- coding: utf-8 -*-
"""渲染指定标注目录的对照表。用法: python sheet_labels.py <标注目录> <图片目录> <输出前缀> [缩略宽] [列] [每表]"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
ANN = Path(sys.argv[1])
SRC = Path(sys.argv[2])
PREFIX = sys.argv[3]
TW = int(sys.argv[4]) if len(sys.argv) > 4 else 300
COLS = int(sys.argv[5]) if len(sys.argv) > 5 else 6
PER = int(sys.argv[6]) if len(sys.argv) > 6 else 36

files = sorted(ANN.glob("*.json"))
for si in range(0, len(files), PER):
    thumbs = []
    for f in files[si:si + PER]:
        rec = json.loads(f.read_text(encoding="utf-8"))
        img = cv2.imread(str(SRC / rec["image"]))
        if img is None:
            continue
        px, py = [int(v) for v in rec["piece"]]
        tx, ty = [int(v) for v in rec["target"]]
        x, y, w, h = [int(v) for v in rec["target_box"]]
        cv2.circle(img, (px, py), 12, (0, 0, 255), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.drawMarker(img, (tx, ty), (255, 0, 255), cv2.MARKER_CROSS, 44, 5)
        thumbs.append(cv2.resize(img, (TW, int(img.shape[0] * TW / img.shape[1]))))
    if not thumbs:
        continue
    th, tw = thumbs[0].shape[:2]
    rows = math.ceil(len(thumbs) / COLS)
    sheet = np.full((rows * th, COLS * tw, 3), 24, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, COLS)
        sheet[r * th:r * th + t.shape[0], c * tw:(c + 1) * tw] = t
    out = ROOT / "preview" / ("%s_%d.png" % (PREFIX, si // PER + 1))
    cv2.imwrite(str(out), sheet)
    print(out, sheet.shape)
