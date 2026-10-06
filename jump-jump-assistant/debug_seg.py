# -*- coding: utf-8 -*-
"""调试单张图的完整标注流程。"""
import sys

import cv2
import numpy as np

import auto_annotate as A

img = cv2.imread(sys.argv[1])
print("piece:", A.find_piece(img))
bg = A.bg_color(img)
diff = np.abs(img.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
print("diff max/mean:", diff.max(), round(float(diff.mean()), 1))
mask = (diff > 40).astype(np.uint8) * 255
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
for i in range(1, n):
    if stats[i, 4] > 3000:
        print("comp", stats[i].tolist())
cv2.imwrite("preview/debug_mask.png", mask)
