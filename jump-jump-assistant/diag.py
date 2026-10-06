# -*- coding: utf-8 -*-
"""渲染单张图的标注叠加(全尺寸), 用于精确诊断标注对错。"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parent
SRC, ANN = ROOT / "screenshots", ROOT / "annotations"

for name in sys.argv[1:]:
    f = sorted(ANN.glob(f"*{name}*.json"))[0]
    rec = json.loads(f.read_text(encoding="utf-8"))
    img = cv2.imread(str(SRC / rec["image"]))
    px, py = [int(v) for v in rec["piece"]]
    x, y, w, h = rec["target_box"]
    tx, ty = [int(v) for v in rec["target"]]
    cv2.circle(img, (px, py), 14, (0, 0, 255), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv2.drawMarker(img, (tx, ty), (255, 0, 255), cv2.MARKER_CROSS, 50, 5)
    out = ROOT / "preview" / f"diag_{f.stem[:20]}.png"
    cv2.imwrite(str(out), img)
    print(out, "piece", rec["piece"], "box", rec["target_box"])
