# -*- coding: utf-8 -*-
"""原始候选探针: 打印所有 anchor 的类别分与关键点, 用于判断"漏检"是分数低还是真没学到。

用法: python probe_cand.py <模型目录> <图片> [最低分]
"""
import sys
from pathlib import Path

import cv2
import ncnn
import numpy as np

from detect import letterbox

model_dir, img_path = sys.argv[1], sys.argv[2]
min_conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

net = ncnn.Net()
net.opt.num_threads = 4
net.load_param(str(Path(model_dir) / "model.ncnn.param"))
net.load_model(str(Path(model_dir) / "model.ncnn.bin"))

img = cv2.imread(img_path)
chw, s, ox, oy = letterbox(img)
ex = net.create_extractor()
ex.input("in0", ncnn.Mat(chw).clone())
_, out = ex.extract("out0")
o = np.array(out)
if o.shape[0] != 9:
    o = o.T
cls = o[4:6].T
kpt = o[6:9].T
best = cls.argmax(1)
conf = cls.max(1)

idx = np.argsort(-conf)[:12]
print("图片", img_path, img.shape)
print("%-6s %-8s %-10s %-18s" % ("rank", "class", "score", "keypoint(原图坐标)"))
for r, i in enumerate(idx):
    if conf[i] < min_conf:
        break
    kx, ky, kc = kpt[i]
    ox_ = (kx - ox) / s
    oy_ = (ky - oy) / s
    print("%-6d %-8d %-10.4f (%.0f, %.0f) kconf=%.2f" % (r, best[i], conf[i], ox_, oy_, kc))
