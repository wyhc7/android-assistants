# -*- coding: utf-8 -*-
"""验证 NCNN 导出模型: 输入为 CHW 连续内存, 并确定 letterbox 约定。"""
import glob

import cv2
import ncnn
import numpy as np

PARAM = "runs/jump_pose-2/weights/best_ncnn_model/model.ncnn.param"
BIN = "runs/jump_pose-2/weights/best_ncnn_model/model.ncnn.bin"


def build(img, center):
    h, w = img.shape[:2]
    s = min(640 / w, 640 / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    r = cv2.resize(img, (nw, nh))
    c = np.full((640, 640, 3), 114, np.uint8)
    ox, oy = ((640 - nw) // 2, (640 - nh) // 2) if center else (0, 0)
    c[oy:oy + nh, ox:ox + nw] = r
    rgb = c[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)), (s, ox, oy)


def run(chw):
    net = ncnn.Net()
    net.load_param(PARAM)
    net.load_model(BIN)
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(chw).clone())
    _, out = ex.extract("out0")
    o = np.array(out)
    if o.shape[0] != 9:
        o = o.T
    cls = o[4:6].T
    i = int(np.argmax(cls.max(axis=1)))
    return cls[i].round(3).tolist(), o[:4, i].round(1).tolist(), o[6:9, i].round(1).tolist()


img = cv2.imread(sorted(glob.glob("dataset/val/images/*.jpg"))[0])
for center in (False, True):
    chw, meta = build(img, center)
    cls, box, kpt = run(chw)
    print("center=%-5s scale=%.4f off=(%d,%d) cls=%s box=%s kpt=%s"
          % (center, meta[0], meta[1], meta[2], cls, box, kpt), flush=True)
