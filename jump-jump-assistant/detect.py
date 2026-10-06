# -*- coding: utf-8 -*-
"""桌面参考实现: 与安卓端完全一致的推理链(NCNN + 居中letterbox + 解码 + 坐标回映)。

用法: python detect.py [模型目录] [图片目录] [输出目录]
默认: runs/jump_pose-3/weights/best_ncnn_model  dataset/val/images  preview/detect
输出: 每张图的检测结果(piece 底部中心 / target 落点)与叠加可视化。
"""
import sys
from pathlib import Path

import cv2
import ncnn
import numpy as np

ROOT = Path(__file__).parent
MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "jump_pose-3" / "weights" / "best_ncnn_model"
SRC = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "dataset" / "val" / "images"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "preview" / "detect"
IMGSZ, PAD = 640, 114
CONF, IOU = 0.35, 0.5


def letterbox(img):
    h, w = img.shape[:2]
    s = min(IMGSZ / w, IMGSZ / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = np.full((IMGSZ, IMGSZ, 3), PAD, np.uint8)
    ox, oy = (IMGSZ - nw) // 2, (IMGSZ - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh))
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)), s, ox, oy


def nms(boxes, scores, iou_thr=IOU):
    keep = []
    order = scores.argsort()[::-1]
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                + (boxes[order[1:], 2] - boxes[order[1:], 0])
                * (boxes[order[1:], 3] - boxes[order[1:], 1]) - inter)
        order = order[1:][inter / np.maximum(area, 1e-9) <= iou_thr]
    return keep


class Detector:
    """与安卓端逐行对应的推理实现。"""

    def __init__(self, model_dir, conf=CONF):
        """conf: 阈值, 可为 (piece阈值, target阈值) 以分别控制两类。"""
        self.conf = (conf, conf) if isinstance(conf, (int, float)) else tuple(conf)
        self.net = ncnn.Net()
        self.net.opt.num_threads = 4
        self.net.load_param(str(Path(model_dir) / "model.ncnn.param"))
        self.net.load_model(str(Path(model_dir) / "model.ncnn.bin"))

    def infer(self, img):
        chw, s, ox, oy = letterbox(img)
        ex = self.net.create_extractor()
        ex.input("in0", ncnn.Mat(chw).clone())
        _, out = ex.extract("out0")
        o = np.array(out)
        if o.shape[0] != 9:          # 输出 9 = 4框 + 2类 + 3关键点
            o = o.T
        cls = o[4:6].T               # (8400, 2) 已 sigmoid
        kpt = o[6:9].T               # (8400, 3) x,y 为 640 空间像素, 第3列置信度
        cx, cy, bw, bh = o[0], o[1], o[2], o[3]
        best_cls = cls.argmax(1)
        best_conf = cls.max(1)
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        result = {0: None, 1: None}
        for c in (0, 1):
            m = (best_cls == c) & (best_conf > self.conf[c])
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            k = nms(xyxy[idx], best_conf[idx])[0]
            i = idx[k]
            result[c] = {
                "conf": float(best_conf[i]),
                # 坐标回映: 减 padding 再除以缩放
                "box": [(float(xyxy[i, 0]) - ox) / s, (float(xyxy[i, 1]) - oy) / s,
                        (float(xyxy[i, 2]) - ox) / s, (float(xyxy[i, 3]) - oy) / s],
                "kpt": [(float(kpt[i, 0]) - ox) / s, (float(kpt[i, 1]) - oy) / s,
                        float(kpt[i, 2])],
            }
        return result


    def candidates(self, img, min_conf=0.05):
        """返回全部高于 min_conf 的原始候选(未按类别取最优), 用于诊断漏检/找脚下方块。

        每条: {"cls": 0|1, "conf": float, "kpt": [x, y, kconf], "box": [x1,y1,x2,y2]}
        坐标已回映到原图。
        """
        chw, s, ox, oy = letterbox(img)
        ex = self.net.create_extractor()
        ex.input("in0", ncnn.Mat(chw).clone())
        _, out = ex.extract("out0")
        o = np.array(out)
        if o.shape[0] != 9:
            o = o.T
        cls = o[4:6].T
        kpt = o[6:9].T
        cx, cy, bw, bh = o[0], o[1], o[2], o[3]
        best_cls = cls.argmax(1)
        best_conf = cls.max(1)
        res = []
        for i in np.nonzero(best_conf > min_conf)[0]:
            res.append({
                "cls": int(best_cls[i]),
                "conf": float(best_conf[i]),
                "kpt": [(float(kpt[i, 0]) - ox) / s, (float(kpt[i, 1]) - oy) / s,
                        float(kpt[i, 2])],
                "box": [(float(cx[i] - bw[i] / 2) - ox) / s, (float(cy[i] - bh[i] / 2) - oy) / s,
                        (float(cx[i] + bw[i] / 2) - ox) / s, (float(cy[i] + bh[i] / 2) - oy) / s],
            })
        res.sort(key=lambda r: -r["conf"])
        return res


def draw(img, res):
    out = img.copy()
    for c, col, name in ((0, (0, 0, 255), "piece"), (1, (0, 255, 0), "target")):
        r = res[c]
        if not r:
            continue
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3)
        kx, ky = int(r["kpt"][0]), int(r["kpt"][1])
        cv2.drawMarker(out, (kx, ky), (255, 0, 255), cv2.MARKER_CROSS, 40, 4)
        cv2.putText(out, "%s %.2f" % (name, r["conf"]), (x1, max(30, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
    if res[0] and res[1]:
        cv2.line(out, (int(res[0]["kpt"][0]), int(res[0]["kpt"][1])),
                 (int(res[1]["kpt"][0]), int(res[1]["kpt"][1])), (255, 255, 0), 3)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    det = Detector(MODEL)
    files = sorted(p for p in SRC.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    ok = 0
    for p in files:
        img = cv2.imread(str(p))
        res = det.infer(img)
        cv2.imwrite(str(OUT / p.name), draw(img, res))
        tag = ""
        if res[0] and res[1]:
            ok += 1
            dx = res[1]["kpt"][0] - res[0]["kpt"][0]
            dy = res[0]["kpt"][1] - res[1]["kpt"][1]
            tag = "距离=%.0fpx (横向%.0f 纵向%.0f)" % ((dx * dx + dy * dy) ** 0.5, dx, dy)
        print("%-42s piece=%-5s target=%-5s %s" % (
            p.name[:42],
            "%.2f" % res[0]["conf"] if res[0] else "-",
            "%.2f" % res[1]["conf"] if res[1] else "-", tag))
    print("\n两目标齐全 %d/%d, 可视化输出: %s" % (ok, len(files), OUT))


if __name__ == "__main__":
    main()
