# -*- coding: utf-8 -*-
"""全量评估: NCNN 模型预测 vs 人工标注(71 张全量)。

用法: python eval_all.py [模型目录]
输出: 棋子/落点像素误差分布, 以及"目标落在棋子下方"的违规帧数(必须为 0)。
"""
import json
import statistics as st
import sys
from pathlib import Path

import cv2

from detect import Detector

ROOT = Path(__file__).parent
MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "jump_pose-3" / "weights" / "best_ncnn_model"


def main():
    det = Detector(MODEL)
    err_p, err_t, miss_p, miss_t, bad_sign = [], [], 0, 0, []
    for ap in sorted((ROOT / "annotations").glob("*.json")):
        ann = json.loads(ap.read_text(encoding="utf-8"))
        img = cv2.imread(str(ROOT / "screenshots" / ann["image"]))
        if img is None:
            continue
        res = det.infer(img)
        if res[0] is None:
            miss_p += 1
        else:
            ex, ey = res[0]["kpt"][0] - ann["piece"][0], res[0]["kpt"][1] - ann["piece"][1]
            err_p.append((ex * ex + ey * ey) ** 0.5)
        if res[1] is None:
            miss_t += 1
        else:
            ex, ey = res[1]["kpt"][0] - ann["target"][0], res[1]["kpt"][1] - ann["target"][1]
            err_t.append((ex * ex + ey * ey) ** 0.5)
            if res[1]["kpt"][1] >= res[0]["kpt"][1] if res[0] else False:
                bad_sign.append(ann["image"][:30])
    n = len(err_p) + miss_p
    print("模型:", MODEL)
    print("样本: %d 张" % n)
    for name, errs, miss in (("棋子", err_p, miss_p), ("落点", err_t, miss_t)):
        if errs:
            pck5 = sum(1 for e in errs if e <= 5) / len(errs) * 100
            pck10 = sum(1 for e in errs if e <= 10) / len(errs) * 100
            print("%s: 漏检 %d | n=%d 平均 %.2fpx 中位 %.2fpx 最大 %.2fpx | PCK@5px %.1f%% PCK@10px %.1f%%"
                  % (name, miss, len(errs), sum(errs) / len(errs), st.median(errs), max(errs),
                     pck5, pck10))
        else:
            print("%s: 全部漏检(%d)" % (name, miss))
    print("目标落在棋子下方的违规帧: %d %s" % (len(bad_sign), bad_sign))


if __name__ == "__main__":
    main()
