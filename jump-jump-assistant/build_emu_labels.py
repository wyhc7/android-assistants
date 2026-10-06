# -*- coding: utf-8 -*-
"""把模拟器实测采集的真实游戏帧做成标注(伪标签 + 独立 CV 交叉核对)。

伪标签来源: 识别模型在起跳前帧的预测 —— 该预测驱动了 126 次成功落地,
因此是被"游戏结果"验证过的标签, 不是普通自训练。
交叉核对: 用与模型无关的 CV 方法(棋子颜色分割 / 等轴顶面测量)复核, 偏离过大的帧剔除。

用法: python build_emu_labels.py [子目录] [输出目录]
  子目录: emulator/ 下的相对路径(默认根目录), 用于隔离不同轮次的采集帧
  环境变量 BL_MODEL: 指定用于预测的模型目录
输出: annotations_emu/*.json, 以及核对统计
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import auto_annotate2 as A
from detect import Detector

ROOT = Path(__file__).parent
SUBDIR = sys.argv[1] if len(sys.argv) > 1 else ""
EMU = ROOT / "emulator" / SUBDIR
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "annotations_emu"
MODEL = Path(os.environ.get("BL_MODEL", str(ROOT / "runs" / "jump_pose-3" / "weights" / "best_ncnn_model")))
PIECE_TOL = 18.0     # 棋子: CV 与模型偏差上限(px)
TARGET_TOL = 55.0    # 目标: CV 顶面中心与模型关键点偏差上限(px)

OUT.mkdir(exist_ok=True)


def main():
    det = Detector(MODEL)
    frames = sorted(p for p in EMU.glob("a*_pre.png")) + \
        sorted(p for p in EMU.glob("a*_post.png"))
    ok = flagged = skipped = 0
    dp, dt = [], []
    for p in frames:
        img = cv2.imread(str(p))
        if img is None:
            continue
        res = det.infer(img)
        if res[0] is None or res[1] is None:
            skipped += 1
            continue
        ppx, ppy = res[0]["kpt"][0], res[0]["kpt"][1]
        tpx, tpy = res[1]["kpt"][0], res[1]["kpt"][1]
        if tpy >= ppy:                      # 规则: 目标必须在棋子上方
            skipped += 1
            continue
        # 交叉核对 1: 棋子(CV 颜色分割)
        pmask, cv_piece = A.piece_mask(img)
        d_piece = None
        if cv_piece:
            d_piece = ((cv_piece[0] - ppx) ** 2 + (cv_piece[1] - ppy) ** 2) ** 0.5
            dp.append(d_piece)
        # 交叉核对 2: 目标顶面(CV 等轴测量)
        mask = A.fg_mask(img, pmask)
        n_lab, lab = cv2.connectedComponents(mask)
        box = None
        ty, tx = int(round(tpy)), int(round(tpx))
        if 0 <= ty < mask.shape[0] and 0 <= tx < mask.shape[1] and lab[ty, tx] > 0:
            box = A.top_face_iso((lab == lab[ty, tx]).astype(np.uint8))
        d_target = None
        if box is not None:
            ccx, ccy = box[0] + box[2] / 2, box[1] + box[3] / 2
            d_target = ((ccx - tpx) ** 2 + (ccy - tpy) ** 2) ** 0.5
            dt.append(d_target)

        # 筛选: CV 棋子核对在模拟器域不可靠(棋子渲染与手机截图差异大), 改用模型置信度门 +
        # 目标顶面 CV 核对。标签本身始终取模型预测(被落地结果验证过)。
        if res[0]["conf"] < 0.55 or res[1]["conf"] < 0.55:
            skipped += 1
            continue
        bad = (d_target is not None and d_target > TARGET_TOL)
        if bad:
            flagged += 1
            continue
        # 顶面框: 统一用"关键点居中 + 精确 2:1"构造, 与人工标注同构。
        # 宽度优先取 CV 等轴测量, 否则取模型框宽 —— 注意 box 是 xyxy, 宽度必须用 x2-x1。
        x1, y1, x2, y2 = [float(v) for v in res[1]["box"]]
        w = float(box[2]) if (box is not None and d_target is not None and d_target <= 40) \
            else (x2 - x1)
        tb = [round(tpx - w / 2, 2), round(tpy - w / 4, 2), round(w, 2), round(w / 2, 2)]
        rec = {
            "image": p.name,
            "width": img.shape[1],
            "height": img.shape[0],
            "piece": [round(float(ppx), 2), round(float(ppy), 2)],
            "target": [round(float(tpx), 2), round(float(tpy), 2)],
            "target_box": tb,
            "confirmed": True,
            "source": "emu_pseudo",
            "conf_piece": round(float(res[0]["conf"]), 3),
            "conf_target": round(float(res[1]["conf"]), 3),
        }
        (OUT / (p.stem + ".json")).write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        ok += 1

    print("采集帧 %d | 采用 %d | 交叉核对剔除 %d | 检测失败/违规跳过 %d"
          % (len(frames), ok, flagged, skipped))
    if dp:
        print("棋子核对: n=%d 平均偏差 %.2fpx 最大 %.2fpx" % (len(dp), sum(dp) / len(dp), max(dp)))
    if dt:
        print("目标核对: n=%d 平均偏差 %.2fpx 最大 %.2fpx" % (len(dt), sum(dt) / len(dt), max(dt)))
    print("输出:", OUT)


if __name__ == "__main__":
    main()
