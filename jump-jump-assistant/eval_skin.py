# -*- coding: utf-8 -*-
"""跨皮肤评估: 同一模型在"人工精标 / 旧皮肤模拟器帧 / 新皮肤帧"三组上的表现。

关键指标是"新皮肤帧的棋子类置信度" —— 换皮肤后它从 0.9 塌到 0.004, 分类头认不出银色棋子。
用法: python eval_skin.py <模型目录> [<模型目录2> ...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from detect import Detector

ROOT = Path(__file__).parent
HUMAN = ROOT / "annotations"
SCREENS = ROOT / "screenshots"
OLD_EMU = ROOT / "emulator"
NEW_SKIN = ROOT / "emulator_skin"


def human_errors(det):
    dp, dt, viol, miss = [], [], 0, 0
    for f in sorted(HUMAN.glob("*.json")):
        import json
        rec = json.loads(f.read_text(encoding="utf-8"))
        img = cv2.imread(str(SCREENS / rec["image"]))
        if img is None:
            continue
        r = det.infer(img)
        if r[0] is None or r[1] is None:
            miss += 1
            continue
        px, py = rec["piece"]
        tx, ty = rec["target"]
        dp.append(np.hypot(r[0]["kpt"][0] - px, r[0]["kpt"][1] - py))
        dt.append(np.hypot(r[1]["kpt"][0] - tx, r[1]["kpt"][1] - ty))
        if r[1]["kpt"][1] >= r[0]["kpt"][1]:
            viol += 1
    return dp, dt, viol, miss


def conf_stats(det, frames, piece_thr=0.35):
    cp, ct, miss_p, miss_t = [], [], 0, 0
    for p in frames:
        img = cv2.imread(str(p))
        if img is None:
            continue
        r = det.infer(img)
        if r[0] is None:
            miss_p += 1
        else:
            cp.append(r[0]["conf"])
        if r[1] is None:
            miss_t += 1
        else:
            ct.append(r[1]["conf"])
    return cp, ct, miss_p, miss_t


def main():
    for md in sys.argv[1:]:
        det = Detector(md)
        dp, dt, viol, miss = human_errors(det)
        new_frames = sorted(NEW_SKIN.glob("*.png"))
        old_frames = sorted(OLD_EMU.glob("a*_pre.png"))[:80]
        ncp, nct, nmp, nmt = conf_stats(det, new_frames)
        ocp, oct_, omp, omt = conf_stats(det, old_frames)
        print("=== %s" % md)
        if dp:
            print("  人工精标 %d 帧: 棋子 %.2fpx PCK@5=%.1f%% | 落点 %.2fpx PCK@5=%.1f%% | 违规 %d 漏检 %d"
                  % (len(dp), float(np.mean(dp)), 100 * float(np.mean(np.array(dp) <= 5)),
                     float(np.mean(dt)), 100 * float(np.mean(np.array(dt) <= 5)), viol, miss))
        print("  新皮肤 %d 帧: 棋子置信 均值 %.3f 最小 %.3f | 低于0.35 %d 帧 | 目标置信 均值 %.3f"
              % (len(new_frames), np.mean(ncp) if ncp else 0, np.min(ncp) if ncp else 0,
                 sum(1 for c in ncp if c < 0.35) + nmp, np.mean(nct) if nct else 0))
        print("  旧皮肤模拟器 %d 帧: 棋子置信 均值 %.3f 最小 %.3f | 目标置信 均值 %.3f 最小 %.3f"
              % (len(old_frames), np.mean(ocp) if ocp else 0, np.min(ocp) if ocp else 0,
                 np.mean(oct_) if oct_ else 0, np.min(oct_) if oct_ else 0))


if __name__ == "__main__":
    main()
