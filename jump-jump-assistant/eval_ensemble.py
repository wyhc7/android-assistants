# -*- coding: utf-8 -*-
"""集成评估: 两个模型都跑, 每个类别各取置信度更高的那个。

动机: 两个模型强弱互补 —— 颜色增广版在"换皮肤后的棋子"上强(0.7 级), 基线在"药瓶类目标"
上强(0.89 vs 0.30)。实测两个难帧:
  药瓶目标帧    基线 0.89 / 颜色增广 0.30
  银色棋子帧    基线 检不出 / 颜色增广 0.75
集成在留出集上每一类都严格占优。

用法: python eval_ensemble.py <模型目录A> <模型目录B>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from detect import Detector
from eval_skin import HUMAN, SCREENS, NEW_SKIN, OLD_EMU, conf_stats

import json


def ensemble(det_a, det_b, img):
    ra, rb = det_a.infer(img), det_b.infer(img)
    out = {}
    for cls in (0, 1):
        cands = [r[cls] for r in (ra, rb) if r[cls] is not None]
        out[cls] = max(cands, key=lambda x: x["conf"]) if cands else None
    return out


def main():
    da, db = Detector(sys.argv[1]), Detector(sys.argv[2])
    print("集成: %s  +  %s" % (sys.argv[1], sys.argv[2]))

    dp, dt, viol, miss = [], [], 0, 0
    for f in sorted(HUMAN.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        img = cv2.imread(str(SCREENS / rec["image"]))
        if img is None:
            continue
        r = ensemble(da, db, img)
        if r[0] is None or r[1] is None:
            miss += 1
            continue
        px, py = rec["piece"]
        tx, ty = rec["target"]
        dp.append(np.hypot(r[0]["kpt"][0] - px, r[0]["kpt"][1] - py))
        dt.append(np.hypot(r[1]["kpt"][0] - tx, r[1]["kpt"][1] - ty))
        if r[1]["kpt"][1] >= r[0]["kpt"][1]:
            viol += 1
    print("  人工精标 %d 帧: 棋子 %.2fpx PCK@5=%.1f%% | 落点 %.2fpx PCK@5=%.1f%% | 违规 %d 漏检 %d"
          % (len(dp), np.mean(dp), 100 * np.mean(np.array(dp) <= 5),
             np.mean(dt), 100 * np.mean(np.array(dt) <= 5), viol, miss))

    for tag, frames in (("新皮肤", sorted(NEW_SKIN.glob("*.png"))),
                        ("旧皮肤模拟器", sorted(OLD_EMU.glob("a*_pre.png"))[:80])):
        cp, ct, mp, mt = [], [], 0, 0
        for p in frames:
            img = cv2.imread(str(p))
            if img is None:
                continue
            r = ensemble(da, db, img)
            if r[0] is None:
                mp += 1
            else:
                cp.append(r[0]["conf"])
            if r[1] is None:
                mt += 1
            else:
                ct.append(r[1]["conf"])
        print("  %s %d 帧: 棋子置信 均值 %.3f 最小 %.3f | 低于0.35 %d 帧 | 目标置信 均值 %.3f"
              % (tag, len(frames), np.mean(cp) if cp else 0, np.min(cp) if cp else 0,
                 sum(1 for c in cp if c < 0.35) + mp, np.mean(ct) if ct else 0))

    for f in ("emulator/r7/fail_pre_035.png", "emulator_skin/fail_post_010.png"):
        p = Path(f)
        if not p.exists():
            continue
        r = ensemble(da, db, cv2.imread(str(p)))
        print("  难帧 %s: piece=%s target=%s"
              % (p.name, "MISS" if r[0] is None else "%.2f" % r[0]["conf"],
                 "MISS" if r[1] is None else "%.2f" % r[1]["conf"]))


if __name__ == "__main__":
    main()
