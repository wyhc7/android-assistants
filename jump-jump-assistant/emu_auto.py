# -*- coding: utf-8 -*-
"""自动连跳压力测试: 一直跳到游戏失败为止, 全程记录。

判定成功 = 起跳后棋子仍被识别到(站在方块上)。
判定失败 = 连续 3 次截屏都识别不到棋子(坠落 / 已进入结算画面)。
用法: python emu_auto.py [最大跳数] [模型目录]
输出: emulator/auto_log.csv 与每跳的前后帧, 失败现场另存 fail_*.png
"""
import csv
import os
import sys
import time
from pathlib import Path

import cv2

import emu
from detect import Detector

MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MODEL_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else emu.MODEL
K = 1.38
LOG = emu.EMU / "auto_log.csv"
PIECE_CONF = float(os.environ.get("PIECE_CONF", "0.35"))
det = Detector(MODEL_DIR, (PIECE_CONF, 0.35))
# 可选集成: 再挂一个模型, 每个类别取置信度更高者(两个模型强弱互补, 见 eval_ensemble.py)
MODEL2 = os.environ.get("EMU_MODEL2")
det2 = Detector(MODEL2, (PIECE_CONF, 0.35)) if MODEL2 else None


def detect(img):
    r = det.infer(img)
    if det2 is None:
        return r
    r2 = det2.infer(img)
    out = {}
    for c in (0, 1):
        cands = [x for x in (r[c], r2[c]) if x is not None]
        out[c] = max(cands, key=lambda y: y["conf"]) if cands else None
    return out


def detect_with_retry(tag, tries=3, gap=0.5):
    """截屏+识别, 失败重试。返回 (img, res, path)。"""
    for i in range(tries):
        img = emu.capture(tag if i == 0 else "%s_r%d" % (tag, i))
        if img is None:
            time.sleep(gap)
            continue
        res = detect(img)
        if res[0] is not None:
            return img, res, emu.EMU / ("%s.png" % (tag if i == 0 else "%s_r%d" % (tag, i)))
        time.sleep(gap)
    return img, res, emu.EMU / ("%s_r%d.png" % (tag, tries - 1))


def stable_detect(tag, tries=8, gap=0.6, tol=8.0):
    """等待"稳定态"再起跳: 连续两帧的关键点一致(差 <= tol)才认为动画结束。

    重启/落地后游戏有入场与镜头动画, 此时棋子位置不可信 —— 实测在动画中起跳会直接失败。
    App 内也应做同样的稳定态判定。
    """
    prev = None
    img = res = None
    for i in range(tries):
        t = tag if i == 0 else "%s_w%d" % (tag, i)
        img = emu.capture(t)
        if img is not None:
            res = detect(img)
            if res[0] is not None and res[1] is not None:
                cur = (res[0]["kpt"][0], res[0]["kpt"][1], res[1]["kpt"][0], res[1]["kpt"][1])
                if prev is not None and max(abs(a - b) for a, b in zip(prev, cur)) <= tol:
                    return img, res
                prev = cur
            else:
                prev = None
        time.sleep(gap)
    return img, res


def main():
    rows = []
    fails = 0
    for i in range(1, MAX + 1):
        img, res = stable_detect("a%03d_pre" % i)
        if res is None or res[0] is None or res[1] is None:
            print("第 %d 跳: 起跳前识别不到目标(piece=%s target=%s), 结束"
                  % (i, res[0] is not None, res[1] is not None))
            fails += 1
            cv2.imwrite(str(emu.EMU / ("fail_pre_%03d.png" % i)), img)
            break
        p, t = res[0]["kpt"], res[1]["kpt"]
        d = ((t[0] - p[0]) ** 2 + (p[1] - t[1]) ** 2) ** 0.5
        ms = d * K
        emu.press(p[0], p[1], ms)
        time.sleep(1.7)
        img2, res2, post = detect_with_retry("a%03d_post" % i)
        ok = res2[0] is not None
        rows.append(dict(idx=i, dist=round(d, 1), press_ms=round(ms),
                         conf_piece=round(res[0]["conf"], 3), conf_target=round(res[1]["conf"], 3),
                         post_piece=ok, post_target=res2[1] is not None))
        print("第 %3d 跳: 跳距=%6.1fpx 按压=%4dms conf=%.2f/%.2f -> %s"
              % (i, d, ms, res[0]["conf"], res[1]["conf"], "落地" if ok else "失败"))
        if not ok:
            fails += 1
            cv2.imwrite(str(emu.EMU / ("fail_post_%03d.png" % i)), img2)
            print("  失败现场已保存 fail_post_%03d.png" % i)
            break

    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["idx", "dist", "press_ms", "conf_piece", "conf_target",
                            "post_piece", "post_target"])
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    ok = sum(1 for r in rows if r["post_piece"])
    print("\n=== 汇总 ===")
    print("总跳数 %d | 成功落地 %d | 失败 %d | 成功率 %.1f%%" % (n, ok, n - ok, 100.0 * ok / max(n, 1)))
    if rows:
        ds = [r["dist"] for r in rows]
        cp = [r["conf_piece"] for r in rows]
        ct = [r["conf_target"] for r in rows]
        print("跳距范围 %.0f–%.0fpx (均值 %.0f)" % (min(ds), max(ds), sum(ds) / len(ds)))
        print("置信度 棋子 %.2f–%.2f 均值 %.3f | 目标 %.2f–%.2f 均值 %.3f"
              % (min(cp), max(cp), sum(cp) / len(cp), min(ct), max(ct), sum(ct) / len(ct)))
    print("日志:", LOG)


if __name__ == "__main__":
    main()
