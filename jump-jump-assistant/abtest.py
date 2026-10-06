# -*- coding: utf-8 -*-
"""k 值 A/B 试验: 同一局里随机交替两个按压系数, 用分数增量判定是否命中中心。

为什么这么做: 图像量测落点误差不可靠(泛洪分割随种子漂移、模板匹配在平坦顶面上退化),
而分数增量是**游戏亲口给的真值**: +1 = 普通落地, +2,4,6... = 命中中心。
两个 k 在同一局里交替, 跳距分布相同, 于是可以直接配对比较命中率, 无偏。

用法:
  python abtest.py <模型目录> <跳数> <臂1> <臂2> [...] [目标分数]
  臂格式 "k" 或 "k:A" (按压 ms = k*d + A, A 用于补偿按压链路的固定开销)
  同局内各臂随机交替, 跳距分布相同 -> 命中率可直接配对比较
结果: emulator/<subdir>/ab.csv, 结束打印每臂总命中率与分跳距命中率
"""
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

import emu
from calib import detect, find_restart_button, game_over, stable_detect, stable_score, OUT
from emu import capture, press
from detect import Detector
from score_ocr import read

PIECE_CONF = float(os.environ.get("PIECE_CONF", "0.15"))
LOG = OUT / "ab.csv"
FIELD = ["idx", "arm", "k", "a", "d", "press", "score_pre", "score_post", "gain", "hit"]


def valid_gain(g):
    """分数增量只可能是 1(普通落地) 或 2,4,...32(连续命中中心)。"""
    return g is not None and (g == 1 or (g >= 2 and g <= 32 and g % 2 == 0))


def parse_arms(args):
    arms = []
    for spec in args:
        if ":" in spec:
            k, a = spec.split(":")
            arms.append((float(k), float(a)))
        else:
            arms.append((float(spec), 0.0))
    return arms


def main():
    model, n = sys.argv[1], int(sys.argv[2])
    rest = sys.argv[3:]
    target = 3000.0
    if rest and ":" not in rest[-1] and len(rest) > 1 and rest[-1].replace(".", "").isdigit() and float(rest[-1]) >= 100:
        target = float(rest[-1])
        rest = rest[:-1]
    arms = parse_arms(rest)
    emu.det = Detector(model, (PIECE_CONF, emu.TARGET_CONF))
    emu.det_lo = Detector(model, (PIECE_CONF, emu.RELAX_TARGET_CONF))
    if os.environ.get("EMU_MODEL2"):
        emu.det2 = Detector(os.environ["EMU_MODEL2"], (PIECE_CONF, emu.TARGET_CONF))
        emu.det2_lo = Detector(os.environ["EMU_MODEL2"], (PIECE_CONF, emu.RELAX_TARGET_CONF))
    print("臂: %s | 按压方式 %s" % (["k=%.3f A=%+.0f" % x for x in arms], os.environ.get("PRESS_MODE", "swipe")))
    rows = []
    best = 0
    fails = 0
    games = []
    cur = None
    for i in range(1, n + 1):
        img, res = stable_detect("a%03d_pre" % i)
        if res is None:
            over, oimg = game_over()
            if over:
                games.append(cur if cur is not None else best)
                print("\n游戏结束: 本局分数 %s | 已跳 %d 跳 | 命中率 %.1f%%"
                      % (cur, len(rows), 100.0 * sum(r["hit"] for r in rows if r["hit"] is not None)
                         / max(sum(1 for r in rows if r["hit"] is not None), 1)))
                if os.environ.get("AUTO_RESTART") == "1":
                    btn = find_restart_button(oimg)
                    press(btn[0], btn[1], 60)
                    print("重开一局")
                    cur = None
                    time.sleep(6.0)
                    if best >= target:
                        print("已达成目标分数 %d >= %d, 停止" % (best, target))
                        break
                    continue
                break                      # 默认: 结算即停, 不再自动续局
            fails += 1
            print("第 %3d 跳: 识别失败(%d/3), 可能是截图坏帧, 重试" % (i, fails))
            if fails >= 3:
                print("连续 3 跳识别失败, 停止")
                break
            continue
        fails = 0
        # 兜底: 连续多跳命中率过低说明参数已经跑飞, 立即停, 别继续瞎打
        last = [r["hit"] for r in rows[-20:] if r["hit"] is not None]
        if len(last) >= 20 and sum(last) <= 3:
            print("最近 20 跳命中率仅 %.0f%%, 判定参数跑飞, 停止" % (100.0 * sum(last) / len(last)))
            break
        p, t = res[0]["kpt"], res[1]["kpt"]
        d = ((t[0] - p[0]) ** 2 + (p[1] - t[1]) ** 2) ** 0.5
        ai = random.randrange(len(arms))
        k, a = arms[ai]
        arm = chr(ord("A") + ai)
        s_pre = read(img)                    # 直接用起跳帧读分数, 省一次截图
        ms = max(20.0, k * d + a)
        press(p[0], p[1], ms)
        time.sleep(1.7)
        img2 = capture("a%03d_post" % i)
        s_post = read(img2)
        if not valid_gain(None if None in (s_pre, s_post) else s_post - s_pre):
            s_pre2, _ = stable_score("%d" % i)
            s_post2, _ = stable_score("p%d" % i)
            if valid_gain(None if None in (s_pre2, s_post2) else s_post2 - s_pre2):
                s_pre, s_post = s_pre2, s_post2
        gain = (s_post - s_pre) if (s_pre is not None and s_post is not None) else None
        hit = None if gain is None else (gain >= 2 and gain % 2 == 0)
        row = {"idx": i, "arm": arm, "k": k, "a": a, "d": round(d, 1), "press": round(ms, 1),
               "score_pre": s_pre, "score_post": s_post, "gain": gain,
               "hit": None if hit is None else int(hit)}
        rows.append(row)
        write(row)
        if s_post is not None:
            cur = s_post
            best = max(best, s_post)
        print("第 %3d 跳 [%s k=%.3f A=%+.0f]: d=%6.1f 按压=%4dms 分数 %s->%s (%s) %s"
              % (i, arm, k, a, d, ms, s_pre, s_post, gain,
                 "" if hit is None else ("命中" if hit else "普通")))
        if best >= target:
            print("达成目标: 分数 %d >= %d" % (best, target))
            break
    if cur is not None:
        games.append(cur)
    print("各局结束分数: %s | 最高 %d" % (games, best))
    summarize(rows, arms)


def write(row):
    if not LOG.exists():
        with LOG.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELD).writeheader()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELD).writerow(row)


def summarize(rows, arms):
    rows = [r for r in rows if r["hit"] is not None and r["gain"] is not None]
    if not rows:
        print("没有有效样本")
        return
    print("\n=== A/B 汇总 ===")
    for ai, (k, a) in enumerate(arms):
        arm = chr(ord("A") + ai)
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        hits = sum(r["hit"] for r in sub)
        print("%s k=%.3f A=%+.0f: n=%3d 命中率 %.1f%%" % (arm, k, a, len(sub), 100.0 * hits / len(sub)))
        for lo, hi in ((0, 250), (250, 400), (400, 550), (550, 950)):
            b = [r for r in sub if lo <= r["d"] < hi]
            if len(b) < 3:
                continue
            print("    跳距 %3d-%3d: n=%2d 命中率 %3.0f%% 平均 d=%5.1f"
                  % (lo, hi, len(b), 100.0 * sum(x["hit"] for x in b) / len(b),
                     np.mean([x["d"] for x in b])))


if __name__ == "__main__":
    main()
