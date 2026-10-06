# -*- coding: utf-8 -*-
"""正式玩法循环(带降噪): 静置 -> 多帧平均关键点 -> 按压 -> 用分数增量判定命中。

相对早期版本的三处降噪:
  1) 落地后先静置 settle 秒, 躲开相机缓动尾巴(相机没停稳时关键点会随帧漂移)
  2) 连续采 frames 帧, 只保留彼此一致(<= tol)的帧, 再对关键点取平均 -> 定位噪声降 ~1/sqrt(n)
  3) 分数连续两次读数一致才算(带滚动动画)
判定: 分数增量 +1 = 普通落地; +2,4,6... = 命中中心(连续命中递增)

用法: python play.py <模型目录> <最大跳数> <k> [目标分数]
"""
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

import emu
from calib import find_restart_button, game_over, merge_ensemble, stable_score
from detect import Detector
from emu import capture, press
from score_ocr import read

ROOT = Path(__file__).parent
OUT = ROOT / "emulator" / os.environ.get("EMU_SUBDIR", "play")
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "play.csv"
PIECE_CONF = float(os.environ.get("PIECE_CONF", "0.15"))
SETTLE = float(os.environ.get("SETTLE", "0.7"))
FRAMES = int(os.environ.get("FRAMES", "2"))
TOL = float(os.environ.get("TOL", "6"))
FIELD = ["idx", "d", "press", "score_pre", "score_post", "gain", "hit",
         "piece_conf", "target_conf", "frames_used"]


def valid_gain(g):
    """分数增量只可能是 1(普通落地) 或 2,4,...32(连续命中); 其它值说明读到了滚动中的数字。"""
    return g is not None and (g == 1 or (g >= 2 and g <= 32 and g % 2 == 0))


def detect(img):
    r = emu.det.infer(img)
    det2 = getattr(emu, "det2", None)
    if det2 is None:
        return r
    return merge_ensemble([r, det2.infer(img)])


def measure(tag):
    """采 FRAMES 帧, 取一致的帧平均关键点。返回 (代表帧, 平均后结果, 采用帧数, 最后一帧)。"""
    time.sleep(SETTLE)
    pts, res_list, imgs = [], [], []
    for t in range(FRAMES):
        img = capture("%s_f%d" % (tag, t))
        if img is None:
            continue
        r = detect(img)
        imgs.append(img)
        if r[0] is not None and r[1] is not None and r[1]["kpt"][1] < r[0]["kpt"][1]:
            pts.append((r[0]["kpt"][0], r[0]["kpt"][1], r[1]["kpt"][0], r[1]["kpt"][1]))
            res_list.append(r)
        if t + 1 < FRAMES:
            time.sleep(0.15)
    if not pts:
        return None, None, 0, (imgs[-1] if imgs else None)
    ref = np.median(np.array(pts), axis=0)
    keep = [p for p in pts if max(abs(a - b) for a, b in zip(p, ref)) <= TOL]
    if len(keep) < 2:
        keep = pts                       # 全都漂移时退化为取中位
    arr = np.array(keep)
    med = np.median(arr, axis=0)
    best = min(res_list, key=lambda r: abs(r[0]["kpt"][0] - med[0]))
    res = {0: dict(best[0]), 1: dict(best[1])}
    res[0]["kpt"] = [float(med[0]), float(med[1]), best[0]["kpt"][2]]
    res[1]["kpt"] = [float(med[2]), float(med[3]), best[1]["kpt"][2]]
    return imgs[-1], res, len(keep), imgs[-1]


def write(row):
    if not LOG.exists():
        with LOG.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELD).writeheader()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELD).writerow(row)


def main():
    model, n, k = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    target = float(sys.argv[4]) if len(sys.argv) > 4 else 3000.0
    emu.det = Detector(model, (PIECE_CONF, emu.TARGET_CONF))
    emu.det_lo = Detector(model, (PIECE_CONF, emu.RELAX_TARGET_CONF))
    if os.environ.get("EMU_MODEL2"):
        emu.det2 = Detector(os.environ["EMU_MODEL2"], (PIECE_CONF, emu.TARGET_CONF))
        emu.det2_lo = Detector(os.environ["EMU_MODEL2"], (PIECE_CONF, emu.RELAX_TARGET_CONF))
    print("玩法循环: k=%.3f | 静置 %.1fs | 平均 %d 帧 | 目标 %d 分" % (k, SETTLE, FRAMES, target))
    hits = miss = 0
    best = 0
    fails = 0
    games = []
    for i in range(1, n + 1):
        img, res, used, last = measure("a%03d" % i)
        if res is None:
            over, oimg = game_over()
            if over:
                games.append(best)
                print("游戏结束: 本局分数 %d | 已跳 %d 跳 | 命中率 %.1f%%"
                      % (best, hits + miss, 100.0 * hits / max(hits + miss, 1)))
                if os.environ.get("AUTO_RESTART") == "1":
                    btn = find_restart_button(oimg)
                    press(btn[0], btn[1], 60)
                    print("重开一局")
                    best = 0
                    time.sleep(6.0)
                    continue
                break
            fails += 1
            print("第 %3d 跳: 识别失败(用了 %d 帧, %d/6), 重试" % (i, used, fails))
            if fails >= 6:
                print("连续 6 跳识别失败, 停止")
                break
            time.sleep(2.5)
            continue
        fails = 0
        p, t = res[0]["kpt"], res[1]["kpt"]
        d = ((t[0] - p[0]) ** 2 + (p[1] - t[1]) ** 2) ** 0.5
        # 分数直接读起跳帧, 省掉一次截图; 只有增量不合理时才回到稳定读数
        s_pre = read(img)
        ms = k * d
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
        if hit:
            hits += 1
        elif hit is False:
            miss += 1
        if s_post:
            best = max(best, s_post)
        write({"idx": i, "d": round(d, 1), "press": round(ms, 1), "score_pre": s_pre,
               "score_post": s_post, "gain": gain,
               "hit": None if hit is None else int(hit),
               "piece_conf": round(res[0]["conf"], 3), "target_conf": round(res[1]["conf"], 3),
               "frames_used": used})
        print("第 %3d 跳: d=%6.1f 按压=%4dms | %s->%s (%s) %s | 命中率 %.1f%% (%d/%d)"
              % (i, d, ms, s_pre, s_post, gain,
                 "" if hit is None else ("命中" if hit else "普通"),
                 100.0 * hits / max(hits + miss, 1), hits, hits + miss))
        if best >= target:
            print("达成目标: %d >= %d (跳数 %d, 命中率 %.1f%%)" % (best, target, i, 100.0 * hits / max(hits + miss, 1)))
            break
    games.append(best)
    print("各局最高分: %s | 总跳数 %d | 命中 %d 普通 %d 命中率 %.1f%%"
          % (games, hits + miss, hits, miss, 100.0 * hits / max(hits + miss, 1)))


if __name__ == "__main__":
    main()
