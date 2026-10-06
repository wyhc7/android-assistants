# -*- coding: utf-8 -*-
"""闭环标定: 用分数增量(命中中心 +2,4,6...)与落点偏移, 反解按压模型 press = A + K*d。

为什么要标定: 达到"命中中心率 90%"要求落点误差控制在约 ±30px(方块顶面宽约 370px),
而当前 k=1.38 只能保证落在大方块上。分数规则给了两个互补的真值:
  - 分数增量 = 1 -> 普通落地;  = 2,4,6... -> 命中中心(连续命中递增)  -> 二值真值
  - 落地帧里棋子底部 与 脚下方块顶面质心 的横向差 -> 误差的**方向与量级**(与命中强相关,
    实测命中组 |偏移| 中位 17px, 普通组 32px)

模型: 落点误差 = α*A + (α*K - 1)*d, 其中 α = 每毫秒位移(px/ms)。
  斜率 β = α*K - 1 与图像测量的常数偏置无关(偏置只进截距), 因此 **k 可以由斜率无偏反解**:
      K_new = K / (1 + β)
  截距含测量偏置, 只用它给 A 一个初值, 最终由命中率确认。

用法:
  python calib.py <模型目录> <跳数> [k] [a]
  python calib.py --fit <csv> [k]        只做回归并给出建议值
"""
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import emu
from emu import capture, press, sh
from score_ocr import read as read_score

ROOT = Path(__file__).parent
SUBDIR = os.environ.get("EMU_SUBDIR", "calib")
OUT = ROOT / "emulator" / SUBDIR
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "calib.csv"
# 等轴投影: 跳跃方向 (1,-0.5) 归一化后, 屏幕横向偏移需乘 1/0.894 才是沿跳跃方向的位移
ISO = 1.118
PIECE_CONF = float(os.environ.get("PIECE_CONF", "0.15"))
MODEL2 = os.environ.get("EMU_MODEL2")

det2 = None
det2_lo = None
if MODEL2:
    from detect import Detector
    det2 = Detector(MODEL2, (PIECE_CONF, emu.TARGET_CONF))
    det2_lo = Detector(MODEL2, (PIECE_CONF, emu.RELAX_TARGET_CONF))


def merge_ensemble(results):
    """集成合并: 棋子先定, 目标只在"棋子上方"的候选里取最高分。

    直接按置信度合并会出事 —— 基线模型会把**棋子下方的方块**也当成目标(实测 0.82),
    比主模型给出的正确目标(0.65)分还高, 合并后目标跑到下方, 触发"目标必须在棋子上方"
    的规则, 结果被判成"无目标"而白白漏跳。
    """
    out = {}
    cands = [x[0] for x in results if x[0] is not None]
    out[0] = max(cands, key=lambda y: y["conf"]) if cands else None
    piece = out[0]
    cands = [x[1] for x in results if x[1] is not None]
    if piece is not None:
        cands = [c for c in cands if c["kpt"][1] < piece["kpt"][1]]
    out[1] = max(cands, key=lambda y: y["conf"]) if cands else None
    return out


def detect(img):
    """主模型 + 可选集成(第二个模型); 目标检不出时用放宽阈值再解一次。"""
    r = emu.det.infer(img)
    det2 = getattr(emu, "det2", None)
    res = r if det2 is None else merge_ensemble([r, det2.infer(img)])
    if res[1] is not None:
        return res
    # 真机渲染会让目标类置信整体下移(同一帧模拟器 0.89 / 真机 0.057), 贴着阈值就会空等
    lo = getattr(emu, "det_lo", None)
    if lo is None:
        return res
    det2_lo = getattr(emu, "det2_lo", None)
    r2 = lo.infer(img)
    res2 = r2 if det2_lo is None else merge_ensemble([r2, det2_lo.infer(img)])
    return res2 if res2[1] is not None else res


def find_restart_button(img):
    """结算画面里找白色胶囊按钮("再玩一局")。

    只按"白且够大"会在游戏画面上误判(白色方块也是白的), 所以额外要求水平居中。
    """
    h, w = img.shape[:2]
    sub = img[int(h * 0.45):, :]
    bright = ((sub[:, :, 0] > 225) & (sub[:, :, 1] > 225) & (sub[:, :, 2] > 225)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bright, 8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (w * 0.3 < bw < w * 0.75 and 40 < bh < 260 and area > 20000):
            continue
        if abs(cent[i][0] - w / 2) > w * 0.25:        # 结算按钮基本居中
            continue
        if best is None or area > best[0]:
            best = (area, cent[i][0], cent[i][1] + int(h * 0.45))
    return None if best is None else (best[1], best[2])


def game_over():
    """判定是否真的结算。

    单帧会被画面上的白色方块骗过去, 所以要求连续两帧都满足: 识别不到棋子/目标,
    且能找到居中的结算按钮。返回 (是否结算, 最后一帧)。
    """
    img = None
    for k in range(2):
        img = capture("over_%d" % k)
        if img is None:
            return False, None
        r = emu.det.infer(img)
        if r[0] is not None and r[1] is not None:
            return False, img
        if find_restart_button(img) is None:
            return False, img
        time.sleep(1.0)
    return True, img


def capture_tmp(tag="tmp"):
    """截图但不留档(分数复读用), 覆盖同一个文件。"""
    remote = "/sdcard/_emu.png"
    sh("shell", "screencap", "-p", remote)
    local = OUT / ("_%s.png" % tag)
    sh("pull", remote, str(local))
    return cv2.imread(str(local))


def stable_score(tag, tries=4, gap=0.5):
    """分数有滚动动画, 连续两次读数一致才认。"""
    prev = None
    for t in range(tries):
        img = capture_tmp("score%s" % tag)
        s = read_score(img) if img is not None else None
        if s is not None and prev == s:
            return s, img
        prev = s
        time.sleep(gap)
    return prev, None


def topface_offset(img, px, py):
    """棋子脚下顶面质心与棋子底的横向差; 面积太小(遮挡/边缘)则返回 None。"""
    h, w = img.shape[:2]
    seed = (int(px), min(int(py) + 30, h - 1))
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img.copy(), mask, seed, 0, (18, 18, 18), (18, 18, 18),
                  4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)
    m = mask[1:-1, 1:-1].astype(bool)
    ys, xs = np.nonzero(m)
    if xs.size < 3000:
        return None
    return float(xs.mean() - px), int(xs.max() - xs.min())


def stable_detect(tag, tries=8, gap=0.6, tol=8.0):
    """连续两帧检测一致才认(躲开落地/起飞动画)。"""
    prev = None
    for t in range(tries):
        img = capture(tag if t == 0 else "%s_w%d" % (tag, t))
        if img is None:
            time.sleep(gap)
            continue
        r = detect(img)
        cur = None
        if r[0] and r[1] and r[1]["kpt"][1] < r[0]["kpt"][1]:
            cur = (r[0]["kpt"][0], r[0]["kpt"][1], r[1]["kpt"][0], r[1]["kpt"][1])
        if cur and prev and max(abs(a - b) for a, b in zip(cur, prev)) < tol:
            return img, r
        prev = cur
        time.sleep(gap)
    # 失败也要把最后一张图返回给调用方, 否则结算画面的"再玩一局"永远找不到
    return img, None


def run(model_dir, n, k, a):
    rows = []
    restarts = 0
    for i in range(1, n + 1):
        img, res = stable_detect("a%03d_pre" % i)
        if res is None:
            btn = find_restart_button(img) if img is not None else None
            if btn is not None:
                press(btn[0], btn[1], 60)
                restarts += 1
                print("重开第 %d 次 (按钮 %.0f,%.0f)" % (restarts, btn[0], btn[1]))
                time.sleep(6.0)
            else:
                print("第 %d 跳: 识别失败且未找到重开按钮, 等待" % i)
                time.sleep(1.5)
            continue
        p, t = res[0]["kpt"], res[1]["kpt"]
        d = ((t[0] - p[0]) ** 2 + (p[1] - t[1]) ** 2) ** 0.5
        direction = 1 if t[0] >= p[0] else -1
        s_pre, _ = stable_score("%d" % i)
        ms = a + k * d
        press(p[0], p[1], ms)
        time.sleep(1.7)
        img2 = capture("a%03d_post" % i)
        s_post, _ = stable_score("p%d" % i)
        r2 = detect(img2) if img2 is not None else {0: None, 1: None}
        gain = (s_post - s_pre) if (s_pre is not None and s_post is not None) else None
        off = None
        if r2[0] is not None:
            off = topface_offset(img2, r2[0]["kpt"][0], r2[0]["kpt"][1])
        row = {
            "idx": i, "d": round(d, 1), "press": round(ms, 1), "dir": direction,
            "score_pre": s_pre, "score_post": s_post, "gain": gain,
            "off": None if off is None else round(off[0], 1),
            "face_w": None if off is None else off[1],
            # 投影到跳跃方向的落点误差: 正 = 过冲(越过中心), 负 = 不足
            # off = 顶面质心 - 棋子底; 棋子比质心更靠前(沿 dir)即过冲
            "proj_err": None if off is None else round(-off[0] * direction * ISO, 1),
            "piece_conf": round(res[0]["conf"], 3), "target_conf": round(res[1]["conf"], 3),
        }
        rows.append(row)
        print("第 %3d 跳: 跳距=%6.1f 按压=%4dms 方向=%+d | 偏移=%s 投影误差=%s | 分数 %s->%s (%s)"
              % (i, d, ms, direction,
                 "  -- " if row["off"] is None else "%+5.1f" % row["off"],
                 "  -- " if row["proj_err"] is None else "%+6.1f" % row["proj_err"],
                 s_pre, s_post, gain))
        if not LOG.exists():
            with LOG.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=list(row)).writeheader()
        with LOG.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(row)).writerow(row)
    print("完成 %d 跳, 日志 %s" % (len(rows), LOG))
    fit(LOG, k)


def fit(path, k=None):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))]
    rows = [r for r in rows if r["proj_err"] not in (None, "", "None")]
    if len(rows) < 12:
        print("有效样本 %d 条, 太少, 先多跑几跳" % len(rows))
        return
    d = np.array([float(r["d"]) for r in rows])
    e = np.array([float(r["proj_err"]) for r in rows])
    g = np.array([float(r["gain"]) if r["gain"] not in (None, "", "None") else np.nan for r in rows])
    beta, c0 = np.polyfit(d, e, 1)
    hits = np.isfinite(g) & (g >= 2) & (g % 2 == 0)
    miss = np.isfinite(g) & (g == 1)
    print("\n样本 %d | 命中中心 %d | 普通落地 %d | 命中率 %.1f%%"
          % (len(rows), hits.sum(), miss.sum(), 100.0 * hits.sum() / max(hits.sum() + miss.sum(), 1)))
    print("误差回归: 投影误差 = %+.4f * d %+.1f px   (斜率即 α*K-1)" % (beta, c0))
    print("  命中组 |误差| 均值 %.1f px | 普通组 %.1f px"
          % (np.mean(np.abs(e[hits])) if hits.any() else -1,
             np.mean(np.abs(e[miss])) if miss.any() else -1))
    if k:
        knew = k / (1 + beta)
        print("  建议 K: %.4f -> %.4f  (β=%+.4f)" % (k, knew, beta))
        print("  参考 A: 截距 %+.1f px 折算 %+.1f ms (含测量偏置, 需用命中率确认)" % (c0, c0 * k))
    for lo, hi in ((0, 250), (250, 400), (400, 550), (550, 900)):
        m = (d >= lo) & (d < hi)
        if m.sum() < 4:
            continue
        print("  跳距 %3d-%3d px: n=%2d 平均误差 %+6.1f 命中率 %.0f%%"
              % (lo, hi, m.sum(), e[m].mean(),
                 100.0 * (hits & m).sum() / max((hits | miss)[m].sum(), 1)))


def main():
    if sys.argv[1] == "--fit":
        fit(Path(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else None)
        return
    model, n = sys.argv[1], int(sys.argv[2])
    k = float(sys.argv[3]) if len(sys.argv) > 3 else 1.38
    a = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    emu.det = __import__("detect").Detector(model, (PIECE_CONF, 0.35))
    print("标定: 模型 %s | 跳数 %d | press = %.1f + %.4f*d | 帧目录 %s" % (model, n, a, k, OUT))
    run(model, n, k, a)


if __name__ == "__main__":
    main()
