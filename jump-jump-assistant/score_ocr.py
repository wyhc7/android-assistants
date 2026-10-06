# -*- coding: utf-8 -*-
"""读分数: 跳一跳的分数是固定位置的深色像素字, 切出每个数字再和模板比对。

为什么要读分数: 计分规则里"普通落地 +1 / 命中中心 +2,4,6...32"意味着**分数变化就是落点
是否命中中心的真值标签**, 比图像估算落点误差可靠得多, 可以直接做闭环标定。

用法:
  python score_ocr.py --build            聚类出数字模板, 输出 preview/digit_clusters.png 待人工确认
  python score_ocr.py --label 0123...    按聚类图顺序给出每个簇对应的数字, 写入 digit_templates.npz
  python score_ocr.py <图片> [...]       读分数
  python score_ocr.py --check            用已知分数的帧验证
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
TPL = ROOT / "digit_templates.npz"
# 分数区域(绝对像素, 1080x1920): 游戏内布局固定在左上角
ROI = (100, 198, 470, 302)
NORM = (24, 36)          # 归一化尺寸
MIN_H = 40               # 数字最小高度, 用于滤掉场景里的深色小块


def extract_digits(img, roi=ROI):
    """返回 [(bitmap, x0)], 按横坐标排序。"""
    x1, y1, x2, y2 = roi
    roi_img = img[y1:y2, x1:x2]
    if roi_img.size == 0:
        return []
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    bw = (gray < 120).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < MIN_H or w < 6 or area < 60:
            continue
        if w > (x2 - x1) * 0.6:          # 连成一整条的多半是背景条纹
            continue
        g = (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        g = cv2.resize(g, NORM, interpolation=cv2.INTER_AREA)
        out.append((g, x))
    out.sort(key=lambda t: t[1])
    return out


def build(frames, max_frames=120):
    """跨帧聚类数字图案, 输出簇心拼图供人工确认。"""
    cl, cnt = [], []
    for p in frames[:max_frames]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        for g, _ in extract_digits(img):
            v = g.astype(np.float32).ravel()
            if not cl:
                cl.append(v)
                cnt.append(1)
                continue
            d = [np.mean((v - c) ** 2) for c in cl]
            j = int(np.argmin(d))
            if d[j] < 900:                # 同一数字
                cnt[j] += 1
                cl[j] = (cl[j] * (cnt[j] - 1) + v) / cnt[j]
            else:
                cl.append(v)
                cnt.append(1)
    order = np.argsort(-np.array(cnt))
    cl = [cl[i] for i in order]
    cnt = [cnt[i] for i in order]
    print("聚类得到 %d 个图案(按出现次数排序): %s" % (len(cl), cnt))
    sheet = np.full((NORM[1] * ((len(cl) + 4) // 5) + 10, NORM[0] * 5 + 10), 255, np.uint8)
    for k, c in enumerate(cl):
        r, cc = divmod(k, 5)
        sheet[r * NORM[1] + 5:r * NORM[1] + 5 + NORM[1],
              cc * NORM[0] + 5:cc * NORM[0] + 5 + NORM[0]] = (
            255 - c.reshape(NORM[1], NORM[0])).astype(np.uint8)
    cv2.imwrite(str(ROOT / "preview" / "digit_clusters.png"), sheet)
    np.savez(TPL, clusters=np.array(cl), counts=np.array(cnt))
    print("簇心已存 %s, 拼图 %s" % (TPL.name, "preview/digit_clusters.png"))


def label(labels):
    z = np.load(TPL)
    cl = list(z["clusters"])
    if len(labels) != len(cl):
        print("需要 %d 个字符, 收到 %d 个" % (len(cl), len(labels)))
        return
    np.savez(TPL, clusters=np.array(cl), counts=z["counts"],
             labels=np.array([c for c in labels]))
    print("模板已写入: %s -> %s" % (labels, TPL.name))


def read(img, debug=False):
    z = np.load(TPL)
    cl = z["clusters"]
    lab = z["labels"] if "labels" in z else None
    if lab is None:
        raise SystemExit("模板还没标号, 先运行 --label")
    ds = extract_digits(img)
    out = ""
    for g, x in ds:
        v = g.astype(np.float32).ravel()
        j = int(np.argmin([np.mean((v - c) ** 2) for c in cl]))
        out += str(lab[j])
    if debug:
        for g, x in ds:
            print("  x=%3d -> %s" % (x, out))
    return int(out) if out else None


def main():
    if sys.argv[1] == "--build":
        frames = sorted(Path(ROOT / "emulator").glob("a*_pre.png"))
        frames += sorted(Path(ROOT / "emulator").glob("a*_post.png"))
        frames += sorted((ROOT / "screenshots").glob("*.png"))
        build(frames)
    elif sys.argv[1] == "--label":
        label(sys.argv[2])
    elif sys.argv[1] == "--check":
        known = [("emulator/fail063_base.png", 1197), ("emulator/r7/fail_pre_035.png", 175),
                 ("emulator/skin_fail38.png", 380), ("emulator/fail107_det.png", 726),
                 ("emulator/fail_post_008.png", 30)]
        for f, s in known:
            img = cv2.imread(str(ROOT / f))
            if img is None:
                continue
            got = read(img)
            print("%-38s 期望 %4d 读到 %s %s" % (f, s, got, "OK" if got == s else "差异"))
    else:
        for f in sys.argv[1:]:
            img = cv2.imread(f)
            print(f, "->", None if img is None else read(img))


if __name__ == "__main__":
    main()
