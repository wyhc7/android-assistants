# -*- coding: utf-8 -*-
"""顶面提取 v2: 从顶顶点沿连通行段下行, 不依赖分水岭切分。

流程:
  1) 逐行中位数估计渐变背景 -> 色差掩码
  2) 扣除棋子像素与顶部UI区
  3) 目标 = 顶顶点最高(最靠上)的方块(新目标总在上方)
  4) 顶面 = 从该顶点出发, 沿含顶点的连通行段下行,
     记录最大宽度W与到达该宽度的高度H, 直到下行深度超过 0.6*W
  5) 输出框 [xc-W/2, y_top, W, H_ratio*W] 与落点中心

同时统计 H/W 比例, 用于标定该游戏的等轴投影参数。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
SRC, ANN = ROOT / "screenshots", ROOT / "annotations"


def bg_image(img):
    """估计纵向渐变背景。

    逐行中位数在方块占据该行近半像素时会失真, 故改取左右边缘条带的
    中位数得到行剖面, 再做纵向中值平滑, 抵消个别行被方块贴边污染的情况。
    """
    h, w = img.shape[:2]
    # 内移条带: 避开贴边的UI(右上胶囊按钮压在最右20px)
    strip = np.concatenate([img[:, 25:50], img[:, w - 50:w - 25]], axis=1)
    prof = np.median(strip.astype(np.float32), axis=1)  # (h, 3)
    k = 51
    pad = np.pad(prof, ((k // 2, k // 2), (0, 0)), mode="edge")
    smooth = np.stack([np.median(pad[i:i + k], axis=0)
                       for i in range(h)]).astype(np.float32)
    return np.repeat(smooth[:, None, :], w, axis=1)


def piece_mask(img):
    """棋子深紫掩码 + 底部中心点。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (100, 50, 20), (170, 255, 150))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 800]
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    pts = c.reshape(-1, 2)
    ymax = pts[:, 1].max()
    xs = pts[pts[:, 1] >= ymax - 3, 0]
    return m, (float(xs.mean()), float(ymax))


def chroma(a):
    """相对色度: (max-min)/max, 用于区分彩色方块与背景色调。"""
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    return (mx - mn) / (mx + 1.0)


def fg_mask(img, pmask):
    """前景掩码: 背景差分 - 阴影 - 棋子 - UI区。

    阴影 = 背景的乘法变暗(同色调、各通道等比降低), 会把相邻方块连成
    一个巨大连通域, 必须剔除。
    """
    f = img.astype(np.float32) + 1.0
    bgf = bg_image(img).astype(np.float32) + 1.0
    ratio = f / bgf
    s = ratio.mean(axis=2)
    dev = np.abs(ratio - s[..., None]).max(axis=2)
    dc = np.abs(chroma(f) - chroma(bgf))
    is_shadow = (s > 0.6) & (s < 1.02) & (dev < 0.07) & (dc < 0.04)
    diff = np.abs(f - bgf).sum(axis=2)
    block = (diff > 40) & (~is_shadow)
    # 方块面内被误判为阴影的散点会打洞、把连通域切碎: 回填被方块包围的部分
    ff = block.astype(np.uint8).copy()
    h0, w0 = ff.shape
    ffmask = np.zeros((h0 + 2, w0 + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 1)
    holes = (ff == 0) & (~block)
    m = (block | holes).astype(np.uint8) * 255
    if pmask is not None:
        m[pmask > 0] = 0
    h = m.shape[0]
    m[:int(h * 0.12)] = 0  # 顶部UI(分数/胶囊按钮)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return m


def top_face_iso(comp, min_run=24, tol=30):
    """等轴投影直接构造顶面框: 顶面为宽W、高W/2的菱形。

    标定依据: 用棋子锚点(棋子底部中心站在顶面中心)测得 H/W 中位 0.507,
    即 2:1 等轴投影。故:
      顶点 = 首个行段宽度>=min_run 的行
      W    = 顶点下方 0.5*W0 范围内的最大行段宽度
      框   = [顶点x - W/2, 顶点y, W, W/2], 落点 = (顶点x, 顶点y + W/4)
    """
    ys, xs = np.nonzero(comp)
    if ys.size == 0:
        return None
    # 面内小洞(阴影误判残留)会切断行段追踪: 先做水平闭运算桥接,
    # 只填水平小间隙, 不改变纵向范围, 也不会把上下两块连起来。
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    W0 = x1 - x0 + 1
    # 顶点: 首个够宽的行段
    apex = None
    for y in range(y0, y1 + 1):
        idx = np.nonzero(comp[y])[0]
        if idx.size == 0:
            continue
        splits = np.nonzero(np.diff(idx) > 1)[0]
        segs = np.split(idx, splits + 1)
        seg = max(segs, key=len)
        if seg[-1] - seg[0] + 1 >= min_run:
            apex, xc = y, int((seg[0] + seg[-1]) / 2)
            break
    if apex is None:
        return None
    # W: 顶点下方 0.5*W0 范围内, 只跟踪含顶点中心的那个行段
    # (取最宽行段会跳到相邻方块上, 导致框横跨两块)
    W = 0
    for y in range(apex, min(y1 + 1, apex + max(1, int(0.5 * W0)) + 1)):
        idx = np.nonzero(comp[y])[0]
        if idx.size == 0:
            continue
        splits = np.nonzero(np.diff(idx) > 1)[0]
        segs = np.split(idx, splits + 1)
        seg = min(segs, key=lambda s: 0 if s[0] <= xc <= s[-1]
                  else min(abs(s[0] - xc), abs(s[-1] - xc)))
        if not (seg[0] - tol <= xc <= seg[-1] + tol):
            break
        w_new = int(seg[-1] - seg[0] + 1)
        if W and w_new > W * 1.5 and W > 0.25 * W0:
            break  # 行段与相邻方块粘连, 截断
        W = max(W, w_new)
    if W < 40:
        return None
    h = int(round(W / 2))  # 2:1 等轴
    return [int(round(xc - W / 2)), apex, W, h]


def walk_top_face(mask, xv, yv, ratio, min_w=40):
    """从顶点(xv,yv)下行, 返回 (box, 实测比例H/W)。

    仅用于 Otsu 光照分割失败时的兜底: 走到宽度不再增长处(顶面前角),
    并以 0.6*W 限深, 避免走进侧面。
    """
    H, W = mask.shape
    xmin = xmax = xv
    width = 0
    y_max = yv
    y = yv
    while y < H:
        idx = np.nonzero(mask[y])[0]
        if idx.size == 0:
            break
        xc = (xmin + xmax) // 2
        splits = np.nonzero(np.diff(idx) > 1)[0]
        segs = np.split(idx, splits + 1)
        seg = min(segs, key=lambda s: 0 if s[0] <= xc <= s[-1]
                  else min(abs(s[0] - xc), abs(s[-1] - xc)))
        if not (seg[0] - 8 <= xc <= seg[-1] + 8):
            break  # 行段断开, 方块结束
        new_w = int(seg[-1] - seg[0] + 1)
        if new_w > width:
            if width == 0 and new_w < min_w:
                # 顶部细尖突: 起点下移, 不计入顶面
                xmin = xmax = int((seg[0] + seg[-1]) / 2)
                yv = y
                y += 1
                continue
            width, xmin, xmax, y_max = new_w, int(seg[0]), int(seg[-1]), y
        if width and (y - yv) > 0.6 * width:
            break
        y += 1
    if width == 0:
        return None, None
    meas = (y_max - yv) / width
    xc = (xmin + xmax) / 2
    h = max(1, int(round(ratio * width)))
    box = [int(round(xc - width / 2)), int(yv), int(width), h]
    return box, meas


def top_face_otsu(comp, gray, ratio, min_frac=0.45):
    """方块内部按亮度分顶面: 顶面受光更亮, 取含顶顶点的亮连通域。

    兜底为几何法(ratio)。min_frac: 亮域宽度至少为方块宽度的比例,
    否则判为分割失败。
    """
    ys, xs = np.nonzero(comp)
    if ys.size == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    w, h = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
    roi = gray[y0:y0 + h, x0:x0 + w]
    m = comp[y0:y0 + h, x0:x0 + w]
    vals = roi[m > 0]
    if vals.size < 200:
        return None
    th, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = ((roi >= th) & (m > 0)).astype(np.uint8)
    n, lab = cv2.connectedComponents(bright)
    top_row = int(np.nonzero(m.any(axis=1))[0][0])
    seeds = set(np.unique(lab[top_row:top_row + 3])) - {0}
    if not seeds:
        return None
    best = max(seeds, key=lambda k: (lab == k).sum())
    bys, bxs = np.nonzero(lab == best)
    bw = int(bxs.max() - bxs.min() + 1)
    bh = int(bys.max() - bys.min() + 1)
    if bw < min_frac * w:
        return None
    return [int(x0 + bxs.min()), int(y0 + bys.min()), bw, bh]


def robust_vertex(comp, min_w=40):
    """取方块可靠顶点: 第一个行段宽度>=min_w 的行, 跳过细尖突。"""
    for y in range(comp.shape[0]):
        idx = np.nonzero(comp[y])[0]
        if idx.size == 0:
            continue
        splits = np.nonzero(np.diff(idx) > 1)[0]
        segs = np.split(idx, splits + 1)
        seg = max(segs, key=len)
        if seg[-1] - seg[0] + 1 >= min_w:
            return y, int((seg[0] + seg[-1]) / 2)
    return None, None


def annotate(path, ratio):
    img = cv2.imread(str(path))
    pmask, piece = piece_mask(img)
    if piece is None:
        return None, "no piece", None
    m = fg_mask(img, pmask)
    H, W = m.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    px, py = piece
    cands = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 15000:
            continue
        if x <= px <= x + bw and y <= py + 4 <= y + bh:
            continue  # 棋子所站的当前方块
        # 被画面左右边缘裁掉的方块: 只露细条, 不作为目标
        touches_edge = (x <= 2 or x + bw >= W - 2)
        if touches_edge and bw < 0.15 * W:
            continue
        if bw < 0.1 * W:
            continue
        comp = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
        vy, vx = robust_vertex(comp)
        if vy is None:
            continue
        top_y = y + vy
        if top_y > H * 0.85:
            continue  # 底部UI(分享按钮等)
        cands.append((top_y, x + vx, i))
    if not cands:
        return None, "no target", None
    # 最靠上的方块优先; 若其顶面宽度不合理则退到下一个候选
    # 目标块规则(已由用户确认): 落点方块永远在棋子当前位置上方(y更小)。
    # 故只接受顶面中心在棋子上方的候选; 一个都没有则放弃, 交人工标注,
    # 绝不退回选下方方块。
    box = meas = None
    for top_y, xv, idx in sorted(cands):
        comp = (lab == idx).astype(np.uint8)
        cand = top_face_iso(comp)
        if cand is None or cand[2] < 0.12 * W:
            continue
        if cand[1] + cand[3] / 2 >= py:
            continue  # 不在棋子上方
        box = cand
        break
    if box is None:
        # 兜底: 目标块与棋子所站方块在掩码里粘连成一个连通域时,
        # 只看棋子所在行以上的部分, 再按等轴法测顶面。
        alt = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 15000 or bw < 0.1 * W or y > H * 0.85:
                continue
            comp = (lab == i).astype(np.uint8).copy()
            comp[int(py):] = 0
            cand = top_face_iso(comp)
            if cand is None or cand[2] < 0.12 * W:
                continue
            if cand[1] + cand[3] / 2 >= py - 8:
                continue  # 仍不够"在棋子上方"
            alt.append(cand)
        if alt:
            box = min(alt, key=lambda b: b[1])
    if box is None:
        return None, "no target above piece", None
    rec = {"image": path.name, "width": W, "height": H,
           "piece": [round(px, 1), round(py, 1)],
           "target_box": box,
           "target": [round(box[0] + box[2] / 2, 1),
                      round(box[1] + box[3] / 2, 1)]}
    return rec, "ok", meas


def main():
    ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.45
    ANN.mkdir(exist_ok=True)
    for f in ANN.glob("*.json"):
        f.unlink()
    ok, fails, ratios = 0, [], []
    for f in sorted(SRC.iterdir()):
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        rec, msg, meas = annotate(f, ratio)
        if rec:
            (ANN / (f.stem + ".json")).write_text(
                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            ok += 1
            if meas is not None:
                ratios.append(meas)
        else:
            fails.append((f.name, msg))
    print(f"成功 {ok} / 失败 {len(fails)}  (ratio={ratio})")
    for n_, m_ in fails:
        print("  ", n_, "->", m_)
    if ratios:
        r = np.array(ratios)
        print(f"实测 H/W: 中位 {np.median(r):.3f} 均值 {r.mean():.3f} "
              f"分位25/75 {np.percentile(r,25):.3f}/{np.percentile(r,75):.3f}")


if __name__ == "__main__":
    main()
