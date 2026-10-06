# -*- coding: utf-8 -*-
"""自动预标注: 棋子底部中心点 + 目标方块顶面框(排除阴影)。

原理:
  棋子: 深紫色 HSV 掩码 (S高 V低), 取最大连通域, 底部中心 = 域最低点行的x均值。
  目标: 背景色用四边采样估计, 色差分得到前景blob; 含棋子的blob为当前方块,
        其余取离棋子最近者为目标。顶面框 = blob 顶顶点(min_y) 到前半高度内
        最大宽度行之间的x范围。地面阴影在方块底部之下, 不参与顶部宽度统计,
        且顶面高度上限 = blob宽*0.6, 进一步隔绝阴影行。
输出 annotations/<name>.json, 由 annotate.py 复核修正。
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
SRC = ROOT / "screenshots"
ANN = ROOT / "annotations"
ANN.mkdir(exist_ok=True)


def bg_color(img):
    """逐行中位数估计渐变背景, 返回与img同形的背景图。"""
    row_med = np.median(img, axis=1)  # (h, 3) 每行背景色
    return np.repeat(row_med[:, None, :], img.shape[1], axis=1)


def find_piece(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 深紫棋子: 高饱和、低亮度、蓝紫色相
    mask = cv2.inRange(hsv, (100, 50, 20), (170, 255, 150))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 800]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    pts = c.reshape(-1, 2)
    y_max = pts[:, 1].max()
    xs = pts[pts[:, 1] >= y_max - 3, 0]
    return (float(xs.mean()), float(y_max))


def find_blobs(img, bg):
    diff = np.abs(img.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
    mask = (diff > 40).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    h, w = mask.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    cnts = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 15000 or (bw > w * 0.95 and bh < h * 0.1):
            continue
        if y + bh / 2 < h * 0.12:
            continue  # 顶部UI区(胶囊按钮/分数)
        sub = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
        # 距离变换找峰值, 分水岭切开粘连方块/阴影
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        mx = cv2.dilate(dist, np.ones((61, 61), np.uint8))
        peaks = ((dist == mx) & (dist > 25)).astype(np.uint8)
        pn, plab = cv2.connectedComponents(peaks)
        if pn <= 2:
            pieces = [(sub, 1)]
        else:
            markers = np.ones(sub.shape, np.int32)  # 1=确定背景
            markers[sub > 0] = 0                    # 0=未知(前景)
            markers[plab > 0] = plab[plab > 0] + 1  # >=2=种子峰
            roi = img[y:y + bh, x:x + bw]
            cv2.watershed(roi, markers)
            pieces = [((markers == k).astype(np.uint8), k)
                      for k in range(2, pn + 1)]
        for seg, _k in pieces:
            if seg.sum() < 15000:
                continue
            sc, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            for c in sc:
                c = c + np.array([[[x, y]]])
                if cv2.contourArea(c) > 15000:
                    cnts.append((c, seg, (x, y)))
    return mask, cnts


def top_face_box(seg, off, img):
    """段掩码的顶面外接框(几何法)。

    顶面 = 段顶顶点(min_y) 到前半高度内最宽行之间的区域;
    中心x取顶顶点x与最宽行x范围的均值以抵抗不对称噪声。
    地面阴影在方块底部之下, 不进入前半高度的宽度统计。
    """
    ys, xs = np.nonzero(seg)
    x0, y0 = xs.min(), ys.min()
    w, h = xs.max() - x0 + 1, ys.max() - y0 + 1
    sub = seg[y0:y0 + h, x0:x0 + w]
    top_h = min(h, int(w * 0.6))
    rows = (sub[:top_h] > 0).sum(axis=1)
    nz = np.nonzero(rows)[0]
    if len(nz) == 0:
        return None
    ty0 = nz[0]
    y_w = ty0 + int(np.argmax(rows[ty0:]))
    band = sub[ty0:y_w + 1]
    cols = np.nonzero((band > 0).sum(axis=0))[0]
    if len(cols) == 0:
        return None
    # 顶顶点x
    top_row = np.nonzero(sub[ty0])[0]
    x_top = (top_row[0] + top_row[-1]) / 2
    cx = (x_top + (cols[0] + cols[-1]) / 2) / 2  # 两次取中抵抗噪声
    ox, oy = off
    bx = int(round(cx - (cols[-1] - cols[0]) / 2))
    bw = int(cols[-1] - cols[0] + 1)
    return [int(ox + x0 + bx), int(oy + y0 + ty0), bw, int(y_w - ty0 + 1)]


def annotate(img_path):
    img = cv2.imread(str(img_path))
    piece = find_piece(img)
    if piece is None:
        return None, "piece not found"
    bg = bg_color(img)
    mask, cnts = find_blobs(img, bg)
    if not cnts:
        return None, "no blobs"
    px, py = piece
    # 排除棋子自身所在的段(棋子底部点落在其上), 其余取最近者
    cands = []
    for c, seg, off in cnts:
        if cv2.pointPolygonTest(c, (px, py - 5), False) >= 0:
            continue  # 当前站立方块
        box = top_face_box(seg, off, img)
        if box is None:
            continue
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        cands.append((cy, box))  # 新目标总在上方, 取中心y最小者
    if not cands:
        return None, "no target"
    box = min(cands)[1]
    h, w = img.shape[:2]
    rec = {"image": img_path.name, "width": w, "height": h,
           "piece": [round(px, 1), round(py, 1)],
           "target_box": box,  # [x, y, w, h] 顶面外接框
           "target": [round(box[0] + box[2] / 2, 1),
                      round(box[1] + box[3] / 2, 1)]}
    return rec, "ok"


def main():
    ok, fail = 0, []
    for f in sorted(SRC.iterdir()):
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        rec, msg = annotate(f)
        if rec:
            (ANN / (f.stem + ".json")).write_text(
                json.dumps(rec, ensure_ascii=False, indent=2),
                encoding="utf-8")
            ok += 1
        else:
            fail.append((f.name, msg))
    print(f"成功 {ok} 张, 失败 {len(fail)} 张")
    for n, m in fail:
        print(" ", n, "->", m)


if __name__ == "__main__":
    main()
