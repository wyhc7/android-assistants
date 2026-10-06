# -*- coding: utf-8 -*-
"""跳一跳标注工具: 一张图一次点击, 标签由人保证正确。

要标的两件事:
  1) 棋子底部中心(棋子与方块接触的那一点) —— 自动检测, 通常已是正确的红点
  2) 目标落点 = 你要跳过去的那块方块【顶面的中心】—— 该方块永远在棋子上方
绿框 = 目标方块顶面; 十字 = 落点; 红点 = 棋子底部中心。

操作:
  左键点击   设置落点(点哪就是哪)
  Enter      保存并下一张
  p + 点击   修正棋子底部中心
  b + 两点   手画顶面框(自动拟合不满意时)
  s 跳过     q 退出
  滚轮/+/-   缩放(以光标为中心)    方向键 平移    f 全图    1 原始100%

输出 annotations/<name>.json, 带 confirmed 标记, 可断点续标。
"""
import ctypes
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import auto_annotate2 as A

ROOT = Path(__file__).parent
SRC, ANN = ROOT / "screenshots", ROOT / "annotations"
DEF_W, DEF_H = 371, 186      # 实测中位顶面框(宽/高), 2:1等轴
FONT = "C:/Windows/Fonts/msyh.ttc"
MAX_ZOOM = 4.0

st = {"mode": "target", "clicks": [], "rec": None, "img": None, "mask": None,
      "zoom": 1.0, "ox": 0.0, "oy": 0.0, "cur": (0, 0), "msg": ""}


def screen_size():
    try:
        u = ctypes.windll.user32
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


SW, SH = screen_size()


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except Exception:
        return ImageFont.load_default()


def put_cn(canvas, lines, size=15):
    """PIL 渲染中文(OpenCV 自带字体画不了中文)。"""
    im = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f = font(size)
    pad, y = 6, 4
    for t, col in lines:
        w = int(d.textlength(t, font=f))
        d.rectangle([pad - 3, y - 2, pad + w + 3, y + size + 4],
                    fill=(0, 0, 0, 165))
        d.text((pad, y), t, font=f, fill=col)
        y += size + 9
    out = Image.alpha_composite(im.convert("RGBA"), ov)
    return cv2.cvtColor(np.array(out.convert("RGB")), cv2.COLOR_RGB2BGR)


def view_rect():
    """当前显示区域(源图坐标)与画布尺寸。"""
    img = st["img"]
    H, W = img.shape[:2]
    dw, dh = st["canvas"]
    src_w = min(W, dw / st["zoom"])
    src_h = min(H, dh / st["zoom"])
    ox = min(max(st["ox"], 0), W - src_w)
    oy = min(max(st["oy"], 0), H - src_h)
    return int(ox), int(oy), int(src_w), int(src_h), dw, dh


def to_src(dx, dy):
    ox, oy, src_w, src_h, dw, dh = view_rect()
    return (ox + dx * src_w / dw, oy + dy * src_h / dh)


def center_on(sx, sy, zoom=None):
    """把源点放到视图中心。"""
    img = st["img"]
    dw, dh = st["canvas"]
    if zoom is not None:
        st["zoom"] = float(min(max(zoom, 0.05), MAX_ZOOM))
    src_w = min(img.shape[1], dw / st["zoom"])
    src_h = min(img.shape[0], dh / st["zoom"])
    st["ox"], st["oy"] = sx - src_w / 2, sy - src_h / 2


def fit_view():
    img = st["img"]
    dw, dh = st["canvas"]
    st["zoom"] = min(dw / img.shape[1], dh / img.shape[0])
    st["ox"] = st["oy"] = 0.0


def fit_box(img, mask, cx, cy):
    """在点击处拟合顶面框(等轴法: 宽W、高W/2的菱形)。"""
    cx, cy = int(round(cx)), int(round(cy))
    if not (0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]):
        return None
    if mask[cy, cx] == 0:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return None
        j = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        cy, cx = int(ys[j]), int(xs[j])
    n, lab = cv2.connectedComponents(mask)
    idx = lab[cy, cx]
    if idx == 0:
        return None
    return A.top_face_iso((lab == idx).astype(np.uint8))


def on_mouse(event, x, y, flags, param):
    sx, sy = to_src(x, y)
    st["cur"] = (sx, sy)
    rec, img, mask = st["rec"], st["img"], st["mask"]
    if event == cv2.EVENT_MOUSEWHEEL:
        f = 1.15 if flags > 0 else 1 / 1.15
        center_on(sx, sy, st["zoom"] * f)
        return
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if st["mode"] == "piece":
        rec["piece"] = [sx, sy]
        st["mode"] = "target"
        st["msg"] = "棋子已更新"
        return
    if st["mode"] == "box":
        st["clicks"].append((sx, sy))
        if len(st["clicks"]) == 2:
            (x1, y1), (x2, y2) = st["clicks"]
            rec["target_box"] = [min(x1, x2), min(y1, y2),
                                 abs(x2 - x1), abs(y2 - y1)]
            st["clicks"] = []
            st["mode"] = "target"
            st["msg"] = "框已手画"
        return
    # 落点: 点击即权威落点, 框自动拟合(仅作显示)
    box = fit_box(img, mask, sx, sy)
    if box is None:
        box = [sx - DEF_W / 2, sy - DEF_H / 2, DEF_W, DEF_H]
    rec["target_box"] = [int(v) for v in box]
    rec["target"] = [float(sx), float(sy)]
    st["msg"] = "落点已设定"


def draw():
    img, rec = st["img"], st["rec"]
    ox, oy, src_w, src_h, dw, dh = view_rect()
    crop = img[oy:oy + src_h, ox:ox + src_w]
    canvas = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_LINEAR)

    def m(p):
        return (int((p[0] - ox) * dw / src_w), int((p[1] - oy) * dh / src_h))

    if rec.get("target_box"):
        x, y, w, h = rec["target_box"]
        p1, p2 = m((x, y)), m((x + w, y + h))
        cv2.rectangle(canvas, p1, p2, (0, 255, 0), 3)
    px, py = rec["piece"]
    cv2.circle(canvas, m((px, py)), 7, (0, 0, 255), -1)
    cv2.drawMarker(canvas, m((px, py)), (0, 0, 255), cv2.MARKER_TILTED_CROSS,
                   22, 2)
    if rec.get("target"):
        t = m(rec["target"])
        cv2.drawMarker(canvas, t, (255, 0, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.circle(canvas, t, 14, (255, 0, 255), 2)

    mode = {"target": "左键点落点", "piece": "点棋子底部中心",
            "box": "点框的两个对角"}[st["mode"]]
    lines = [
        (f"[{st['cur_n']}/{st['cur_tot']}] {st['rec']['image'][-18:]}", (255, 255, 0)),
        (f"当前: {mode}   缩放 {st['zoom']:.2f}x", (255, 255, 255)),
        ("左键=落点(目标顶面中心)  Enter=保存下一张", (255, 255, 255)),
        ("p=改棋子  b=手画框  s=跳过  q=退出", (255, 255, 255)),
        ("滚轮/+/-=缩放  方向键=平移  f=全图", (255, 255, 255)),
    ]
    if st["msg"]:
        lines.insert(1, (st["msg"], (0, 255, 128)))
    return put_cn(canvas, lines)


def main():
    files = sorted(p for p in SRC.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    todo = []
    for f in files:
        ap = ANN / (f.stem + ".json")
        if not ap.exists():
            todo.append(f)
            continue
        try:
            if not json.loads(ap.read_text(encoding="utf-8")).get("confirmed"):
                todo.append(f)
        except Exception:
            todo.append(f)
    if not todo:
        print("全部已确认, 无需标注")
        return
    print(f"待确认 {len(todo)} 张 (已确认的会自动跳过)")

    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("label", on_mouse)
    st["cur_tot"] = len(todo)

    for k, f in enumerate(todo, 1):
        img = cv2.imread(str(f))
        if img is None:
            continue
        ap = ANN / (f.stem + ".json")
        rec = {}
        if ap.exists():
            try:
                rec = json.loads(ap.read_text(encoding="utf-8"))
            except Exception:
                rec = {}
        pmask, piece = A.piece_mask(img)
        if not rec.get("piece"):
            rec["piece"] = list(piece) if piece else [img.shape[1] / 2,
                                                     img.shape[0] / 2]
        rec.setdefault("target_box", None)
        rec.setdefault("target", None)
        rec.update({"image": f.name, "width": img.shape[1],
                    "height": img.shape[0]})
        # 画布: 等比适配屏幕, 绝不拉伸
        s = min(SW * 0.60 / img.shape[1], SH * 0.90 / img.shape[0])
        st.update(rec=rec, img=img, mask=A.fg_mask(img, pmask),
                  mode="target", clicks=[], cur_n=k, msg="")
        st["canvas"] = (max(200, int(img.shape[1] * s)),
                        max(200, int(img.shape[0] * s)))
        fit_view()  # 默认显示全图(便于判断哪块是目标), 滚轮可放大
        while True:
            cv2.imshow("label", draw())
            key = cv2.waitKey(20) & 0xFF
            if key == 13:  # Enter
                if not rec.get("target_box"):
                    st["msg"] = "还没有落点, 请先左键点击"
                    continue
                rec["confirmed"] = True
                ap.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                              encoding="utf-8")
                break
            if key == ord("p"):
                st["mode"] = "piece"
            elif key == ord("b"):
                st["mode"], st["clicks"] = "box", []
            elif key == ord("f"):
                fit_view()
            elif key == ord("1"):
                center_on(*st["cur"], zoom=1.0)
            elif key in (ord("+"), ord("=")):
                center_on(*st["cur"], zoom=st["zoom"] * 1.25)
            elif key in (ord("-"), ord("_")):
                center_on(*st["cur"], zoom=st["zoom"] / 1.25)
            elif key in (81, 2, 2424832):      # Left
                st["ox"] -= view_rect()[2] * 0.15
            elif key in (83, 3, 2555904):      # Right
                st["ox"] += view_rect()[2] * 0.15
            elif key in (82, 0, 2490368):      # Up
                st["oy"] -= view_rect()[3] * 0.15
            elif key in (84, 1, 2621440):      # Down
                st["oy"] += view_rect()[3] * 0.15
            elif key == ord("s"):
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                print("已退出")
                return
    cv2.destroyAllWindows()
    print("完成")


if __name__ == "__main__":
    main()
