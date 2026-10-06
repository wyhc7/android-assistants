# -*- coding: utf-8 -*-
"""混合微调: 人工精标(手机截图) + 模拟器实测真实帧(伪标签) -> 续训。

为什么要混合: 只喂伪标签会强化模型自身的偏差, 只喂手机截图则学不到模拟器域
(1080x1920 新分辨率、微信 UI 覆盖、细柱/圆柱/蜂巢等新皮肤)。
学习率压到 1/50, 避免破坏已有能力。

用法: python finetune.py [epochs] [权重]
"""
import json
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
DS = ROOT / "dataset_ft"
SOURCES = [(ROOT / "annotations", ROOT / "screenshots", "手机截图"),
           (ROOT / "annotations_emu", ROOT / "emulator", "模拟器真实帧")]
# 可选: 追加一轮新采集的数据(标注目录 + 图片目录), 用 env 传入以便隔离不同轮次
if os.environ.get("EXTRA_ANN") and os.environ.get("EXTRA_IMG"):
    SOURCES.append((Path(os.environ["EXTRA_ANN"]), Path(os.environ["EXTRA_IMG"]),
                    os.environ.get("EXTRA_TAG", "新增真实帧")))
PIECE_BOX = 60
VAL_RATIO = 0.12
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
BASE = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "runs" / "jump_pose-3" / "weights" / "best.pt"


def build():
    for sub in ("images", "labels"):
        d = DS / sub
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            old.unlink()
    stats = {}
    for ann_dir, img_dir, tag in SOURCES:
        n = 0
        for f in sorted(ann_dir.glob("*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            src = img_dir / rec["image"]
            if not src.exists():
                continue
            W, H = rec["width"], rec["height"]
            px, py = rec["piece"]
            bx, by, bw, bh = rec["target_box"]
            cx, cy = bx + bw / 2, by + bh / 2
            if not (cy < py):        # 硬规则: 目标必须在棋子上方
                continue
            lines = [
                f"0 {px / W:.6f} {py / H:.6f} {PIECE_BOX / W:.6f} "
                f"{PIECE_BOX / H:.6f} {px / W:.6f} {py / H:.6f} 2",
                f"1 {cx / W:.6f} {cy / H:.6f} {bw / W:.6f} {bh / H:.6f} "
                f"{cx / W:.6f} {cy / H:.6f} 2",
            ]
            name = src.name
            # 不同来源可能有同名文件(如各轮都叫 a001_pre.png), 直接复制会互相覆盖,
            # 导致图片与标签错配 —— 冲突时加来源前缀。
            if (DS / "images" / name).exists():
                name = "%s_%s" % (tag, name)
            shutil.copy2(src, DS / "images" / name)
            (DS / "labels" / (Path(name).stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            n += 1
        stats[tag] = n
    print("数据集:", stats, "合计", sum(stats.values()))
    return sum(stats.values())


def split():
    imgs = sorted((DS / "images").glob("*"))
    random.seed(7)
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * VAL_RATIO))
    val, train = imgs[:n_val], imgs[n_val:]
    for s, items in (("train", train), ("val", val)):
        for sub in ("images", "labels"):
            d = DS / s / sub
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("*"):
                old.unlink()
        for p in items:
            shutil.copy2(p, DS / s / "images" / p.name)
            shutil.copy2(DS / "labels" / (p.stem + ".txt"), DS / s / "labels" / (p.stem + ".txt"))
    cfg = {"path": str(DS), "train": "train/images", "val": "val/images",
           "names": {0: "piece", 1: "target"}, "kpt_shape": [1, 3]}
    (DS / "data.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    print("train=%d val=%d" % (len(train), len(val)))


def main():
    from ultralytics import YOLO
    if build() == 0:
        raise SystemExit("无数据")
    split()
    model = YOLO(str(BASE))
    res = model.train(data=str(DS / "data.yaml"), epochs=EPOCHS, imgsz=640, batch=16,
                      lr0=LR, lrf=0.1, project=str(ROOT / "runs"), name="jump_pose_ft2",
                      hsv_h=0.02, hsv_s=0.5, hsv_v=0.4, scale=0.5)
    best = Path(res.save_dir) / "weights" / "best.pt"
    print("best:", best)
    YOLO(str(best)).export(format="ncnn", half=False)
    print("NCNN 导出:", best.parent / "best_ncnn_model")


if __name__ == "__main__":
    main()
