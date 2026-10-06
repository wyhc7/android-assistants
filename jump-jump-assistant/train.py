# -*- coding: utf-8 -*-
"""数据集构建 + 训练 YOLOv8n-pose。

前置: pip install ultralytics
标签: annotations/*.json -> YOLO-pose
      piece : 以底部中心为关键点的固定小框
      target: 真实顶面框 + 中心关键点
"""
import json
import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
DS = ROOT / "dataset"
ANN = ROOT / "annotations"
SRC = ROOT / "screenshots"
VAL_RATIO = 0.15
PIECE_BOX = 60  # 棋子伪框边长(px)


def build_labels():
    """annotations JSON -> dataset/{images,labels} (YOLO-pose)。"""
    for sub in ("images", "labels"):
        d = DS / sub
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            old.unlink()  # 清理上一轮残留, 否则已删除标注的旧标签会混入
    n = 0
    for f in sorted(ANN.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        src = SRC / rec["image"]
        if not src.exists():
            continue
        W, H = rec["width"], rec["height"]
        px, py = rec["piece"]
        bx, by, bw, bh = rec["target_box"]
        cx, cy = bx + bw / 2, by + bh / 2
        lines = [
            # piece: 固定小框 + 底部中心关键点
            f"0 {px / W:.6f} {py / H:.6f} {PIECE_BOX / W:.6f} "
            f"{PIECE_BOX / H:.6f} {px / W:.6f} {py / H:.6f} 2",
            # target: 真实顶面框 + 中心关键点
            f"1 {cx / W:.6f} {cy / H:.6f} {bw / W:.6f} {bh / H:.6f} "
            f"{cx / W:.6f} {cy / H:.6f} 2",
        ]
        shutil.copy2(src, DS / "images" / src.name)
        (DS / "labels" / (src.stem + ".txt")).write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        n += 1
    print(f"标签构建: {n} 张")
    return n


def split_dataset():
    imgs = sorted((DS / "images").glob("*"))
    random.seed(42)
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * VAL_RATIO))
    val, train = imgs[:n_val], imgs[n_val:]
    for split, items in (("train", train), ("val", val)):
        for sub in ("images", "labels"):
            d = DS / split / sub
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("*"):
                old.unlink()
        for p in items:
            shutil.copy2(p, DS / split / "images" / p.name)
            shutil.copy2(DS / "labels" / (p.stem + ".txt"),
                         DS / split / "labels" / (p.stem + ".txt"))
    cfg = {"path": str(DS), "train": "train/images", "val": "val/images",
           "names": {0: "piece", 1: "target"}, "kpt_shape": [1, 3]}
    (DS / "data.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    print(f"train={len(train)} val={len(val)}")


def export_android(weights: Path):
    """导出安卓部署权重。

    Windows 上 LiteRT/TFLite 导出不可用(仅 Linux x86/macOS), 故用 NCNN:
    ARM 上性能好, 体积约 12MB。
    """
    from ultralytics import YOLO
    m = YOLO(str(weights))
    m.export(format="ncnn", half=False)
    print("NCNN 导出完成, 见", weights.parent / f"{weights.stem}_ncnn_model")


def train():
    from ultralytics import YOLO
    model = YOLO("yolov8n-pose.pt")
    res = model.train(data=str(DS / "data.yaml"), epochs=200, imgsz=640,
                      batch=8, project=str(ROOT / "runs"), name="jump_pose",
                      hsv_h=0.02, hsv_s=0.5, hsv_v=0.4, scale=0.5, fliplr=0.5)
    # 用本次实际保存目录, 避免 ultralytics 自动加后缀后导出到旧模型
    best = Path(res.save_dir) / "weights" / "best.pt"
    print("best weights:", best)
    export_android(best)


if __name__ == "__main__":
    if build_labels() == 0:
        raise SystemExit("无标注数据")
    split_dataset()
    train()
