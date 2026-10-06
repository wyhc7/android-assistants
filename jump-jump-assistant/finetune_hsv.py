# -*- coding: utf-8 -*-
"""颜色不变性微调: 用强颜色增广逼模型改用"形状"而不是"颜色"识别棋子。

背景: 换皮肤后棋子从深紫变成银色, 分类头分数从 0.9 塌到 0.004(关键点头仍准),
说明分类头过度依赖颜色。hsv_s 是饱和度乘性缩放(可到 0.3), 配合色相抖动与亮度缩放
能造出"银灰棋子", 从而在不采集新皮肤数据的前提下获得跨皮肤泛化能力。

用法: python finetune_hsv.py [epochs] [lr] [hsv_h] [hsv_s] [hsv_v] [name]
"""
import os
import sys
from pathlib import Path

import finetune

ROOT = Path(__file__).parent
# finetune 在导入时用 sys.argv 取 BASE, 会和本脚本的参数冲突, 这里显式覆盖;
# 也可用 BASE_PT 指定别的起点权重(例如从颜色增广版继续训)。
finetune.BASE = Path(os.environ.get("BASE_PT", str(ROOT / "runs" / "jump_pose-3" / "weights" / "best.pt")))
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
HSV_H = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
HSV_S = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
HSV_V = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
NAME = sys.argv[6] if len(sys.argv) > 6 else "jump_pose_hsv"


def main():
    from ultralytics import YOLO
    if finetune.build() == 0:
        raise SystemExit("无数据")
    finetune.split()
    model = YOLO(str(finetune.BASE))
    res = model.train(data=str(finetune.DS / "data.yaml"), epochs=EPOCHS, imgsz=640, batch=16,
                      lr0=LR, lrf=0.1, project=str(ROOT / "runs"), name=NAME,
                      hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V, scale=0.5)
    best = Path(res.save_dir) / "weights" / "best.pt"
    print("best:", best)
    YOLO(str(best)).export(format="ncnn", half=False)
    print("NCNN 导出:", best.parent / "best_ncnn_model")


if __name__ == "__main__":
    main()
