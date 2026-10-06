"""App 自己连跳时, 从桌面侧定时采样分数, 用于还原每一跳的得分增量。

分数增量是游戏给的真值: +1 = 普通落地, +2,4,6... = 命中中心(连续命中递增, 上限 32)。
App 内部不做 OCR, 所以命中率必须在外部量。

用法: python app_monitor.py [输出csv] [时长秒]
"""
import csv
import os
import sys
import time

import emu
from score_ocr import read

OUT = sys.argv[1] if len(sys.argv) > 1 else "emulator/app_mon.csv"
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
t0 = time.time()
n = 0
last = None
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["t", "score"])
    while time.time() - t0 < DUR:
        img = emu.capture("mon")
        s = read(img) if img is not None else None
        w.writerow(["%.2f" % (time.time() - t0), "" if s is None else s])
        f.flush()
        n += 1
        if s is not None and s != last:
            last = s
        time.sleep(0.3)
print("采样 %d 次, 写入 %s" % (n, OUT))
