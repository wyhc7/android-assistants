# -*- coding: utf-8 -*-
"""安卓端闭环实测: adb 截屏 -> NCNN 识别 -> adb 按压 -> 再截屏验证落点。

用法:
  python emu.py shot                     仅截屏+识别, 输出叠加图
  python emu.py jump <ms>                按识别距离用指定按压时长起跳, 并验证落点
  python emu.py calib <k>                用系数 k(ms/px) 起跳一次(标定用)
  python emu.py tap <x> <y> <ms>         直接在指定位置按压(手动测试用)

说明: adb 的 "input swipe x y x y <ms>" 与 App 内 AccessibilityService 的
dispatchGesture 是同一动作(定点按压指定时长), 因此本脚本能完整验证玩法闭环。
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2

from detect import Detector

ADB = r"D:\Program Files\Netease\MuMu\nx_device\15.0\shell\adb.exe"
DEV = "127.0.0.1:16384"
ROOT = Path(__file__).parent
# 每轮采集放到独立子目录: 复用同名文件会覆盖上一轮的帧, 导致已建好的标注与图片错配
EMU = ROOT / "emulator" / os.environ.get("EMU_SUBDIR", "")
MODEL = ROOT / "runs" / "jump_pose-3" / "weights" / "best_ncnn_model"
EMU.mkdir(parents=True, exist_ok=True)

# 换皮肤后棋子类分数可能整体塌陷(例如银色棋子只有 0.10), 用环境变量临时放宽棋子阈值
PIECE_CONF = float(os.environ.get("PIECE_CONF", "0.35"))
# 目标类阈值不能贴着实际分布: 药瓶类目标连颜色增广版也只有 ~0.30, 真机上基线模型退化到
# 0.057 后就会整帧判成"无目标"。正常阈值 0.20, 检不出时用 RELAX 再兜一次。
TARGET_CONF = float(os.environ.get("TARGET_CONF", "0.20"))
RELAX_TARGET_CONF = float(os.environ.get("RELAX_TARGET_CONF", "0.08"))

det = Detector(MODEL, (PIECE_CONF, TARGET_CONF))
det_lo = Detector(MODEL, (PIECE_CONF, RELAX_TARGET_CONF))


def sh(*args, timeout=30):
    return subprocess.run([ADB, "-s", DEV, *args], capture_output=True, timeout=timeout)


def capture(tag, tries=3):
    """截当前屏幕到 emulator/<tag>.png 并返回位图。

    screencap 偶尔会写出半截 PNG(读出来 libpng error), 所以失败要重试,
    否则一次坏图会被上层误判成"识别失败/游戏结束"。
    """
    remote = "/sdcard/_emu.png"
    local = EMU / ("%s.png" % tag)
    for t in range(tries):
        if local.exists():
            local.unlink()
        sh("shell", "screencap", "-p", remote)
        sh("pull", remote, str(local))
        img = cv2.imread(str(local)) if local.exists() else None
        if img is not None and img.size > 0:
            return img
        time.sleep(0.3)
    return None


def press(x, y, ms):
    """定点按压 ms 毫秒。PRESS_MODE=sendevent 时绕开 input swipe 的时序抖动。"""
    if os.environ.get("PRESS_MODE", "swipe") == "sendevent":
        return press_precise(x, y, ms)
    sh("shell", "input", "swipe", str(int(x)), str(int(y)), str(int(x)), str(int(y)), str(int(ms)))


def press_precise(x, y, ms, dev=None):
    """直接写 /dev/input 事件: 按下 -> sleep(ms) -> 抬起。

    与 input swipe 的区别: 关键区间(按下到抬起)由设备端 sleep 控制, 不再受
    adb -> shell -> input 这条链上每步调度抖动的影响。
    """
    dev = dev or os.environ.get("TOUCH_DEV", "/dev/input/event4")
    x, y = int(x), int(y)
    down = ["sendevent %s 3 47 0" % dev, "sendevent %s 3 57 1" % dev,
            "sendevent %s 3 53 %d" % (dev, x), "sendevent %s 3 54 %d" % (dev, y),
            "sendevent %s 0 0 0" % dev]
    up = ["sendevent %s 3 47 0" % dev, "sendevent %s 3 57 -1" % dev, "sendevent %s 0 0 0" % dev]
    return sh("shell", "; ".join(down) + "; sleep %.3f; " % (ms / 1000.0) + "; ".join(up))


def analyze(img, tag):
    res = det.infer(img)
    out = EMU / ("%s_det.png" % tag)
    cv2.imwrite(str(out), D_draw(img, res))
    return res, out


def D_draw(img, res):
    import detect
    return detect.draw(img, res)


def report(res):
    if not res[0] or not res[1]:
        print("  piece=%s target=%s" % (res[0] is not None, res[1] is not None))
        return None
    p, t = res[0]["kpt"], res[1]["kpt"]
    dx, dy = t[0] - p[0], p[1] - t[1]
    d = (dx * dx + dy * dy) ** 0.5
    print("  棋子底部中心 (%.0f, %.0f) conf=%.2f" % (p[0], p[1], res[0]["conf"]))
    print("  落点         (%.0f, %.0f) conf=%.2f" % (t[0], t[1], res[1]["conf"]))
    print("  位移 dx=%.0f dy=%.0f(正=目标在上) 跳距=%.1fpx" % (dx, dy, d))
    return d


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "shot"
    if cmd == "run":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        k = float(sys.argv[3]) if len(sys.argv) > 3 else 1.38
        for i in range(1, n + 1):
            img = capture("r%02d_pre" % i)
            res, _ = analyze(img, "r%02d_pre" % i)
            if not res[0] or not res[1]:
                print("第 %d 跳: 识别失败, 停止" % i)
                return
            p, t = res[0]["kpt"], res[1]["kpt"]
            d = ((t[0] - p[0]) ** 2 + (p[1] - t[1]) ** 2) ** 0.5
            ms = d * k
            print("第 %d 跳: 跳距=%.1fpx 按压=%dms (k=%.3f) conf=%.2f/%.2f"
                  % (i, d, ms, k, res[0]["conf"], res[1]["conf"]))
            press(p[0], p[1], ms)
            time.sleep(1.7)
            img2 = capture("r%02d_post" % i)
            analyze(img2, "r%02d_post" % i)
        print("完成 %d 跳, 帧存于 emulator/" % n)
        return

    if cmd == "shot":
        img = capture("now")
        print("截图:", img.shape)
        res, out = analyze(img, "now")
        report(res)
        print("叠加图:", out)
        return

    if cmd == "tap":
        x, y, ms = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        press(x, y, ms)
        print("已按压 (%d,%d) %dms" % (x, y, ms))
        return

    # 起跳: 先识别, 再按压, 再验证
    img = capture("pre")
    res, out = analyze(img, "pre")
    report(res)
    if not res[0] or not res[1]:
        print("识别失败, 不跳")
        return
    p, t = res[0]["kpt"], res[1]["kpt"]
    d = ((t[0] - p[0]) ** 2 + (p[1] - t[1]) ** 2) ** 0.5
    if cmd == "jump":
        ms = float(sys.argv[2])
    else:
        k = float(sys.argv[2])
        ms = d * k
    print("按压 %dms (跳距 %.1fpx)" % (ms, d))
    press(p[0], p[1], ms)
    time.sleep(1.6)
    img2 = capture("post")
    res2, out2 = analyze(img2, "post")
    if res2[0]:
        got = res2[0]["kpt"]
        print("落地后棋子 (%.0f, %.0f) | 与目标落点偏差 dx=%.0f dy=%.0f 距离=%.0fpx"
              % (got[0], got[1], got[0] - t[0], got[1] - t[1],
                 ((got[0] - t[0]) ** 2 + (got[1] - t[1]) ** 2) ** 0.5))
    else:
        print("落地后未识别到棋子(可能已坠落或画面已重置)")
    print("叠加图:", out2)


if __name__ == "__main__":
    main()
