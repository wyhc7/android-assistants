# 安卓端集成说明（跳一跳识别模块）

本目录是识别模块在安卓上的落地部分。**所有接口参数都在桌面端用导出后的模型实测过**，
不是推测值；实测方法与结论见文末「验证记录」。

---

## 0. 当前实现（2026-10-05 定稿）

推理后端从 NCNN 换成 **ONNX Runtime**：本机没有 NDK，JNI/CMake 那条路要多装一套工具链，
而 ONNX Runtime 只要一个 Gradle 依赖，模型也能用 ultralytics 直接导出。

### 0.1 模型导出（在仓库根目录执行）

```powershell
python -c "from ultralytics import YOLO; YOLO(r'runs\jump_pose_hsv\weights\best.pt').export(format='onnx', imgsz=640, opset=12)"
python -c "from ultralytics import YOLO; YOLO(r'runs\jump_pose-3\weights\best.pt').export(format='onnx', imgsz=640, opset=12)"
# 产物放到 android/src/main/assets/jump_hsv.onnx 与 jump_base.onnx
```

I/O 已核对：输入 `images` `1x3x640x640` float32 RGB/255 CHW，输出 `output0` `(1,9,8400)`，
布局与桌面端 NCNN 版完全一致（`[0..3]` 框 xywh、`[4..5]` 类别分、`[6..8]` 关键点 x,y,conf），
所以 letterbox 与坐标回映逻辑可以原样复用。

### 0.2 构建

```powershell
# AGP 8.7.3 不接受 JDK 25，必须用 JDK 17~21；Gradle 8.14 已在本机缓存
& "$env:USERPROFILE\.gradle\wrapper\dists\gradle-8.14-bin\*\gradle-8.14\bin\gradle.bat" `
    -p android assembleDebug --no-daemon `
    "-Dorg.gradle.java.home=C:\Users\w\jdk21t\jdk-21.0.12.1+1"
```

`settings.gradle.kts` 里配了阿里云镜像，依赖拉取快很多。`local.properties` 指向本机 SDK。

### 0.2.1 包体积

按 ABI 拆包（`splits.abi`），每个设备只装自己那一份：

| 产物 | 体积 | 用途 |
| --- | --- | --- |
| `android/build/outputs/apk/debug/jump-assist-arm64-v8a-debug.apk` | **42.5 MB** | 真机 |
| `android/build/outputs/apk/debug/jump-assist-x86_64-debug.apk` | **45.1 MB** | MuMu 等 x86_64 模拟器 |

不拆包时的 95.2 MB 构成（`libonnxruntime.so` 占 74%）：

| 项 | 大小 |
| --- | --- |
| `libonnxruntime.so` × 4 个 ABI（x86_64/x86/arm64/armv7） | 70.4 MB |
| `jump_hsv.onnx` + `jump_base.onnx` | 25.3 MB |
| `libonnxruntime4j_jni.so` × 4 | 3.2 MB |
| classes.dex + 资源 | 2.5 MB |

注意 `ndk.abiFilters` 与 `splits.abi` **不能同时设**，Gradle 会直接报
`Conflicting configuration ... cannot be present when splits abi filters are set`。

还想再小，只有三条路：模型转 FP16（省约一半模型体积，但要重验关键点精度）、只留一个模型
（省 12.6 MB，代价是丢掉集成互补）、或者换 NCNN 后端（原生库从 20 MB/ABI 降到 2~3 MB/ABI，
但需要 NDK 工具链）。

### 0.3 集成合并的一个坑（已修）

集成原按「每类取最高置信」合并，实测会出事：基线模型会把**棋子下方的方块**也识别成目标
（0.82），比主模型给出的正确目标（0.65）分还高 —— 合并后目标跑到棋子下方，触发「目标必须在
棋子上方」的规则，结果被判成「无目标」，白白漏跳（实测连续 6 跳误判）。
**几何规则必须在合并之前对每个模型分别生效**：先定棋子，再只在「棋子上方」的候选里取最高分。
Python（`calib.merge_ensemble`）与 Android（`JumpDetector.merge`）都按此修正。

### 0.4 悬浮窗与使用流程

悬浮窗是**紧凑条**：只放一个「▶ 连跳 / ■ 停止」按钮和一行状态，默认贴右上角（那里是天空，
不压棋盘），**可拖动**。窗口是 `FLAG_NOT_FOCUSABLE`，除这一小块外点击全部穿透。

```
1) 打开 App -> 「1. 授权截屏并开始」-> 选「共享整个屏幕」-> 确定
   （注意：必须选整屏。选「共享一个应用」时系统只给单应用画面，与整屏 VirtualDisplay 不兼容）
2) 「2. 显示悬浮窗」
3) 切到微信打开跳一跳，点悬浮条上的「▶ 连跳」
```

也可以一条命令直达自动模式（悬浮窗照常创建）：

```powershell
adb -s 127.0.0.1:16384 shell am start -n com.jump.assist/.MainActivity --ez auto true --ef k 1.35
```

`--ef k <float>` 是**运行时标定入口**：`adb input swipe` 与 App 内 `dispatchGesture` 的时序
并不完全等价，系数必须在设备上重标，有它就不用每换一个系数重新打包。每次起跳都会写日志：

```powershell
adb -s 127.0.0.1:16384 logcat -d | Select-String 'JumpAssist:'
# I JumpAssist: JUMP 12 d=435.0 press=587
```

### 0.5 App 自身路径实测

| 项 | 结果 |
| --- | --- |
| 按压系数 | `k=1.35`（与桌面端一致，未偏离） |
| 命中中心率 | **90.2%（46/51 得分事件）** |
| 单局表现 | 连续 97 跳、分数 1221 且仍在继续（全程无人工干预） |
| 识别 | 每跳前连抓两帧、关键点位移 ≤6px 才认，避免在落地动画中途按下一跳 |

命中率由**外部分数采样**判定（App 内不做 OCR）：`python app_monitor.py <csv> <秒>` 定时读分数，
`+1` 记普通落地、`+2/4/6…` 记命中中心（连续命中递增，上限 32）。

---

## 1. 模型

| 项 | 值 |
| --- | --- |
| 文件 | `model.ncnn.param` + `model.ncnn.bin`，主：`runs/jump_pose_hsv/weights/best_ncnn_model/`，辅：`runs/jump_pose-3/weights/best_ncnn_model/` |
| 体积 | 约 12 MB |
| 任务 | YOLOv8n-pose，2 类关键点：`0=piece`、`1=target`，`kpt_shape=[1,3]` |
| 输入 | blob `in0`，`1x3x640x640` float32，**RGB、/255、通道优先(CHW) 连续内存** |
| 输出 | blob `out0`，`(9, 8400)` |
| 输出布局 | `[0..3]` 框 `xywh`（640 空间像素）｜`[4..5]` 类别分（已 sigmoid）｜`[6..8]` 关键点 `x, y, conf`（640 空间像素） |

选用 `jump_pose_hsv`（颜色增广版）而不是更早的基线，是因为**换皮肤后基线会失效**：棋子皮肤
从深紫变成银色时，目标类仍正常（0.87–0.95），但棋子类分数从 0.9 塌到 0.004，59 帧里 50 帧
完全检不出。颜色增广版把"检不出"降到 5/59，旧皮肤反而更稳（棋子置信最小 0.509 → 0.901）。

**阈值必须按类别分开**：棋子 `0.20`、目标 `0.35`（`JumpDetector.CONF` 已是数组）。棋子类在
低对比度场景（银色棋子站在白色方块上）只有约 0.35，放宽到 0.20 可把跨皮肤检出率 93% → 97%，
而在旧皮肤 80 帧上实测零变化、无误报。

类别顺序由官方实现保证（`Detect._inference` 先拼框与类别分，`Pose._inference` 再拼关键点），
已在本地用同一张图对照官方推理逐像素核对。

### 1.1 推荐：双模型集成

两个模型强弱互补，实测同一个难帧上：
**药瓶类目标**基线 0.89 / 颜色增广版 0.30；**换皮肤后的银色棋子**基线检不出 / 颜色增广版可检出。
做法是同时加载两个模型、对每个类别各取置信度更高者：

| 配置 | 初始皮肤实测 | 换皮肤后实测 |
| --- | --- | --- |
| 颜色增广版单跑 | 34 跳 34 中（停在药瓶帧） | 60 跳 60 中 |
| **集成（颜色增广 + 基线）** | **120 跳 120 中（100%）** | 60 跳 60 中 |

集成在 72 张人工精标上：棋子 1.78px（PCK@5 97.2%）、落点 9.36px、违规 0。
代价是 2× 推理（桌面 37.5ms → ~75ms/帧，玩法只需 ~500ms 决策窗口），资源 2×12MB。
只装一个模型就用 `jump_pose_hsv`。

## 2. 预处理（必须与训练一致）

```
s    = min(640/W, 640/H)
nw   = round(W*s) ; nh = round(H*s)
ox   = (640-nw)/2 ; oy = (640-nh)/2      # 居中填充, 填充值 114
canvas[oy:oy+nh, ox:ox+nw] = resize(img, (nw, nh))
input = RGB(canvas) / 255, 转 CHW
```

**居中填充是实测结论**：同一张图，居中填充回映得到 `(770.5, 1086.0)`，官方 ultralytics NCNN
推理得到 `(770.6, 1086.0)`；左上贴边会偏 16 px 以上。不要改成贴边。

## 3. 坐标回映（640 空间 -> 屏幕）

```
x_screen = (x_640 - ox) / s
y_screen = (y_640 - oy) / s
```

## 4. 关键点语义（这是玩法唯一需要的量）

- `piece` 关键点 = **棋子底部中心**（棋子与所站方块接触的那一点）。
- `target` 关键点 = **落点**，即目标方块顶面菱形中心。
- **目标方块永远在棋子上方**（屏幕 y 更小）。`JumpDetector` 里已内置兜底：若模型给出的目标
  `y >= 棋子 y`，直接判定为「无目标」，交上层重截一帧，**绝不按下方方块起跳**。

跳距：`d = hypot(target.x - piece.x, piece.y - target.y)`。

## 5. 按压时长标定（必须实测）

按压时长与跳距在本游戏内近似线性：`press_ms = d * k`。`k` 随设备分辨率、游戏版本变化，
`JumpCalibration.msPerPx` 的默认值只是**初值**，必须标定：

1. 固定一局，记录 3 组「识别跳距 `d`（px）」与「刚好跳到中心所需的按压时长 `ms`」。
2. 线性拟合 `ms = k * d + b`，取 `k`（一般 `b` 很小可忽略）。
3. 写回 `JumpCalibration.msPerPx`。
4. 用第 4 组数据验证：误差落在 ±15 ms 内即可（约 ±1% 屏宽）。

## 6. 接入步骤

1. 拷模型：把 `best_ncnn_model/` 里的两个文件放到 `app/src/main/assets/jump_pose_ncnn/`。
2. 拷代码：`JumpDetector.kt`（推理）、`JumpAssistService.kt`（无障碍起跳）、`CaptureService.kt`
   与 `MainActivity.kt`（截屏与悬浮窗）、`jni/jump_jni.cpp`（JNI 桥）。
3. NDK 配置：`CMakeLists.txt` 已给最小示例，依赖 ncnn（官方 AAR 或预编译静态库）。
4. 权限与清单：见 `AndroidManifest.xml` 片段与 `accessibility_service_config.xml`。
5. 运行时顺序：授权截屏 -> 启动 `CaptureService` -> 开启无障碍 -> 悬浮窗点「识别并跳」。

`jump_jni.cpp` 里 `out0` 按 `9x8400` 连续内存整体拷贝，与 Kotlin 侧 `out[ch*8400 + anchor]`
的索引方式一致；`ncnn::Mat in_mat(640, 640, 3)` 的参数顺序是 `(w, h, c)`，别写反。

## 7. 已知边界

- 棋子框是**固定 60 px 的伪框**，只有它的中心有意义，框大小不代表棋子尺寸；目标框才是真实顶面框。
- 地面阴影已做乘法暗化建模并剔除；但背景出现硬地平线（墙/地两段纯色）且方块与背景亮度接近时，
  分割可能整片粘连，此时自动标注会留白，模型侧靠数据覆盖，识别不到目标就重截一帧。
- 目标方块在棋子上方是**游戏规则**；若某帧确实无上方方块（画面切换中），不要强行起跳。
- **必须先判定稳定态再起跳**：重启后的入场动画、落地后的镜头回位都会让棋子位置不可信
  （实测在入场动画中起跳，棋子被检到 y=1820 而非正常的 1000–1200，该跳直接失败）。
  做法：连续两帧截屏，两个关键点位置差 ≤8px 且置信度达标才起跳，否则重截（见 `emu_auto.py`
  的 `stable_detect`，这是安卓端实测出来的约束）。

## 8. 验证记录（桌面端已完成）

- `detect.py` 是与本目录逐行对应的桌面参考实现（同一 letterbox、同一输出解析、同一回映）。
- 用 `runs/jump_pose-2` 的 NCNN 模型跑 `dataset/val/images`：10/10 两目标齐全，
  piece 置信度 0.88–0.94，target 0.70–0.99。
- 与 ultralytics 官方 NCNN 推理在同一张图上坐标一致（误差 <0.2 px）。
- 关键点像素误差（PyTorch 模型，验证集）：棋子平均 2.74 px（PCK@5px 100%）、
  落点平均 6.15 px。
