# android-apps

自己写的安卓应用，一个目录一个应用。放一起是为了以后好找：不用为一个小工具再开一个仓库。

| 应用 | 作用 | 目录 | 状态 |
| --- | --- | --- | --- |
| 跳一跳助手 | 微信小游戏「跳一跳」自动连跳：识别棋子与落点，按标定系数按压起跳 | [`jump-jump-assistant/`](jump-jump-assistant/) | 可用，实测单局 4040 分、命中中心率 90.2% |
| 喵喵助手 | 无障碍服务改写聊天文本，支持 QQ / 抖音 / 抖音极速版 | [`miao-assistant/`](miao-assistant/) | 可用，v1.2.0 |

## 怎么用

每个应用都是**自包含**的：进去看它自己的 `README.md`，怎么装、怎么用、哪里有坑都写在里面。

两个都靠系统敏感权限干活，装完都得手动开：

- **跳一跳助手**：屏幕录制（授权时**必须选「共享整个屏幕」**，选「共享一个应用」拿不到整屏画面）＋ 无障碍服务
- **喵喵助手**：无障碍服务（微信 8.0.6x+ 屏蔽第三方无障碍，该应用对微信永久不可用）

都只用 debug 签名、没有上架计划，直接装 `assembleDebug` 出来的包即可。

## 目录约定

```
<应用名>/
  README.md     这个应用怎么用、怎么验证、有什么坑
  android/      安卓工程（源码 + 资源 + 模型）
```

`jump-jump-assistant/` 在安卓工程之外还带一层 Python：识别模型是先在桌面端用这套脚本训出来、
再标定、再验证的（`train.py` 训练、`auto_annotate2.py` 预标注、`calib.py` 标按压系数、
`play.py` 跑整局、`app_monitor.py` 从外部采样分数算命中率）。训练数据、采集帧与训练产物
体积大且可复现，不随仓库分发，脚本里都写了怎么重新生成。

## 构建

各自目录里，用自带的 wrapper：

```bash
# 跳一跳助手
cd jump-jump-assistant/android && ./gradlew assembleDebug
# 按 ABI 出两个包: build/outputs/apk/debug/jump-assist-{arm64-v8a,x86_64}-debug.apk

# 喵喵助手
cd miao-assistant && ./gradlew assembleDebug
```

跳一跳助手要 JDK 17~21（**AGP 8.7.3 不接受 JDK 25**，报错只打版本号，很坑）；喵喵助手 JDK 17。
两个工程的 CI 都会在 push 时编 debug 包并上传 artifact。

## 许可

MIT，见 [LICENSE](LICENSE)。
