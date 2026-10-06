package com.jump.assist

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.abs

/**
 * 主界面: 申请截屏权限 -> 启动截屏服务 -> 悬浮窗显示识别结果, 可单次起跳或自动连跳。
 *
 * 使用前需:
 *   1) 系统设置里开启本应用的"无障碍服务"(JumpAssistService)
 *   2) 授予"悬浮窗"权限
 *   3) 首次运行同意截屏授权弹窗
 */
class MainActivity : Activity() {

    private val TAG = "JumpAssist"
    private val REQ_PROJECTION = 1001
    private val REQ_OVERLAY = 1002

    /** 判稳阈值: 连抓两帧的四组关键点坐标位移都不超过它, 才认为画面已停。 */
    private val TOL = 6f

    private lateinit var detector: JumpDetector
    private var floatView: View? = null
    private var floatLp: WindowManager.LayoutParams? = null
    private var autoButton: Button? = null
    private var preview: ImageView? = null
    private lateinit var status: TextView

    private val handler = Handler(Looper.getMainLooper())

    /** 拖动悬浮条的落点偏移。 */
    private var dragDx = 0f
    private var dragDy = 0f

    /** 自动连跳状态。 */
    @Volatile
    private var autoRun = false
    private var autoThread: Thread? = null

    /** 用 "am start --ez auto true" 启动时, 授权截屏后自动开始连跳(便于脚本化验证)。 */
    private var autoBoot = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        detector = JumpDetectorHolder.get(this)
        applyK(intent)
        Log.i(TAG, "onCreate auto=${intent?.getBooleanExtra("auto", false)} k=${JumpCalibration.msPerPx}")

        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 96, 32, 32)
            addView(Button(this@MainActivity).apply {
                text = "1. 授权截屏并开始"
                setOnClickListener { requestProjection() }
            })
            addView(Button(this@MainActivity).apply {
                text = "2. 显示悬浮窗"
                setOnClickListener { showFloatWindow() }
            })
            addView(Button(this@MainActivity).apply {
                text = "3. 打开无障碍设置(起跳需要)"
                setOnClickListener {
                    startActivity(Intent(android.provider.Settings.ACTION_ACCESSIBILITY_SETTINGS))
                }
            })
            status = TextView(this@MainActivity).apply { text = "未开始" }
            addView(status)
        })

        // 注意顺序: setContentView 必须在前, 否则它会覆盖 status 字段, 悬浮窗上的状态文字就再也不更新了
        autoBoot = intent?.getBooleanExtra("auto", false) == true
        if (autoBoot) {
            // 悬浮窗便于随时启停; 面板很小且贴右上角, 不会挡住棋盘
            showFloatWindow()
            requestProjection()
        }
    }

    private fun requestProjection() {
        val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        startActivityForResult(mpm.createScreenCaptureIntent(), REQ_PROJECTION)
    }

    /**
     * 已经在前台/后台时再次 "am start --ez auto true", 走的是这里而不是 onCreate,
     * 所以自动启动逻辑必须在这里再判一次, 否则第二次以后就静默不生效。
     */
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        applyK(intent)
        if (!intent.getBooleanExtra("auto", false)) return
        autoBoot = true
        if (CaptureService.instance != null) {
            handler.postDelayed({ toggleAuto() }, 500)
        } else {
            requestProjection()
        }
    }

    /**
     * 运行时标定: "am start ... --ef k 1.42" 直接改按压系数, 不必重新打包。
     * adb input swipe 与 App 内 dispatchGesture 的时序不完全等价, 系数必须在设备上重标。
     */
    private fun applyK(intent: Intent?) {
        val k = intent?.getFloatExtra("k", 0f) ?: 0f
        if (k > 0f) {
            JumpCalibration.msPerPx = k
            Log.i(TAG, "k set to $k")
        }
    }

    @Deprecated("startActivityForResult 已废弃, 生产代码改用 ActivityResultLauncher")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        Log.i(TAG, "onActivityResult req=$requestCode ok=${resultCode == Activity.RESULT_OK} autoBoot=$autoBoot")
        if (requestCode == REQ_PROJECTION && resultCode == Activity.RESULT_OK && data != null) {
            startForegroundService(Intent(this, CaptureService::class.java).apply {
                putExtra(CaptureService.EXTRA_CODE, resultCode)
                putExtra(CaptureService.EXTRA_DATA, data)
            })
            status.text = "截屏已启动"
            if (autoBoot) handler.postDelayed({ toggleAuto() }, 2000)
        }
    }

    /**
     * 悬浮窗: 只留一个"开始/停止连跳 + 一行状态"的紧凑条, 默认贴右上角, 可拖动。
     * 面板做成 FLAG_NOT_FOCUSABLE 且尽量小, 棋盘区域和操作都不受影响。
     */
    private fun showFloatWindow() {
        if (floatView != null) return
        val wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0x88000000.toInt())
            setPadding(10, 10, 10, 10)
            addView(Button(this@MainActivity).apply {
                text = "▶ 连跳"
                textSize = 12f
                setPadding(10, 0, 10, 0)
                setOnClickListener { toggleAuto() }
            }.also { autoButton = it })
            addView(TextView(this@MainActivity).apply {
                setTextColor(0xFFFFFFFF.toInt())
                textSize = 10f
                text = "就绪"
            }.also { status = it })
        }
        val type = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O)
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        else
            @Suppress("DEPRECATION") WindowManager.LayoutParams.TYPE_PHONE
        val lp = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            type, WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE, PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            // 贴右上角: 那里通常是天空/背景, 不压棋盘
            x = resources.displayMetrics.widthPixels - 300
            y = 240
        }
        bar.setOnTouchListener(dragListener(wm, lp))
        wm.addView(bar, lp)
        floatView = bar
        floatLp = lp
    }

    /** 拖动悬浮条(按钮本身仍可点击: 位移小于阈值就不算拖动)。 */
    private fun dragListener(wm: WindowManager, lp: WindowManager.LayoutParams) =
        View.OnTouchListener { _, e ->
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    dragDx = e.rawX - lp.x
                    dragDy = e.rawY - lp.y
                    false
                }

                MotionEvent.ACTION_MOVE -> {
                    if (Math.abs(e.rawX - lp.x - dragDx) > 8 || Math.abs(e.rawY - lp.y - dragDy) > 8) {
                        lp.x = (e.rawX - dragDx).toInt()
                        lp.y = (e.rawY - dragDy).toInt()
                        wm.updateViewLayout(floatView, lp)
                        true
                    } else {
                        false
                    }
                }

                else -> false
            }
        }

    /** 截一帧 -> 识别 -> 画叠加 -> 可选起跳。 */
    private fun recognize(jump: Boolean) {
        val svc = CaptureService.instance
        if (svc == null) {
            status.text = "截屏服务未启动"
            return
        }
        val shot = svc.capture()
        if (shot == null) {
            status.text = "截屏失败(等待一帧后重试)"
            return
        }
        val res = detector.detect(shot)
        preview?.setImageBitmap(detector.overlay(shot, res))

        val d = res.delta()
        if (res.piece == null || res.target == null) {
            status.text = if (res.piece == null) "未识别到棋子" else "未识别到上方目标方块"
            return
        }
        val ms = JumpCalibration.pressMs(d!![2])
        status.text = "距离=%.0fpx 按压=%dms".format(d[2], ms)

        if (jump) {
            val a11y = JumpAssistService.instance
            if (a11y == null) {
                status.text = "无障碍服务未开启, 无法起跳"
                return
            }
            // 等预览刷新后再发手势, 避免与服务自身 UI 抢占
            handler.postDelayed({
                a11y.press(res.piece!!.kpt[0], res.piece!!.kpt[1], ms) { ok ->
                    runOnUiThread { status.text = if (ok) "已起跳 %dms".format(ms) else "手势被拒绝" }
                }
            }, 120)
        }
    }

    // ---------- 自动连跳 ----------

    private fun toggleAuto() {
        Log.i(TAG, "toggleAuto autoRun=$autoRun cap=${CaptureService.instance != null} a11y=${JumpAssistService.instance != null}")
        if (autoRun) {
            autoRun = false
            status.text = "正在停止..."
            return
        }
        if (CaptureService.instance == null) {
            status.text = "截屏服务未启动"
            return
        }
        if (JumpAssistService.instance == null) {
            status.text = "无障碍服务未开启, 无法起跳"
            return
        }
        autoRun = true
        autoButton?.text = "■ 停止"
        autoThread = Thread { autoLoop() }.also { it.isDaemon = true; it.start() }
    }

    /**
     * 自动连跳主循环。判稳方式与桌面版一致: 连抓两帧, 四组关键点坐标位移都小于 TOL 才认,
     * 避免在落地动画中途按出下一跳(那是实测里最主要的一类误跳)。
     */
    private fun autoLoop() {
        var jumps = 0
        var fails = 0
        Log.i(TAG, "autoLoop start")
        while (autoRun && jumps < 500) {
            // 服务可能被系统重启, 每轮重新取; 拿不到就等, 绝不静默退出
            val svc = CaptureService.instance
            if (svc == null) {
                post("等待截屏服务...")
                if (++fails >= 20) break
                sleep(500)
                continue
            }
            val res = measure(svc)
            val d = res?.delta()
            if (res == null || d == null) {
                fails++
                post("等待识别 %d/120".format(fails))
                // 容忍 2 分钟: 授权后用户还要切回游戏(小程序要重新打开), 期间画面里什么都没有
                if (fails >= 120) break
                sleep(1000)
                continue
            }
            fails = 0
            val p = res.piece ?: break
            val ms = JumpCalibration.pressMs(d[2])
            Log.i(TAG, "JUMP %d d=%.1f press=%d".format(jumps + 1, d[2], ms))
            post("第 %d 跳: 距离=%.0fpx 按压=%dms".format(jumps + 1, d[2], ms))
            val latch = CountDownLatch(1)
            handler.post {
                JumpAssistService.instance?.press(p.kpt[0], p.kpt[1], ms) { latch.countDown() }
            }
            latch.await(3, TimeUnit.SECONDS)
            jumps++
            sleep(880)      // 等落地动画与镜头归位
        }
        autoRun = false
        Log.i(TAG, "autoLoop end jumps=$jumps fails=$fails")
        post("自动连跳结束, 共 %d 次".format(jumps))
        runOnUiThread { autoButton?.text = "▶ 连跳" }
    }

    /** 连抓两帧, 关键点位移小于 TOL 才认为画面已稳。 */
    private fun measure(svc: CaptureService): JumpDetector.Result? {
        var last: JumpDetector.Result? = null
        var lastK: FloatArray? = null
        repeat(8) {
            if (!autoRun) return null
            val shot = svc.capture()
            if (shot == null) {
                sleep(250)
                return@repeat
            }
            val r = detector.detect(shot)
            val k = keypoints(r)
            Log.i(TAG, "measure piece=${r.piece != null} target=${r.target != null}")
            val prev = lastK
            if (k != null && prev != null && TOL >= maxOf(
                    abs(k[0] - prev[0]), abs(k[1] - prev[1]),
                    abs(k[2] - prev[2]), abs(k[3] - prev[3])
                )
            ) {
                post("距离=%.0fpx".format(r.delta()!![2]))
                return r
            }
            last = r
            lastK = k
            sleep(200)
        }
        return last
    }

    /** [棋子x, 棋子y, 目标x, 目标y], 缺一个就是 null。 */
    private fun keypoints(r: JumpDetector.Result): FloatArray? {
        val p = r.piece ?: return null
        val t = r.target ?: return null
        return floatArrayOf(p.kpt[0], p.kpt[1], t.kpt[0], t.kpt[1])
    }

    private fun post(text: String) = runOnUiThread { status.text = text }

    private fun sleep(ms: Long) {
        try {
            Thread.sleep(ms)
        } catch (_: InterruptedException) {
        }
    }

    override fun onDestroy() {
        autoRun = false
        floatView?.let {
            (getSystemService(Context.WINDOW_SERVICE) as WindowManager).removeView(it)
        }
        floatView = null
        super.onDestroy()
    }
}

/**
 * 识别器单例。模型直接以 assets 里的 onnx 字节建 ONNX Runtime 会话(不需要解包到私有目录)。
 *
 * 传两个模型即启用集成: 每个类别取所有模型里置信度最高的候选。实测(初始皮肤)两个模型
 * 单独用都会偶发漏检 —— hsv 模型能认白底上的银棋子, 基线模型能认瓶子类目标 —— 取高分后
 * 120/120 全部识别成功。
 */
object JumpDetectorHolder {
    private val ASSETS = arrayOf("jump_hsv.onnx", "jump_base.onnx")

    @Volatile
    private var inst: JumpDetector? = null

    fun get(ctx: Context): JumpDetector =
        inst ?: synchronized(this) {
            inst ?: JumpDetector.fromAssets(ctx.applicationContext, *ASSETS).also { inst = it }
        }
}
