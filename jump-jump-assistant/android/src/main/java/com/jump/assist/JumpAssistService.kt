package com.jump.assist

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.os.Handler
import android.os.Looper
import android.view.accessibility.AccessibilityEvent

/**
 * 起跳执行器: 用无障碍手势完成一次按压, 按压时长 = 跳距(px) x 标定系数。
 *
 * 必须在系统设置里手动开启本服务(设置 -> 无障碍 -> 跳一跳助手)。
 * 说明: 本模块只做"按下-抬起", 不读取其它应用内容; 需要 android:canPerformGestures="true"。
 */
class JumpAssistService : AccessibilityService() {

    companion object {
        @Volatile
        var instance: JumpAssistService? = null
            private set
    }

    private val handler = Handler(Looper.getMainLooper())

    override fun onServiceConnected() {
        instance = this
    }

    override fun onDestroy() {
        instance = null
        super.onDestroy()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) = Unit

    override fun onInterrupt() = Unit

    /**
     * 执行一次跳跃按压。
     * @param x,y   按压点(屏幕坐标), 通常用棋子当前位置
     * @param durationMs 按压时长, 由 JumpCalibration.pressMs(跳距) 给出
     * @param done  回调, true 表示手势已被系统执行
     */
    fun press(x: Float, y: Float, durationMs: Long, done: ((Boolean) -> Unit)? = null) {
        val path = Path().apply { moveTo(x, y) }
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0, durationMs))
            .build()
        val ok = dispatchGesture(gesture, object : GestureResultCallback() {
            override fun onCompleted(gestureDescription: GestureDescription?) {
                done?.invoke(true)
            }

            override fun onCancelled(gestureDescription: GestureDescription?) {
                done?.invoke(false)
            }
        }, handler)
        if (!ok) done?.invoke(false)
    }

    /** 按识别结果起跳。 */
    fun jumpTo(res: JumpDetector.Result, done: ((Boolean) -> Unit)? = null): Boolean {
        val d = res.delta() ?: return false
        val p = res.piece ?: return false
        press(p.kpt[0], p.kpt[1], JumpCalibration.pressMs(d[2]), done)
        return true
    }
}
