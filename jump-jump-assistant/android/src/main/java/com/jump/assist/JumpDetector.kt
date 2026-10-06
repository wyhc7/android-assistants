package com.jump.assist

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 跳一跳识别模块: ONNX Runtime 推理 + 与训练完全一致的 letterbox + 坐标回映。
 *
 * 为什么用 ONNX Runtime 而不是 NCNN: 本机没有 NDK, JNI/CMake 路线需要额外工具链;
 * ONNX Runtime 只需一个 Gradle 依赖, 且模型已用 ultralytics 直接导出。
 *
 * 规格(已实测, 与桌面 NCNN 版本逐位对齐):
 *   输入  "images":  1x3x640x640 float32, RGB, /255, 通道优先(CHW)
 *   letterbox: 等比缩放 min(640/W, 640/H), 居中贴到 114 灰底
 *   输出  "output0": (1, 9, 8400) = [0..3]框 xywh(640空间) | [4..5]类别分 | [6..8]关键点 x,y,conf
 *   坐标回映: orig = (coord640 - pad) / scale
 *
 * 关键点语义:
 *   类别 0 = 棋子, 关键点 = 棋子底部中心(与方块接触的点)
 *   类别 1 = 目标, 关键点 = 落点(目标方块顶面中心)。目标方块永远在棋子上方。
 *
 * 集成: 同时加载多个模型, 每个类别取所有模型里置信度最高的候选。
 * 两个模型弱点互补 —— hsv 模型认得出白底上的银棋子, 基线模型认得出瓶子类目标;
 * 实测(初始皮肤 120 跳)两者单独用都会偶发漏检, 取高分后 120/120 全部识别成功。
 */
class JumpDetector private constructor(
    private val env: OrtEnvironment,
    private val sessions: List<OrtSession>,
    private val inName: String,
    private val outName: String,
) {

    companion object {
        const val IMGSZ = 640
        const val PAD = 114

        /**
         * 分类别阈值。棋子类在"换皮肤/低对比度"场景下分数会明显偏低(银色棋子站在白色方块上
         * 约 0.35、站在深灰圆柱上约 0.20), 放宽到 0.15 可把跨皮肤检出率从 93%(0.35) 提到 98%
         * (实测曲线: 0.35→93%, 0.20→97%, 0.10→98%), 而在旧皮肤 80 帧与 72 张人工精标上
         * **零变化**(位置与误差都不变, 无误报)。
         *
         * 目标类原来是 0.35, 这个值一直贴着实际分布: 药瓶类目标连颜色增广版也只有 ~0.30,
         * 靠基线模型(0.89)才过得去; 真机上基线退化到 0.057 后, 只剩 0.347, 整帧判成"无目标"
         * (实测真机截图, 分数停在 4866)。改成 0.20, 并在检不出时用 CONF_RELAX 再兜一次。
         * 实测: 400 张真实对局帧上 0.35→0.20→0.10 的选择完全一致(零漏检零误选);
         * 72 张人工精标上阈值从 0.35 扫到 0.05 各项指标不动 —— 真正的过滤是几何规则(落点必须
         * 在棋子上方), 阈值只是多余的闸门。 */
        val CONF = floatArrayOf(0.15f, 0.20f)

        /** 兜底阈值: 正常阈值整帧检不出目标时再降到这里重解一次。 */
        val CONF_RELAX = floatArrayOf(0.15f, 0.08f)
        const val CLS_PIECE = 0
        const val CLS_TARGET = 1
        private const val NUM_ANCHORS = 8400
        private const val NUM_CH = 9
        private const val PLANE = IMGSZ * IMGSZ

        /** 从 assets 里的 onnx 建会话。传入多个文件名即启用集成。 */
        fun fromAssets(ctx: Context, vararg assetNames: String): JumpDetector {
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) }
            val sessions = assetNames.map { name ->
                val bytes = ctx.assets.open(name).use { it.readBytes() }
                env.createSession(bytes, opts)
            }
            require(sessions.isNotEmpty()) { "未加载任何模型" }
            val first = sessions.first()
            return JumpDetector(
                env, sessions,
                first.inputNames.first(),
                first.outputNames.first(),
            )
        }
    }

    /** 一条检测结果: 屏幕坐标。 */
    data class Hit(val cls: Int, val conf: Float, val box: FloatArray, val kpt: FloatArray)

    data class Result(val piece: Hit?, val target: Hit?) {
        val jumpable: Boolean get() = piece != null && target != null

        /** 落点相对棋子底部中心的位移与跳距(像素)。 */
        fun delta(): FloatArray? {
            val p = piece ?: return null
            val t = target ?: return null
            val dx = t.kpt[0] - p.kpt[0]
            val dy = p.kpt[1] - t.kpt[1]      // 正数表示目标在上方
            return floatArrayOf(dx, dy, sqrt(dx * dx + dy * dy))
        }
    }

    fun close() {
        sessions.forEach { it.close() }
    }

    /** 完整推理: 位图 -> 结果(屏幕坐标)。 */
    fun detect(bmp: Bitmap): Result {
        val w = bmp.width
        val h = bmp.height
        val scale = min(IMGSZ.toFloat() / w, IMGSZ.toFloat() / h)
        val nw = (w * scale).toInt()
        val nh = (h * scale).toInt()
        val padX = (IMGSZ - nw) / 2
        val padY = (IMGSZ - nh) / 2

        val input = letterbox(bmp, nw, nh, padX, padY)
        val perModel = ArrayList<Array<Hit?>>(sessions.size)
        val relaxed = ArrayList<Array<Hit?>>(sessions.size)
        val tensor = OnnxTensor.createTensor(env, input, longArrayOf(1, 3, IMGSZ.toLong(), IMGSZ.toLong()))
        tensor.use { t ->
            for (s in sessions) {
                s.run(mapOf(inName to t)).use { out ->
                    @Suppress("UNCHECKED_CAST")
                    val planes = (out[0].value as Array<Array<FloatArray>>)[0]
                    perModel.add(decode(planes, scale, padX, padY, CONF))
                    // 同一份输出再解一次, 只是阈值放宽, 几乎不花时间
                    relaxed.add(decode(planes, scale, padX, padY, CONF_RELAX))
                }
            }
        }
        val strict = merge(perModel)
        if (strict.target != null) return strict
        // 正常阈值下目标检不出时再放宽一次。真机渲染会让目标类置信整体下移(实测同一帧模拟器
        // 0.89 / 真机 0.057), 贴着阈值就会整帧判成"无目标"而空等。
        val loose = merge(relaxed)
        return if (loose.target != null) loose else strict
    }

    /**
     * 集成合并: 棋子先定, 目标只在"棋子上方"的候选里取最高分。
     *
     * 直接按置信度合并会出事 —— 实测基线模型会把**棋子下方的方块**也当成目标(0.82),
     * 比 hsv 模型给出的正确目标(0.65)分还高, 合并后目标跑到下方, 触发"目标必须在棋子上方"
     * 的规则, 结果被判成"无目标"而白白漏跳。所以几何规则必须在合并**之前**逐模型生效。
     */
    private fun merge(perModel: List<Array<Hit?>>): Result {
        val piece = perModel.mapNotNull { it[CLS_PIECE] }.maxByOrNull { it.conf }
        val target = perModel.mapNotNull { it[CLS_TARGET] }
            .filter { piece != null && it.kpt[1] < piece.kpt[1] }
            .maxByOrNull { it.conf }
        return Result(piece, target)
    }

    /** 等比缩放 -> 居中贴到 114 灰底 -> CHW float32 RGB /255。 */
    private fun letterbox(bmp: Bitmap, nw: Int, nh: Int, padX: Int, padY: Int): FloatBuffer {
        val canvas = Bitmap.createBitmap(IMGSZ, IMGSZ, Bitmap.Config.ARGB_8888)
        Canvas(canvas).apply {
            drawColor(Color.rgb(PAD, PAD, PAD))
            drawBitmap(
                bmp, null,
                android.graphics.Rect(padX, padY, padX + nw, padY + nh),
                Paint(Paint.FILTER_BITMAP_FLAG)
            )
        }
        val pixels = IntArray(PLANE)
        canvas.getPixels(pixels, 0, IMGSZ, 0, 0, IMGSZ, IMGSZ)
        canvas.recycle()

        val buf = ByteBuffer.allocateDirect(3 * PLANE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        val arr = FloatArray(3 * PLANE)
        for (i in pixels.indices) {
            val c = pixels[i]
            arr[i] = ((c shr 16) and 0xFF) / 255f            // R
            arr[PLANE + i] = ((c shr 8) and 0xFF) / 255f     // G
            arr[2 * PLANE + i] = (c and 0xFF) / 255f         // B
        }
        buf.put(arr).rewind()
        return buf
    }

    /** 单个模型: 每类取最优候选(屏幕坐标)。 */
    private fun decode(
        planes: Array<FloatArray>, scale: Float, padX: Int, padY: Int, conf: FloatArray
    ): Array<Hit?> {
        val best = arrayOfNulls<Hit>(2)
        for (cls in 0..1) {
            val row = planes[4 + cls]
            var bi = -1
            var bc = 0f
            for (a in 0 until NUM_ANCHORS) {
                val s = row[a]
                if (s > bc) {
                    bc = s
                    bi = a
                }
            }
            if (bi < 0 || bc < conf[cls]) continue
            val cx = planes[0][bi]
            val cy = planes[1][bi]
            val bw = planes[2][bi]
            val bh = planes[3][bi]
            best[cls] = Hit(
                cls, bc,
                floatArrayOf(
                    (cx - bw / 2 - padX) / scale, (cy - bh / 2 - padY) / scale,
                    (cx + bw / 2 - padX) / scale, (cy + bh / 2 - padY) / scale
                ),
                floatArrayOf(
                    (planes[6][bi] - padX) / scale,
                    (planes[7][bi] - padY) / scale,
                    planes[8][bi]
                )
            )
        }
        return best
    }

    /** 识别结果叠加到原图, 用于人工核对。 */
    fun overlay(src: Bitmap, res: Result): Bitmap {
        val out = src.copy(Bitmap.Config.ARGB_8888, true)
        val cv = Canvas(out)
        val boxPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f }
        val markPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f }
        res.piece?.let {
            boxPaint.color = Color.RED
            markPaint.color = Color.RED
            cv.drawRect(it.box[0], it.box[1], it.box[2], it.box[3], boxPaint)
            cv.drawCircle(it.kpt[0], it.kpt[1], 12f, markPaint)
        }
        res.target?.let {
            boxPaint.color = Color.GREEN
            markPaint.color = Color.MAGENTA
            cv.drawRect(it.box[0], it.box[1], it.box[2], it.box[3], boxPaint)
            cv.drawCircle(it.kpt[0], it.kpt[1], 18f, markPaint)
            cv.drawLine(it.kpt[0] - 30, it.kpt[1], it.kpt[0] + 30, it.kpt[1], markPaint)
            cv.drawLine(it.kpt[0], it.kpt[1] - 30, it.kpt[0], it.kpt[1] + 30, markPaint)
        }
        return out
    }
}

/** 跳距 -> 按压时长。k 必须实测标定: 同局内交替两个 k 按跳距配对比较命中率, 取命中率最高者。 */
object JumpCalibration {
    /**
     * 实测(模拟器 1080x1920, adb input swipe, 每次按压 = k x 跳距):
     *   k=1.33 → 68% | k=1.34 → 88~90% | k=1.35 → 97% | k=1.38 → 68%
     * 峰值在 1.35, 两侧下降很快, 换设备/分辨率后必须重新标定。
     */
    var msPerPx: Float = 1.35f
    var minMs: Long = 40
    var maxMs: Long = 1600

    fun pressMs(distancePx: Float): Long =
        (distancePx * msPerPx).toLong().coerceIn(minMs, maxMs)
}
