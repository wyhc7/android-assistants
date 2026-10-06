package com.jump.assist

import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.DisplayMetrics
import android.view.WindowManager

/**
 * 截屏服务: MediaProjection + VirtualDisplay + ImageReader。
 * 只保留最后一帧, 上层调用 capture() 取当前屏幕位图。
 *
 * 注意(Android 14+): 必须在 AndroidManifest 声明
 *   <uses-permission android:name="android.permission.FOREGROUND_SERVICE_MEDIA_PROJECTION"/>
 * 且服务类型为 mediaProjection, 否则 startForeground 抛 SecurityException。
 */
class CaptureService : Service() {

    companion object {
        const val CHANNEL_ID = "jump_capture"
        const val EXTRA_CODE = "code"
        const val EXTRA_DATA = "data"
        @Volatile
        var instance: CaptureService? = null
            private set
    }

    private var projection: MediaProjection? = null
    private var display: VirtualDisplay? = null
    private var reader: ImageReader? = null
    private var width = 0
    private var height = 0
    private var density = 0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        instance = this
        val wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        wm.defaultDisplay.getRealMetrics(metrics)
        width = metrics.widthPixels
        height = metrics.heightPixels
        density = metrics.densityDpi
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && nm.getNotificationChannel(CHANNEL_ID) == null) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "跳一跳助手", NotificationManager.IMPORTANCE_LOW)
            )
        }
        val notification: Notification = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("跳一跳助手")
            .setContentText("截屏识别运行中")
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .build()
        startForeground(1, notification)

        val code = intent?.getIntExtra(EXTRA_CODE, Activity.RESULT_CANCELED) ?: Activity.RESULT_CANCELED
        @Suppress("DEPRECATION")
        val data = intent?.getParcelableExtra<Intent>(EXTRA_DATA)
        if (code == Activity.RESULT_OK && data != null && projection == null) {
            val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            projection = mpm.getMediaProjection(code, data)
            startVirtualDisplay()
        }
        return START_STICKY
    }

    private fun startVirtualDisplay() {
        reader?.close()
        reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2)
        // Android 14+ 要求先注册回调, 否则 createVirtualDisplay 会抛异常
        projection?.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                projection = null
            }
        }, Handler(Looper.getMainLooper()))
        display = projection?.createVirtualDisplay(
            "jump_capture", width, height, density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, reader?.surface, null, null
        )
    }

    /** 取当前屏幕位图(失败返回 null)。 */
    fun capture(): Bitmap? {
        val r = reader ?: return null
        val image: Image = r.acquireLatestImage() ?: return null
        return try {
            val plane = image.planes[0]
            val rowStride = plane.rowStride
            val pixelStride = plane.pixelStride
            val rowPadding = rowStride - pixelStride * width
            val bmp = Bitmap.createBitmap(
                width + rowPadding / pixelStride, height, Bitmap.Config.ARGB_8888
            )
            bmp.copyPixelsFromBuffer(plane.buffer)
            val cropped = Bitmap.createBitmap(bmp, 0, 0, width, height)
            if (cropped !== bmp) bmp.recycle()
            cropped
        } catch (e: Exception) {
            null
        } finally {
            image.close()
        }
    }

    override fun onDestroy() {
        instance = null
        display?.release()
        reader?.close()
        projection?.stop()
        super.onDestroy()
    }
}
