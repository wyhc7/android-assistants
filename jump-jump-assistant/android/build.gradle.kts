plugins {
    id("com.android.application") version "8.7.3"
    id("org.jetbrains.kotlin.android") version "1.9.25"
}

android {
    namespace = "com.jump.assist"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.jump.assist"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        // 见下方 splits: 按架构拆包, 每个设备只装自己那一份(两条不能同时设, 否则 Gradle 直接报冲突)
    }

    /**
     * 按 ABI 拆包。ONNX Runtime 的原生库每个架构 17~21MB, 打成一个通用包就是白白多带一份。
     * 真机(arm64-v8a)装 arm64 包, MuMu 模拟器(x86_64)装 x86_64 包。
     */
    splits {
        abi {
            isEnable = true
            reset()
            include("arm64-v8a", "x86_64")
            isUniversalApk = false
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    // 模型放 assets, 不压缩以便直接读字节建 ONNX Runtime 会话
    androidResources {
        noCompress += "onnx"
    }

    sourceSets["main"].apply {
        manifest.srcFile("src/main/AndroidManifest.xml")
        java.srcDirs("src/main/java")
        res.srcDirs("src/main/res")
        assets.srcDirs("src/main/assets")
    }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
}
