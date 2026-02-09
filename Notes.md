
## Building to Android:

```powershell
$env:ANDROID_HOME = "C:\Users\redst\AppData\Local\Android\Sdk"; $env:ANDROID_NDK_ROOT = "C:\Users\redst\AppData\Local\Android\Sdk\ndk\29.0.14206865"
```

```powershell
cargo apk build -p feature_complete_android --target aarch64-linux-android
```