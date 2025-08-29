# Android Related Commands

For building a project command line and starting Android Instrumentation Tests.

## Build

```bash
./gradlew :app:assembleAndroidTest
./gradlew :app:assembleDebug
```

## Install

```bash
adb install -r app/build/outputs/apk/stable/debug/app-debug.apk
adb install -r app/build/outputs/apk/androidTest/stable/debug/app-debug-androidTest.apk 
```

## Check Connected Devices

```bash
adb devices
```

## Validate

```bash
adb shell pm list packages | grep com.example.app
```

## Start Test

```bash
adb shell am instrument -w com.exanmple.app.debug.test/androidx.test.runner.AndroidJUnitRunner
```

## Restart ADB

```bash
adb kill-server
adb start-server
```
