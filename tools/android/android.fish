#!/bin/fish

# steps:
# 0. rename config_android.toml to config.toml
# 1. source tools/android/android.fish
# 2. build the project with `cargo tauri android dev`
# 3. run change_arm.sh
# 4. build again

set me $(pwd)

export ANDROID_HOME={$me}/local/Android/Sdk

export PATH="$ANDROID_HOME/emulator:$ANDROID_HOME/cmdline-tools:$ANDROID_HOME/cmdline-tools/bin:$ANDROID_HOME/platform-tools:$PATH"

export NDK_PATH=$ANDROID_HOME/ndk/28.0.13004108

export NDK_HOME=$NDK_PATH

export ANDROID_NDK=$NDK_PATH

export ANDROID_NDK_HOME=$NDK_PATH
