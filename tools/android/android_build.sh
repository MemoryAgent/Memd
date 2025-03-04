#!/bin/bash

# https://github.com/rust-lang/rust-bindgen/issues/1229

me=$(pwd)

ANDROID_HOME={$me}/local/Android/Sdk

PATH="$ANDROID_HOME/emulator:$ANDROID_HOME/cmdline-tools:$ANDROID_HOME/cmdline-tools/bin:$ANDROID_HOME/platform-tools:$PATH"

NDK_PATH=$ANDROID_HOME/ndk/28.0.13004108

NDK_HOME=$NDK_PATH

ANDROID_NDK=$NDK_PATH

ANDROID_NDK_HOME=$NDK_PATH

