#!/bin/bash

# first make clean the llama.cpp

# this weird hack is for llama.cpp to build in Armv8 Systems.

me=$(pwd)

target_dir=$me/src-tauri/target/aarch64-linux-android

find $target_dir -type f -exec sed -i -e 's/armv7-none-linux-androideabi28/aarch64-linux-android28/g' {} \;
# armv7-a -> armv8-a

