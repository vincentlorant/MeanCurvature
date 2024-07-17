#!/bin/bash

# Directory containing the compile_commands.json
BUILD_DIR=../build

# Source directory to check
SRC_DIR=../impl/mean_curvature_02/main.cc

# Specify clang-tidy checks (you can modify this to suit your needs)
CHECKS='modernize-*'

# Ensure the build directory exists and contains compile_commands.json
if [ ! -f "$BUILD_DIR/compile_commands.json" ]; then
    echo "Error: compile_commands.json not found in $BUILD_DIR"
    exit 1
fi

# Run clang-tidy on all source files in the SRC_DIR
find $SRC_DIR -name '*.cc' | while read file; do
    echo "Running clang-tidy on $file"
    clang-tidy -p=$BUILD_DIR $file -checks=$CHECKS
done
