#!/bin/bash
# Build and install this repository's DD4hep plugin library.
#
# Usage:
#   source bash/setup_env.sh
#   bash bash/build.sh
#
# What it does:
# 1) Removes any previous local build directory.
# 2) Configures CMake with install prefix set to $MUOGRAPHY.
# 3) Compiles and installs plugin/library files.

rm -rf build
mkdir build
cd build || exit 1
cmake .. -DCMAKE_INSTALL_PREFIX="$MUOGRAPHY"
make install -j24
