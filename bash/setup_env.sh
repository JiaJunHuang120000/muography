#!/bin/bash
# Set shell environment for this repository.
#
# Exports paths needed by:
# - ddsim / DD4hep binaries,
# - local plugin library (built from src/*.cpp),
# - detector configuration paths used by scripts and XML includes.
#
# Must be sourced in each new shell session:
#   source bash/setup_env.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-ubuntu2204-gcc11-opt/setup.sh
source $PWD/DD4hep/bin/thisdd4hep.sh
export MUOGRAPHY=$HOME/.local/muography
# --- Detector XML path ---
export DETECTOR_PATH=$PWD
export DETECTOR_CONFIG=${MUOGRAPHY}/share/muography
#--- Plugin library path ---
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH

# --- DD4hep binaries and other dependencies ---
export PATH=$PWD/DD4hep/bin:$PATH
export LD_LIBRARY_PATH=$PWD/DD4hep/lib:$LD_LIBRARY_PATH