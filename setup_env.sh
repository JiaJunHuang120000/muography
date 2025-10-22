#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-ubuntu2204-gcc11-opt/setup.sh
source $PWD/DD4hep/bin/thisdd4hep.sh
# --- Detector XML path ---
export DETECTOR_PATH=./
export DETECTOR_CONFIG=${MUOGRAPHY}/share/muography
#--- Plugin library path ---
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH

# --- DD4hep binaries and other dependencies ---
export PATH=$PWD/DD4hep/bin:$PATH
export LD_LIBRARY_PATH=$PWD/DD4hep/lib:$LD_LIBRARY_PATH
