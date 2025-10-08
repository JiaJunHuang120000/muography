#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-ubuntu2204-gcc11-opt/setup.sh
source /home/jiajun/DD4hep/bin/thisdd4hep.sh
# --- Detector XML path ---
export DETECTOR_PATH=~/muography_backup

# --- Plugin library path ---
export LD_LIBRARY_PATH=~/muography_backup/build:$LD_LIBRARY_PATH

# --- DD4hep binaries and other dependencies ---
export PATH=~/DD4hep/bin:$PATH
export LD_LIBRARY_PATH=~/DD4hep/lib:$LD_LIBRARY_PATH

# Optional: source ROOT if not already sourced
#source /cvmfs/sft.cern.ch/lcg/releases/ROOT/6.28.04-*/bin/thisroot.sh

