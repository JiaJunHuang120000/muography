#!/bin/bash
# Clone and build DD4hep locally for this repository.
#
# This script assumes CERN LCG environment from CVMFS is available.
# It creates a local DD4hep installation under ./DD4hep.
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-ubuntu2204-gcc11-opt/setup.sh

git clone https://github.com/AIDASoft/DD4hep.git 

cd DD4hep

mkdir build && cd build

cmake -DDD4HEP_USE_GEANT4=ON -DBoost_NO_BOOST_CMAKE=ON -DDD4HEP_USE_HEPMC3=ON -DBUILD_TESTING=ON -DROOT_DIR=$ROOTSYS -DDD4HEP_USE_EDM4HEP=ON -D CMAKE_BUILD_TYPE=Release ..

make install -j32
cd ../..
