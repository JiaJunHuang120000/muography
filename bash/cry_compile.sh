#!/bin/bash
# Build CRY and compile the local C++ tools used by this repository.
#
# Outputs:
# - cry_v1.7/ (CRY source/build)
# - remote executable at repo root (CRY text -> HepMC3 converter)
#
# Run once during first-time setup.
wget https://nuclear.llnl.gov/simulation/cry_v1.7.tar.gz
tar -xvzf cry_v1.7.tar.gz
rm cry_v1.7.tar.gz
cd cry_v1.7/
make

cd test/
cp ../../cpp/testMain.cc .
g++ -I. -I../src -o testMain testMain.cc -L../lib -lCRY

cd ../..

export LD_LIBRARY_PATH=$PWD/hepmc3-install/lib:$LD_LIBRARY_PATH
HEPMC3_INSTALL=$PWD/hepmc3-install
g++ -o remote cpp/remote.cxx -I$HEPMC3_INSTALL/include -L$HEPMC3_INSTALL/lib -lHepMC3     `root-config --cflags --libs` -lEG -lGeom
