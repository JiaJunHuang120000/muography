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
