wget https://nuclear.llnl.gov/simulation/cry_v1.7.tar.gz
tar -xvzf cry_v1.7.tar.gz
rm cry_v1.7.tar.gz
cd cry_v1.7/
make
cd ..

export LD_LIBRARY_PATH=$HOME/tmp/muography/hepmc3-install/lib:$LD_LIBRARY_PATH
HEPMC3_INSTALL=$HOME/tmp/muography/hepmc3-install
g++ -I. -I../src -o back_project_single cpp/back_project_single.cxx  -L../lib -lCRY  -I$HEPMC3_INSTALL/include -L$HEPMC3_INSTALL/lib -lHepMC3 `root-config --cflags --libs` -lEG -lGeom