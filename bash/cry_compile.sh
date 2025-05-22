wget https://nuclear.llnl.gov/simulation/cry_v1.7.tar.gz
tar -xvzf cry_v1.7.tar.gz
rm cry_v1.7.tar.gz
cd cry_v1.7/
make
cd ..
export LD_LIBRARY_PATH=$HOME/hepmc3-install/lib:$LD_LIBRARY_PATH
HEPMC3_INSTALL=$HOME/muography/hepmc3-install
g++ -o cry_muon_converter cry_muon_converter.cxx     -I$HEPMC3_INSTALL/include -L$HEPMC3_INSTALL/lib -lHepMC3     `root-config --cflags --libs` -lEG -lGeom
