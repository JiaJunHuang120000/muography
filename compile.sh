cd cry_v1.7/test
export LD_LIBRARY_PATH=$HOME/hepmc3-install/lib:$LD_LIBRARY_PATH
HEPMC3_INSTALL=/home/jiajun/hepmc3-install
g++ -o cry_muon_converter cry_muon_converter.cxx     -I$HEPMC3_INSTALL/include -L$HEPMC3_INSTALL/lib -lHepMC3     `root-config --cflags --libs` -lEG -lGeom
