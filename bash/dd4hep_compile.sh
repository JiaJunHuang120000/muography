source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-ubuntu2204-gcc11-opt/setup.sh

git clone https://github.com/AIDASoft/DD4hep.git

mkdir build && cd build

cmake -DDD4HEP_USE_GEANT4=ON -DBoost_NO_BOOST_CMAKE=ON -DDD4HEP_USE_HEPMC=ON -DBUILD_TESTING=ON -DROOT_DIR=$ROOTSYS -D CMAKE_BUILD_TYPE=Release ..

make install -j32
cd ..

