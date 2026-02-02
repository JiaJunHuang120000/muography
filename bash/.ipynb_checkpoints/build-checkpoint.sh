rm -rf build
mkdir build; cd build; cmake .. -DCMAKE_INSTALL_PREFIX=$MUOGRAPHY; make install -j24;
