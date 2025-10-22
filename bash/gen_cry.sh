cd cry_v1.7/test/
./testMain ../../setup.file 100 > output.txt

cd ../..
mkdir -p hepmc
./remote cry_v1.7/test/output.txt hepmc/output.hepmc 10 0 0 0 0.1 0 100
#rm cry_v1.7/test/output.txt

