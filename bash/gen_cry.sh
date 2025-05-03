cd cry_v1.7/test/
./testMain ../../setup.file 9999 > output.txt

cd ../..
mkdir -p hepmc
./cry_muon_converter cry_v1.7/test/output.txt hepmc/output.hepmc
rm cry_v1.7/test/output.txt

