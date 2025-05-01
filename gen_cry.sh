cd cry_v1.7/test/
./testMain setup.file 10000 > output.txt
./cry_muon_converter output.txt output.hepmc
rm output.txt
cd ../..
mkdir hepmc
mv cry_v1.7/test/output.hepmc hepmc/
 
