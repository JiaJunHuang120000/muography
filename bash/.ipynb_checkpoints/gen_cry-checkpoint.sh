cd cry_v1.7/test/
./testMain ../../setup.file 100 > output.txt

cd ../..
mkdir -p hepmc
./back_project_single cry_v1.7/test/output.txt hepmc/output.hepmc 100 0 0 0 0.1 10
rm cry_v1.7/test/output.txt

