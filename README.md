### Pipe line for generating CRY simulated muons for detectors

## Compiling this github
Run the code to clone the github for compiling
``` 
git clone https://github.com/JiaJunHuang120000/muography.git
```

Run the following scripts after cloning the {muography} github repository

```
bash compile.sh
bash gen_cry.sh
curl -L https://github.com/eic/eic-shell/raw/main/install.sh | bash
./eic-shell
source setup_env.sh
bash build.sh
ddsim --compactFile metal.xml --outputFile data/test.root --inputFile hepmc/output.hepmc --numberOfEvents 5

wget https://nuclear.llnl.gov/simulation/cry_v1.7.tar.gz
tar -xvzf cry_v1.7.tar.gz
rm cry_v1.7.tar.gz
cd cry_v1.7/
make
```


## Standalone instruction on how to run CRY library for generating cosmic muon and filtering for detector level muons:

1. Install the CRY package from website https://nuclear.llnl.gov/simulation/ if current version is not working correctly.

2. Run "make" in the top repository to compile for testMain excutable in /test/ folder.

3. Change the settings in "setup.file" to desire and run the command "./testMain setup.file 10 > out.txt" for the total number of events of muons and ouput into file out.txt.

4. Compile the "cry_muon_converter" in the /test/ folder, the excutable to convert CRY output to .hepmc format

'''source ../setup.sh'''

```source /home/ucr/root_install/bin/thisroot.sh```

```export LD_LIBRARY_PATH=$HOME/hepmc3-install/lib:$LD_LIBRARY_PATH```

```HEPMC3_INSTALL=/home/jiajun/hepmc3-install```

```g++ -I. -I../src -o cry_muon_converter cry_muon_converter.cxx  -L../lib -lCRY  -I$HEPMC3_INSTALL/include -L$HEPMC3_INSTALL/lib -lHepMC3 `root-config --cflags --libs` -lEG -lGeom```

5. Run the command "./cry_muon_converter out.txt out.hepmc" to obtain the file.

6. If you would like the Y/Z coordinate exchanged, use the zdc converter with the changes on the compiling code accordingly.

7. Everytime you change the parameter in the converter, you will have to rerun the compiling code for changes to be updated.

8. The converter changes the plane where the muons will be generated and set a detection plane with user defined area for muon detection.
