### Pipe line for generating CRY simulated muons for detectors and DD4HEP Simulation

## Pre-Requisite
- Ubuntu 22.04.5 LTS
- CRY version 1.7
- HepMC3-3.2.6
- DD4HEP v01-32-01
- ROOT Version: 6.32.02

## Standalone Proceed for DD4HEP Installation
The current guide of DD4HEP is install using cvmfs, following the guidline from https://docs.google.com/document/d/1XPHIn0BTb-bTih-vyRavrLSQryHR4qhdpqHkqsGROx4/edit?tab=t.0 . This guide also contains instruction on installing DD4HEP locally.

## Compiling this github
Run the code to clone the github for compiling
``` 
git clone https://github.com/JiaJunHuang120000/muography.git
cd muography/
```

Run the following scripts after cloning the {muography} github repository

```
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-ubuntu2204-gcc11-opt/setup.sh
bash bash/hepmc_compile.sh
bash bash/cry_compile.sh
bash bash/gen_cry.sh
bash bash/dd4hep_compile.sh

source setup_env.sh
bash build.sh
mkdir data
ddsim --compactFile xml/square_2x4x8.xml       --outputFile data/test.root       --inputFiles hepmc/output.hepmc       --numberOfEvents 5       --steeringFile steering.py
```

### Parameters for CRY

1. To change the area of muons hit at sea level, change the [subboxLength] in [setup.file] (units in m). This will create a 2d box of length^2.
3. To change the other parameters of the muon generation, you will need to either change the parameters inside [bash/gen_cry.sh] or in the terminal where you manualy enter the command in the above section.
4. For the number of events taken from raw CRY output, change line 2 and the number [100] to desired number of events you need. You will need higher events than needed if you are applying a energy cut later, it is suggested you generate a large pool of events as a base events such you can grab later on.
5. In line 6 is the rest of the parameters you can change in the process of CRY to hepmc, following the variables <input_file.txt> <output_file.hepmc> <muon_generation_height> <detector_position_x> <detector_position_y> <detector_position_z> <z_offset> <E_cut> <number_of_events>.
   - <input_file.txt> = raw CRY file for event filtering and processing
   - <output_file.hepmc> = output hepmc path for ddsim events
   - <muon_generation_height> = height of the generate muons (m)
   - <detector_position_xyz> = xyz position of the center of the detector, where the muons will always cross (m)
   - <z_offset> = the z/2 length of the detector such that muons will be generate as a volumn that encovers the detector rather than a plane, this is a normal random distribution offset of the plane applied manualy in the code (m)
   - <E_cut> = the energy filtering of the raw CRY events, note it does not modify for energy but just filter for events from raw CRY output that passes (GeV)
   - <number_of_events> = number of events you want inside the output hepmc file, the number is based on the filtering so it may be less than expected.
6. If you would like realistic comparison of the number of events at different height, you need to use the same <number_of_events> with different <E_cut> (height/2) for comparison.


## (Not needed but good to have) Standalone instruction on how to run CRY library for generating cosmic muon and filtering for detector level muons:

1. Install the CRY package from website https://nuclear.llnl.gov/simulation/ if current version is not working correctly.

2. Run "make" in the top repository to compile for testMain excutable in /test/ folder.

3. Change the settings in "setup.file" to desire and run the command "./testMain $HOME/muography/setup.file 10 > out.txt" for the total number of events of muons and ouput into file out.txt.

4. Compile the "remote.cxx" in the /cpp/ folder, the excutable to convert CRY output to .hepmc format

```source $HOME/root_install/bin/thisroot.sh```

```export LD_LIBRARY_PATH=$HOME/muography/hepmc3-install/lib:$LD_LIBRARY_PATH```

```HEPMC3_INSTALL=$HOME/muography/hepmc3-install```

```g++ -I. -I../src -o remote cpp/remote.cxx  -L../lib -lCRY  -I$HEPMC3_INSTALL/include -L$HEPMC3_INSTALL/lib -lHepMC3 `root-config --cflags --libs` -lEG -lGeom```

5. Run the command "./remote <input_file.txt> <output_file.hepmc> <muon_generation_height> <detector_position_x> <detector_position_y> <detector_position_z> <z_offset> <E_cut> <number_of_events>" to obtain the file.

6. Everytime you change the parameter in the converter, you will have to rerun the compiling code for changes to be updated.

7. The converter changes the plane where the muons will be generated and set a detection plane with user defined area for muon detection.