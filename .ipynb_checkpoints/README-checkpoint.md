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
ddsim --help
```

For each time you start a new environment, please do ```source setup_env.sh``` in order for ddsim and python scripts to work properly.

For running the scripts for multi-detectors, you will need to first change all of the parameters inside ```bash/config.sh``` to the desired options. Afterward, make a ```detector.xml``` file for the ```script/generate_soil_target.py``` to generate both the target world and the free-sky world for ```merge.xml``` to call and put it inside the ```xml/``` folder. Once finished, follow the steps below in ```~/muography``` directory
```
source setup_env.sh 
soruce bash/config.sh

cd $output_file_path
python3 $DETECTOR_PATH/bash/config.sh/generate_soil_target.py $DETECTOR_PATH/soil_target.template.xml
pyhton3 $DETECTOR_PATH/bash/config.sh/dd4hep_viz.py $detector_file_path/soil_target.xml
cd $DETECTOR_PATH

bash bash/multi.sh
bash bash/iteration.sh

python3 script/root_to_pkl_and_splitting.py

```


### Parameters for CRY

1. ~~To change the area of muons hit at sea level, change the [subboxLength] in [setup.file] (units in m). This will create a 2d box of length^2.~~ This step has been updated to change the xy of the hepmc muon generation rather than CRY level generation for a more adoptive cahnge needed. Thus you will only need one large CRY output .txt file 
3. To change the other parameters of the muon generation, you will need to either change the parameters inside [bash/gen_cry.sh] or in the terminal where you manualy enter the command in the above section.
4. For the number of events taken from raw CRY output, change line 2 and the number [100] to desired number of events you need. You will need higher events than needed if you are applying a energy cut later, it is suggested you generate a large pool of events as a base events such you can grab later on.
5. In line 6 is the rest of the parameters you can change in the process of CRY to hepmc, following the variables <input_file.txt> <output_file.hepmc> <muon_generation_height> <detector_position_x> <detector_position_y> <detector_position_z> <z_offset> <E_cut> <number_of_events>.
   - <input_file.txt> = raw CRY file for event filtering and processing
   - <output_file.hepmc> = output hepmc path for ddsim events
   - <muon_generation_height> = height of the generate muons (m)
   - <detector_position_xyz> = xyz position of the center of the detector, where the muons will always cross (m)
   - <z_offset> = the z/2 length of the detector such that muons will be generate as a volumn that encovers the detector rather than a plane, this is a normal random distribution offset of the plane applied manualy in the code (m)
   - <y_offset> = the y/2 length of the detector such that muons will be generate as a volumn that encovers the detector rather than a plane, this is a normal random distribution offset of the plane applied manualy in the code (m)
   - <x_offset> = the x/2 length of the detector such that muons will be generate as a volumn that encovers the detector rather than a plane, this is a normal random distribution offset of the plane applied manualy in the code (m)
   - <E_cut> = the energy filtering of the raw CRY events, note it does not modify for energy but just filter for events from raw CRY output that passes (GeV)
   - <number_of_events> = number of events you want inside the output hepmc file, the number is based on the filtering so it may be less than expected.
6. If you would like realistic comparison of the number of events at different height, you need to use the same <number_of_events> with different <E_cut> (height/2) for comparison.
