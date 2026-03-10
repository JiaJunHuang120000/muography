### Muography pipeline: CRY → HepMC3 → DD4hep/DDG4 → ROOT/PKL

This repository simulates cosmic muons through configurable detector geometries and target volumes.
It is organized as a reproducible pipeline:

1. Generate raw cosmic-ray events with **CRY** (`cpp/testMain.cc`).
2. Convert/filter those events into **HepMC3** (`cpp/remote.cxx`).
3. Simulate detector response with **DD4hep + DDG4/ddsim**.
4. Post-process EDM4hep ROOT files into merged/split datasets (`python/root_to_pkl_and_splitting.py`).

---

## 1) What each folder is for

- `bash/`: environment setup, dependency builds, and pipeline run scripts.
- `cpp/`: C++ executables for CRY text generation and CRY→HepMC conversion.
- `src/`: DD4hep geometry plugins compiled as a detector plugin library.
- `xml/`: detector/world XML definitions and templates.
- `python/`: geometry generation, ddsim steering, post-processing, and visualization helpers.
- `data/`, `plots/`: generated datasets/analysis artifacts.
- `compact/`: DD4hep material/color definitions.

---

## 2) Prerequisites

Expected baseline:

- Ubuntu 22.04.5 LTS
- CRY 1.7
- HepMC3 3.2.6
- DD4hep v01-32-01
- ROOT 6.32.02
- CVMFS access for CERN LCG view (default workflow uses this)

---

## 3) First-time setup (full install from this repo)

Clone:

```bash
git clone https://github.com/JiaJunHuang120000/muography.git
cd muography
```

Build dependency/toolchain pieces in order:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-ubuntu2204-gcc11-opt/setup.sh
bash bash/hepmc_compile.sh
bash bash/cry_compile.sh
bash bash/gen_cry.sh
bash bash/dd4hep_compile.sh
```

Build and install the local DD4hep detector plugin from `src/*.cpp`:

```bash
source bash/setup_env.sh
bash bash/build.sh
ddsim --help
```

> For every new shell, run `source bash/setup_env.sh` before using `ddsim` or project scripts.

---

## 4) Minimal quick-start run (single pass)

```bash
source bash/setup_env.sh
source bash/config.sh

# Generate detector-specific world XML files in detectors/${detector_name}/
bash bash/xml_compile.sh

# Convert CRY text to per-detector HepMC files
bash bash/multi.sh

# Run ddsim for both world configs (free + target)
bash bash/iteration.sh

# Build merged PKL outputs and split ROOT chunks
python3 python/root_to_pkl_and_splitting.py
```

Expected main outputs:

- HepMC: `hepmc/${detector_name}/${detector_name}_<i>.hepmc`
- DDsim ROOT: `data/${detector_name}/${detector_name}_{free|target}_<i>.edm4hep.root`
- Merged pickle: `data/${detector_name}/{free_merge.pkl,target_merge.pkl}`

---

## 5) How configuration works (most important file)

`bash/config.sh` controls almost the entire workflow.

### Detector scan controls

- `number_of_detector`
- `detector_pos_x`, `detector_pos_y`, `detector_pos_z` (arrays, meters)
- `number_of_events` (ddsim events per detector)

Array lengths must be consistent with `number_of_detector`.

### CRY/HepMC conversion controls

- `CRY_num_of_events` (max accepted events target for each detector index)
- `generation_height` (m)
- `detector_z_offset`, `detector_y_offset`, `detector_x_offset` (m)
- `energy_cutoff` (GeV)
- `input_cry_file`

### World and target controls

- `world_area`, `world_depth`
- `world_top_material`, `world_bottom_material`
- `TARGETS=(...)` entries using compact shape syntax, e.g.
  - `sphere r=5*m x=0*m y=0*m z=-10*m material=LeadOxide`
  - `cube xdim=10*m ydim=10*m zdim=10*m x=0*m y=0*m z=-20*m material=Steel235`

`python/generate_soil_target.py` parses `TARGETS` and emits:

- `soil_free.xml` (no inserted solid targets)
- `soil_target.xml` (with generated target solids)

---

## 6) Script-by-script guide (what to run and when)

### Bash scripts

- `bash/setup_env.sh`: exports `DETECTOR_PATH`, plugin/library paths, DD4hep path.
- `bash/hepmc_compile.sh`: downloads + builds HepMC3 in `hepmc3-install/`.
- `bash/cry_compile.sh`: downloads/builds CRY and builds `remote` converter executable.
- `bash/gen_cry.sh`: quick smoke test generation (`testMain` + one `remote` invocation).
- `bash/dd4hep_compile.sh`: clones/builds DD4hep.
- `bash/build.sh`: builds this repo’s DD4hep plugin and installs into `$MUOGRAPHY`.
- `bash/xml_compile.sh`: copies XML templates and runs `generate_soil_target.py`.
- `bash/multi.sh`: loops detectors and runs `./remote` to create HepMC files.
- `bash/iteration.sh`: loops `{free,target}` × detectors and runs `ddsim`.

### C++ scripts

- `cpp/testMain.cc`: CRY event text producer (prints one line per secondary particle).
- `cpp/remote.cxx`: filters muons, maps geometry offsets, writes HepMC3 events.

### Python scripts

- `python/generate_soil_target.py`: template expander for world/target XML.
- `python/steering.py`: DD4hep simulation configuration object used by `ddsim`.
- `python/root_to_pkl_and_splitting.py`: ROOT reading, reconstruction, smearing, merge/split output.
- `python/dd4hep_viz.py`: parser + visualizer + voxelizer helper for geometry debugging.

---

## 7) Common learning path for newcomers

1. Read `bash/config.sh` and change only one knob at a time.
2. Run `bash/xml_compile.sh` and inspect generated XML in `detectors/${detector_name}/`.
3. Run `bash/multi.sh` with small event counts first.
4. Run `bash/iteration.sh` with low `number_of_events` and verify ROOT output exists.
5. Run post-processing and inspect `free_merge.pkl` and `target_merge.pkl`.
6. Only then scale to larger statistics.

---

## 8) Practical notes / pitfalls

- Ensure your shell has environment from `bash/setup_env.sh`.
- Keep `number_of_detector` consistent with array sizes in `config.sh`.
- Keep units straight:
  - config detector/world position inputs are generally meters,
  - DD4hep often uses mm internally,
  - `remote.cxx` converts CRY MeV kinetic energy to GeV total energy for HepMC.
- Start with low event counts; full runs can be expensive.

---

## 9) Legacy note

Older quick instructions are still valid at a high level, but this README is the recommended onboarding path.

---


## 10) Suggested short file aliases (<=5 chars)

These are **suggested names** to make intent obvious at a glance.
They are not applied automatically in this repository.

### bash/
- `setup_env.sh` → `envup.sh`
- `config.sh` → `cfg.sh`
- `hepmc_compile.sh` → `hmcmp.sh`
- `cry_compile.sh` → `crycp.sh`
- `dd4hep_compile.sh` → `ddcmp.sh`
- `build.sh` → `bld.sh`
- `gen_cry.sh` → `crygn.sh`
- `xml_compile.sh` → `xmlgn.sh`
- `multi.sh` → `hloop.sh`
- `iteration.sh` → `dloop.sh`

### cpp/
- `testMain.cc` → `crytx.cc`
- `remote.cxx` → `tohmc.cxx`
- `setup.file` → `crycf.file`

### python/
- `generate_soil_target.py` → `mkxml.py`
- `steering.py` → `ddcfg.py`
- `root_to_pkl_and_splitting.py` → `mkpkl.py`
- `dd4hep_viz.py` → `ddviz.py`
