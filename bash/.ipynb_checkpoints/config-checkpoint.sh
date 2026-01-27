#!/bin/bash

export number_of_detector=3
export detector_name='planar_detector'
export output_file_path="${DETECTOR_PATH}/data/${detector_name}"
export hepmc_file_path="${DETECTOR_PATH}/hepmc/${detector_name}"

# Create directories
mkdir -p "${output_file_path}"
mkdir -p "${hepmc_file_path}"

# CRY Variables
export CRY_num_of_events=(100000 100000 100000)
export generation_height=20 #(m)
export detector_z_offset=0.3 #(m)
export detector_y_offset=0.3 #(m)
export detector_x_offset=0.3 #(m)
export energy_cutoff=(0 0 0) #(GeV)
export input_cry_file="${DETECTOR_PATH}/hepmc/100M_events.txt"

# DDsim Variables
export world_configurations=('free' 'target')
export detector_xml_name="square_flat_target"
export input_detector_file="${DETECTOR_PATH}/detectors/${detector_name}/merge.xml"

export detector_pos_x=(0 -5 5) #(m)
export detector_pos_y=(0 -5 5) #(m)
export detector_pos_z=(0.5 0.5 0.5) #(m)
export number_of_events=(100000 100000 100000) # Events simulated in ddsim

# =========================
# Target geometry definition
# =========================

export world_area=2000 #(m) Do not make less than 2000 meters for a generation height of 20 meters, as this is the generation width of muons at large angles
export world_depth=200 #(m) dz of the world, should made such is larger than generation_height
export world_top_material="Air" # Surface material
export world_bottom_material="Rock" # Underground material
export pixel_size=3 #(m) Size for each voxel generation width 

TARGETS=(
#"sphere r=3*m   x=0*m   y=5*m    z=5*m    material=LeadOxide"
"cube   xdim=10*m ydim=10*m zdim=10*m x=0*m  y=0*m  z=10*m    material=Steel235"
  # "sphere r=20*m   x=50*m   y=0*m    z=-20*m    material=Vacuum"
  # "sphere r=5*m    x=0*m    y=20*m   z=-5*m    material=Vacuum"
  # "sphere r=5*m    x=5*m    y=0*m    z=-5*m    material=Vacuum"
  # "sphere r=5*m    x=-10*m  y=0*m    z=-35*m    material=Vacuum"
  # "sphere r=10*m   x=0*m    y=40*m   z=-65*m    material=Vacuum"
  # "sphere r=10*m   x=0*m    y=-60*m  z=-20*m    material=Vacuum"
  # "cube   xdim=10*m ydim=10*m zdim=10*m x=12*m  y=7*m  z=-25*m    material=LeadOxide"
  # "cube   xdim=5*m  ydim=5*m  zdim=5*m  x=-30*m y=-30*m z=-10*m    material=LeadOxide"
) # Make sure to use the same format above, current generation structure; sphere, cube
