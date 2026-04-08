#!/bin/bash

export number_of_detector=3
export detector_name='planar_underground_v4'
export output_file_path="${DETECTOR_PATH}/data/${detector_name}"
export hepmc_file_path="${DETECTOR_PATH}/hepmc/${detector_name}"
export detector_file_path="${DETECTOR_PATH}/detectors/${detector_name}"

# Create directories
mkdir -p "${output_file_path}"
mkdir -p "${hepmc_file_path}"
mkdir -p "${detector_file_path}"

# CRY Variables
export CRY_num_of_events=(300000000 300000000 300000000)
export generation_height=10 #(m)
export detector_z_offset=0.20 #(m)
export detector_y_offset=0.0 #(m)
export detector_x_offset=0.0 #(m)
export energy_cutoff=(50 50 50) #(GeV)
export input_cry_file="/home/jiajun/github/muography/hepmc/30x30_cm_300M_events.txt"  
#"${DETECTOR_PATH}/hepmc/20x20cm2_1M_events.txt"

# DDsim Variables
export world_configurations=('free' 'target')
export detector_xml_name="planar_detector_rotated_z_90"
export input_detector_file="${DETECTOR_PATH}/detectors/${detector_name}/merge.xml"

export detector_pos_x=(-50 0 50) #(m)
export detector_pos_y=(0 0 0) #(m)
export detector_pos_z=(-100 -100 -100) #(m)
export number_of_events=(300000000 300000000 300000000) # Events simulated in ddsim

export detector_pos_x_env="${detector_pos_x[@]}" # Export to string for Python
export detector_pos_y_env="${detector_pos_y[@]}"
export detector_pos_z_env="${detector_pos_z[@]}" 

# =========================
# Target geometry definition
# =========================

export world_area=200 #(m) Do not make less than 2000 meters for a generation height of 20 meters, as this is the generation width of muons at large angles
export world_depth=150 #(m) dz of the world, should made such is larger than generation_height
export world_top_material="Air" # Surface material
export world_bottom_material="Rock" # Underground material
export pixel_size=8 #(m) Size for each voxel generation width 

TARGETS=(
#"cube   xdim=0.3*m ydim=0.3*m zdim=0.38*m x=0*m  y=0.15*m  z=0.835*m    material=Steel235"
"cube   xdim=10*m ydim=500*m zdim=10*m x=0*m  y=0*m  z=-50*m  rx=0 ry=0 rz=0.785  material=Steel235"
"sphere r=10*m   x=50*m   y=0*m    z=-50*m    material=Steel235"
  # "sphere r=20*m   x=50*m   y=0*m    z=-20*m    material=Vacuum"
  # "sphere r=5*m    x=0*m    y=20*m   z=-5*m    material=Vacuum"
  # "sphere r=5*m    x=5*m    y=0*m    z=-5*m    material=Vacuum"
  # "sphere r=5*m    x=-10*m  y=0*m    z=-35*m    material=Vacuum"
  # "sphere r=10*m   x=0*m    y=40*m   z=-65*m    material=Vacuum"
  # "sphere r=10*m   x=0*m    y=-60*m  z=-20*m    material=Vacuum"
  # "cube   xdim=10*m ydim=10*m zdim=10*m x=12*m  y=7*m  z=-25*m    material=LeadOxide"
  # "cube   xdim=5*m  ydim=5*m  zdim=5*m  x=-30*m y=-30*m z=-10*m    material=LeadOxide"
  # "sphere r=1*m   x=10*m    y=-10*m  z=-10*m    material=Vacuum"
) # Make sure to use the same format above, current generation structure; sphere, cube
