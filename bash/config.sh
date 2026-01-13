#!/bin/bash

export number_of_detector=10
export detector_name='small_pixel_64'
export output_file_path="${DETECTOR_PATH}/data/${detector_name}"
export hepmc_file_path="${DETECTOR_PATH}/hepmc/${detector_name}"

# Create directories
mkdir -p "${output_file_path}"
mkdir -p "${hepmc_file_path}"

# CRY Variables
export CRY_num_of_events=(1000000 5000000 5000000 5000000 5000000 5000000 5000000 5000000 5000000 5000000)
export generation_height=10
export detector_total_length=0.3
export energy_cutoff=(0 12.5 25 50 12.5 25 50 12.5 25 50)
export hepmc_output_file_path="${DETECTOR_PATH}/hepmc/ten_4x16_test"
export input_cry_file="${DETECTOR_PATH}/hepmc/120M_events.txt"

# DDsim Variables
export world_configurations=('free' 'target')
export input_detector_file="${DETECTOR_PATH}/detectors/square_v2/merge.xml"
export detector_pos_x=(0 0 0 0 50 50 50 -25 -25 -25)
export detector_pos_y=(0 0 0 0 0 0 0 -25 -25 -25)
export detector_pos_z=(1 -25 -50 -100 -25 -50 -100 -25 -50 -100)
export number_of_events=(1000 1000 1000 1000 1000 1000 1000 1000 1000 1000)