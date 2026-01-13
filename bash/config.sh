##Variable control for running both hepmc generation to DDsim simulation
export number_of_detector = 10
export detector_name = 'small_pixel_64'
export output_file_path = "${DETECTOR_PATH}/data/${detector_name}"
export hepmc_file_path = "${DETECTOR_PATH}/hepmc/${detector_name}"

mkdir ${output_file_path}
mkdir ${hepmc_file_path}


### CRY Variable
export CRY_num_of_events = (1000000 5000000 5000000 5000000 5000000 5000000 5000000 5000000 5000000 5000000) #number of events in raw CRY output file per detector

export generation_height = 10 #initial height (z) at where the muons are coming from the sky, not too high as muons are selected as ground level but higher than the detectors and objects
export detector_total_length = 0.3 
export energy_cutoff = (0 12.5 25 50 12.5 25 50 12.5 25 50)
export hepmc_output_file_path = "${DETECTOR_PATH}/hepmc/ten_4x16_test"
export input_cry_file = "${DETECTOR_PATH}/hepmc/120M_events.txt"


### DDsim Variable
export world_configurations = ('free' 'target')
export input_detector_file = "${DETECTOR_PATH}/detectors/square_v2/merge.xml"

export detector_pos_x = (0 0 0 0 50 50 50 -25 -25 -25)
export detector_pos_y = (0 0 0 0 0 0 0 -25 -25 -25)
export detector_pos_z = (1 -25 -50 -100 -25 -50 -100 -25 -50 -100)
export number_of_events = (1000 1000 1000 1000 1000 1000 1000 1000 1000 1000)
