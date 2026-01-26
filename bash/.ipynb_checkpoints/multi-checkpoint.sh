#!/bin/bash
source ${DETECTOR_PATH}/bash/config.sh

# Start measuring time
start=$(date +%s)

# Loop over detectors to generate HEPMC files
for i in $(seq 0 $(( ${number_of_detector}-1))); do
    hepmc_file="${hepmc_file_path}/${detector_name}_${i}.hepmc"
    echo "Running HEPMC generation for detector ${i}..."
    echo "${input_cry_file}" "${hepmc_file}" "${generation_height}" "${detector_pos_x[$i]}" "${detector_pos_y[$i]}" "${detector_pos_z[$i]}" "${detector_z_offset}" "${detector_y_offset}" "${detector_x_offset}" "${energy_cutoff[$i]}" "${CRY_num_of_events[$i]}"
    ./remote "${input_cry_file}" "${hepmc_file}" "${generation_height}" "${detector_pos_x[$i]}" "${detector_pos_y[$i]}" "${detector_pos_z[$i]}" "${detector_z_offset}" "${detector_y_offset}" "${detector_x_offset}" "${energy_cutoff[$i]}" "${CRY_num_of_events[$i]}"
    
done

# End time for HEPMC generation
end=$(date +%s)
runtime=$((end-start))
echo "Total runtime for HEPMC generation: ${runtime} seconds"