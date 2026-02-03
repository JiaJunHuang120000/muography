#!/bin/bash

source ${DETECTOR_PATH}/bash/config.sh
# Start measuring time for DDsim
start=$(date +%s)

# Loop over world configurations and detectors for DDsim
for j in $(seq 0 $(( ${#world_configurations[@]}-1))); do
    for i in $(seq 0 $(( ${number_of_detector}-1))); do
        # Export variables for DDsim
        export DET_X="${detector_pos_x[$i]}"
        export DET_Y="${detector_pos_y[$i]}"
        export DET_Z="${detector_pos_z[$i]}"
        export CONFIG="${world_configurations[$j]}"

        # Run DDsim with the current configuration
        ddsim \
            --compactFile "${input_detector_file}" \
            --inputFiles "${hepmc_file_path}/${detector_name}_${i}.hepmc" \
            --outputFile "${output_file_path}/${detector_name}_${world_configurations[$j]}_${i}.edm4hep.root" \
            --numberOfEvents ${number_of_events[$i]} \
            --steeringFile "${DETECTOR_PATH}/python/steering.py"
    done
done

# End time for DDsim simulation
end=$(date +%s)
runtime=$((end-start))
echo "Total runtime for DDsim simulation: ${runtime} seconds"
