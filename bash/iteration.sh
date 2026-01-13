#!/bin/bash

# Start measuring time for DDsim
start=$(date +%s)

numEvent=("${number_of_events[@]}")
configs=("${world_configurations[@]}")
hepmc_path="${hepmc_file_path}"

# Loop over world configurations and detectors for DDsim
for j in $(seq 0 $(( ${#configs[@]} - 1 ))); do
    for i in $(seq 0 $((number_of_detector - 1))); do
        
        # Export variables for DDsim
        export DET_X="${x[$i]}"
        export DET_Y="${y[$i]}"
        export DET_Z="${z[$i]}"
        export CONFIG="${configs[$j]}"

        # Run DDsim with the current configuration
        ddsim \
            --compactFile "${input_detector_file}" \
            --inputFiles "${hepmc_file_path}/${detector_name}_${i}.hepmc" \
            --outputFile "${output_file_path}/${detector_name}_${configs[$j]}_${i}.edm4hep.root" \
            --numberOfEvents "${numEvent[$i]}" \
            --steeringFile "${DETECTOR_PATH}/steering.py"
    done
done

# End time for DDsim simulation
end=$(date +%s)
runtime=$((end-start))
echo "Total runtime for DDsim simulation: ${runtime} seconds"