#!/bin/bash

# Start measuring time
start=$(date +%s)

# Assign arrays from environment variables
numEvent=("${CRY_num_of_events[@]}")
x=("${detector_pos_x[@]}")
y=("${detector_pos_y[@]}")
z=("${detector_pos_z[@]}")
offset=$(echo "${detector_total_length}/2" | bc -l)  # Ensure offset is a float
cut=("${energy_cutoff[@]}")

# Loop over detectors to generate HEPMC files
for i in $(seq 0 $((number_of_detector - 1))); do
    hepmc_file="${hepmc_file_path}/${detector_name}_${i}.hepmc"
    echo "Running HEPMC generation for detector ${i}..."
    
    ./remote "${input_cry_file}" "${hepmc_file}" "${generation_height}" "${x[$i]}" "${y[$i]}" "${z[$i]}" "${offset}" "${cut[$i]}" "${numEvent[$i]}"
    
done

# End time for HEPMC generation
end=$(date +%s)
runtime=$((end-start))
echo "Total runtime for HEPMC generation: ${runtime} seconds"