start=$(date +%s)
numEvent=${number_of_events}
configs=${world_configurations}
x=${detector_pos_x}
y=${detector_pos_y}
z=${detector_pos_z}
hepmc_path=${input_hepmc_path}

for j in $(seq 0 ${#world_configurations}); do
    for i in $(seq 0 ${number_of_detector}); do
    
        export DET_X=${x[i]}
        export DET_Y=${y[i]}
        export DET_Z=${z[i]}
        export CONFIG=${configs[j]}

        
        ddsim \
            --compactFile ${input_detector_file} \
            --inputFiles   "${hepmc_file_path}/${detector_name}_${i}.hepmc" \
            --outputFile   "${output_file_path}/${detector_name}_${configs[j]}_${i}.edm4hep.root" \
            --numberOfEvents ${numEvent[i]} \
    	    --steeringFile ${DETECTOR_PATH}/steering.py
    done

done
end=$(date +%s)
runtime=$((end-start))
echo "Total runtime: ${runtime} seconds"
