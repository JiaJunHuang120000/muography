start=$(date +%s)

numEvent=${CRY_num_of_events}
x=${detector_pos_x}
y=${detector_pos_y}
z=${detector_pos_z}
offset=${detector_total_length}/2
cut=${energy_cutoff}


for i in $(seq 0 ${number_of_detector}); do
	#output_file="${output_file_path}/${detector_name}_${i}.txt"
	hepmc_file="${hepmc_file_path}/${detector_name}_${i}.hepmc"
	#./testMain setup.file ${numEvent[i]} > ${output_file}
	./remote ${input_cry} ${hepmc_file} ${generation_height} ${x[i]} ${y[i]} ${z[i]} ${offset} ${cut[i]} ${numEvent[i]}
	#rm ${output_file}

done
end=$(date +%s)
runtime=$((end-start))
echo "Total runtime: ${runtime} seconds"
