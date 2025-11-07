start=$(date +%s)
iteration=3
numEvent=(100000 7000000 30000000 30000000)
target_height=10
x=(0 0 0 0)
y=(0 0 0 0)
z=(1 -25 -50 -100)
offset=0.5
cut=(0 12.5 25 50)
for i in $(seq 0 $iteration); do
	output_file="output_${i}.txt"
	hepmc_file="five_detector_${i}.hepmc"
	#./testMain setup.file ${numEvent[i]} > ${output_file}
	./back_project_single 120M_events.txt ${hepmc_file} ${target_height} ${x[i]} ${y[i]} ${z[i]} ${offset} ${cut[i]} 100 #${numEvent[i]}
	#rm ${output_file}

done
end=$(date +%s)
runtime=$((end-start))
echo "Total runtime: ${runtime} seconds"
