start=$(date +%s)

iteration=9
numEvent=(1000000 120000000 120000000 120000000 120000000 120000000 120000000 120000000 120000000 120000000)
target_height=10
x=(0 0 0 0 50 50 50 -25 -25 -25)
y=(0 0 0 0 0 0 0 -25 -25 -25)
z=(1 -25 -50 -100 -25 -50 -100 -25 -50 -100)
offset=0.5
cut=(0 12.5 25 50 12.5 25 50 12.5 25 50)
for i in $(seq 0 $iteration); do
	output_file="output_${i}.txt"
	hepmc_file="nine_detectors_${i}.hepmc"
	#./testMain setup.file ${numEvent[i]} > ${output_file}
	./remote 120M_events.txt ${hepmc_file} ${target_height} ${x[i]} ${y[i]} ${z[i]} ${offset} ${cut[i]} ${numEvent[i]}
	#rm ${output_file}

done
end=$(date +%s)
runtime=$((end-start))
echo "Total runtime: ${runtime} seconds"
