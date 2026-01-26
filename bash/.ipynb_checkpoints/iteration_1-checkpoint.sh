start=$(date +%s)
# events (pass as first argument, default 1000)
numEvent=${1:-1000}
configs=('free' 'target')
x=(0 0 0 0 50 50 50 -25 -25 -25)
y=(0 0 0 0 0 0 0 -25 -25 -25)
z=(1 -25 -50 -100 -25 -50 -100 -25 -50 -100)
output_path="${DETECTOR_PATH}/data/nine_v5_1"
hepmc_path="${DETECTOR_PATH}/hepmc/ten_4x16"
xml="${DETECTOR_PATH}/detectors/square_v2/merge_1.xml"
# Detector XML configs
# Loop over XML files and i=0..5

mkdir ${output_path}
for j in {0..1}; do
    output_files=()
    for i in {0..9}; do
    
        export DET_X=${x[i]}
        export DET_Y=${y[i]}
        export DET_Z=${z[i]}
        export CONFIG=${configs[j]}

        
        output_file="ten_4x16_${configs[j]}_${i}.edm4hep.root"
        hepmc_file="nine_detectors_${i}.hepmc"
        ddsim \
            --compactFile ${xml} \
            --inputFiles   ${hepmc_path}/${hepmc_file} \
            --outputFile   ${output_path}/${output_file} \
            --numberOfEvents 5000000 \
    	    --steeringFile steering.py
    done

done
end=$(date +%s)
runtime=$((end-start))
echo "Total runtime: ${runtime} seconds"
