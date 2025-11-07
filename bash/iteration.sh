start=$(date +%s)
# events (pass as first argument, default 1000)
numEvent=${1:-1000}
configs=('free' 'target')
x=(0 0 0 0)
y=(0 0 0 0)
z=(1 -25 -50 -100)
# Detector XML configs
# Loop over XML files and i=0..5

for j in {0..1}; do
    output_files=()
    for i in {0..3}; do
    
        export DET_X=${x[i]}
        export DET_Y=${y[i]}
        export DET_Z=${z[i]}
        export CONFIG=${configs[j]}

        xml="test/merge.xml"
        output_file="testing_merge_${configs[j]}_${i}.edm4hep.root"

        ddsim \
            --compactFile ~/muography_backup/detectors/${xml} \
            --inputFiles   ~/muography/cry_v1.7/test/square_2x4x8_v2_1.hepmc \
            --outputFile   ~/muography/data/${output_file} \
            --numberOfEvents ${numEvent} \
	    --steeringFile steering.py
    done

done
end=$(date +%s)
runtime=$((end-start))
echo "Total runtime: ${runtime} seconds"
