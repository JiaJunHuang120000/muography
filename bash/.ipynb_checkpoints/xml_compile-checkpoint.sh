#!/bin/bash
source ${DETECTOR_PATH}/bash/config.sh

cd ${detector_file_path}

cp ${DETECTOR_PATH}/xml/soil_target.template.xml .
cp ${DETECTOR_PATH}/xml/merge.xml .

python3 ${DETECTOR_PATH}/python/generate_soil_target.py soil_target.template.xml


