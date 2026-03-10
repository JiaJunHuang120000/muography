#!/bin/bash
# Prepare detector XML inputs for simulation.
#
# Steps:
# 1) Copy merge and soil target template XML into detector-specific folder.
# 2) Generate soil_free.xml and soil_target.xml from TARGETS in config.sh.
source ${DETECTOR_PATH}/bash/config.sh

cd ${detector_file_path}

cp ${DETECTOR_PATH}/xml/soil_target.template.xml .
cp ${DETECTOR_PATH}/xml/merge.xml .

python3 ${DETECTOR_PATH}/python/generate_soil_target.py soil_target.template.xml

