#!/bin/bash
# Quick smoke-test generator for CRY and HepMC conversion.
#
# Steps:
# 1) Run CRY testMain with cpp/setup.file and save text output.
# 2) Run ./remote once to produce a sample HepMC file.
#
# Useful for validating a fresh install before running the full pipeline.
cd cry_v1.7/test/
./testMain ../../cpp/setup.file 10000 > ../../hepmc/output.txt

cd ../..
mkdir -p hepmc
./remote cry_v1.7/test/output.txt hepmc/output.hepmc 10 0 0 0 0.1 0 100
#rm cry_v1.7/test/output.txt

