bash bash/hepmc_compile.sh
bash bash/cry_compile.sh
echo "Compiled CRY Library ############################"

bash bash/gen_cry.sh
echo "Generated CRY Cosmic Muons ###########################"

curl -L https://github.com/eic/eic-shell/raw/main/install.sh | bash
./eic-shell
source setup_env.sh
bash build.sh
ddsim --compactFile xml/telescope.xml --outputFile data/test.root --inputFile hepmc/output.hepmc --numberOfEvents 5
