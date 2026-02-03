#include "HepMC3/GenEvent.h"
#include "HepMC3/WriterAscii.h"
#include "HepMC3/Print.h"
#include "HepMC3/WriterAsciiHepMC2.h"

#include <TDatabasePDG.h>
#include <TParticlePDG.h>
#include <TLorentzVector.h>
#include <TVector3.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <random>

using namespace HepMC3;

struct MuonEvent {
    double ke;          // kinetic energy in MeV
    double x, y, z;     // position in meters
    double u, v, w;     // direction cosines (unit vector)
};

void gen_cosmic_muons(const std::string& input_file,
                      const std::string& output_file,
                      float target_height,
                      float detector_x,
                      float detector_y,
                      float detector_z,
                      float z_off,
                      float y_off,
                      float x_off,
                      float E_cut,
                      int max_events)   // 🚨 stop after this many general events
{
    const double target_z = target_height * 1000.0;

    std::ifstream input(input_file);
    if (!input.is_open()) {
        std::cerr << "Could not open input file" << std::endl;
        return;
    }

    WriterAscii hepmc_output(output_file.c_str());

    TDatabasePDG *pdg = new TDatabasePDG();
    TParticlePDG *muon_pdg = pdg->GetParticle("mu-");
    if (!muon_pdg) {
        std::cerr << "Error: mu- not found in PDG database" << std::endl;
        return;
    }

    const double muon_mass = muon_pdg->Mass(); // GeV
    const int muon_pdgID = muon_pdg->PdgCode();

    std::string line;
    int accepted_events = 0;
    int processed_events = 0;

    while (std::getline(input, line)) {
        if (processed_events >= max_events) break;

        if (line.find("Secondary") == std::string::npos || 
            line.find("muon") == std::string::npos) {
            continue;
        }

        processed_events++;

        // --- Parse muon from line ---
        MuonEvent muon;
        std::istringstream iss(line);
        std::string dummy;

        iss >> dummy >> dummy >> dummy; // Skip "Secondary 0 muon"
        iss >> dummy; 
        size_t eq_pos = dummy.find('=');
        if (eq_pos == std::string::npos) continue;
        muon.ke = std::stod(dummy.substr(eq_pos + 1));
        iss >> dummy; // Skip "(MeV)"
        iss >> dummy; // Skip "(x,y,z)="
        iss >> muon.x >> muon.y >> muon.z;
        iss >> dummy; // Skip "(u,v,w)="
        iss >> muon.u >> muon.v >> muon.w;
        
        muon.x *= 1000.0;
        muon.y *= 1000.0;
        muon.z *= 1000.0;
        
        // Normalize direction
        double norm = sqrt(muon.u*muon.u + muon.v*muon.v + muon.w*muon.w);
        if (norm > 0) {
            muon.u /= norm;
            muon.v /= norm;
            muon.w /= norm;
        }

        // --- Event building ---
        const double start_x = detector_x * 1000.0;
        const double start_y = detector_y * 1000.0;
        const double start_z = detector_z * 1000.0; 

        const double total_energy = (muon.ke / 1000.0) + muon_mass; // MeV->GeV
        if (total_energy < E_cut) continue;
        const double p_mag = sqrt(total_energy*total_energy - muon_mass*muon_mass);

        const double t = (target_z - start_z) / muon.w;
        const double x_intersect = muon.x + t * muon.u;
        const double y_intersect = muon.y + t * muon.v;

        TVector3 direction(-1*muon.u, -1*muon.v, muon.w);
        const TVector3 momentum = direction * p_mag;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> z_dist(-z_off * 1000.0, z_off * 1000.0);
        std::uniform_real_distribution<> y_dist(-y_off * 1000.0, y_off * 1000.0);
        std::uniform_real_distribution<> x_dist(-x_off * 1000.0, x_off * 1000.0);

        
        const TVector3 position(-1*(x_intersect)+start_x+x_dist(gen), 
                                -1*(y_intersect)+start_y+y_dist(gen), 
                                target_z+z_dist(gen));

        // Create HepMC3 event
        GenEvent evt(Units::GEV, Units::MM);
        evt.set_event_number(accepted_events);

        // Beam particles
        GenParticlePtr p1 = std::make_shared<GenParticle>(
            FourVector(0.0, 0.0, 10.0, 10.0), 11, 4);
        GenParticlePtr p2 = std::make_shared<GenParticle>(
            FourVector(0.0, 0.0, 0.0, 0.938), 2212, 4);

        GenVertexPtr primary_vtx = std::make_shared<GenVertex>();
        primary_vtx->add_particle_in(p1);
        primary_vtx->add_particle_in(p2);
        //evt.add_vertex(primary_vtx);

        // Muon
        GenParticlePtr p_muon = std::make_shared<GenParticle>(
            FourVector(momentum.X(), momentum.Y(), momentum.Z(), total_energy), 
            muon_pdgID, 1);

        GenVertexPtr muon_vtx = std::make_shared<GenVertex>(
            FourVector(position.X(), position.Y(), position.Z(), 0.0));
        muon_vtx->add_particle_in(p1);
        muon_vtx->add_particle_out(p_muon);
        evt.add_vertex(muon_vtx);

        hepmc_output.write_event(evt);
        accepted_events++;

        if (accepted_events == 1) {
            std::cout << "First accepted event: " << std::endl;
            Print::listing(evt);
            std::cout << "Intersection at z=0: (" << x_intersect 
                      << ", " << y_intersect << ")" << std::endl;
        }

        if (accepted_events % 10000 == 0) {
            std::cout << "\rAccepted events: " << accepted_events << std::flush;
            std::cout << std::endl;
        }
    }

    hepmc_output.close();
    std::cout << "Finished after " << processed_events << " general events." << std::endl;
    std::cout << "Accepted muons: " << accepted_events << " (" << (100.0*accepted_events/max_events) << "%)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 12) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file.txt> <output_file.hepmc>"
                  << " <muon_generation_height>"
                  << " <detector_position_x>"
                  << " <detector_position_y>"
                  << " <detector_position_z>" 
                  << " <z_offset>"
                  << " <y_offset>"
                  << " <x_offset>"
                  << " <E_cut>"
                  << " <max_events>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    float target_height = std::stof(argv[3]);
    float detector_x     = std::stof(argv[4]);
    float detector_y     = std::stof(argv[5]);
    float detector_z     = std::stof(argv[6]);
    float z_off          = std::stof(argv[7]);
    float y_off          = std::stof(argv[8]);
    float x_off          = std::stof(argv[9]);
    float E_cut          = std::stof(argv[10]);
    int max_events       = std::stoi(argv[11]);

    gen_cosmic_muons(input_file, output_file, target_height,
                     detector_x, detector_y, detector_z,
                     z_off, y_off, x_off, E_cut, max_events);

    return 0;
}
