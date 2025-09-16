#include "HepMC3/GenEvent.h"
#include "HepMC3/WriterAscii.h"
#include "HepMC3/Print.h"

#include <TDatabasePDG.h>
#include <TParticlePDG.h>
#include <TLorentzVector.h>
#include <TVector3.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>

using namespace HepMC3;

struct MuonEvent {
    double ke;          // kinetic energy in MeV
    double x, y, z;     // position in meters
    double u, v, w;     // direction cosines (unit vector)
};

std::vector<MuonEvent> readCRYMuonEvents(const std::string& filename) {
    std::vector<MuonEvent> events;
    std::ifstream input(filename);
    std::string line;
    
    while (std::getline(input, line)) {
        if (line.find("Secondary") != std::string::npos && line.find("muon") != std::string::npos) {
            MuonEvent event;
            std::istringstream iss(line);
            std::string dummy;
            
            // Parse line format: "Secondary 0 muon ke=1183.05 (MeV) (x,y,z)= -1.18511 -0.580884 0 (u,v,w)= 0.213874 0.502602 -0.837645 (m)"
            iss >> dummy >> dummy >> dummy; // Skip "Secondary 0 muon"
            
            // Parse kinetic energy (ke=1183.05)
            iss >> dummy; 
            size_t eq_pos = dummy.find('=');
            if (eq_pos == std::string::npos) continue;
            event.ke = std::stod(dummy.substr(eq_pos + 1));
            
            iss >> dummy; // Skip "(MeV)"
            
            // Parse position (x,y,z)
            iss >> dummy; // Skip "(x,y,z)="
            iss >> event.x >> event.y >> event.z;
            
            // Parse direction (u,v,w)
            iss >> dummy; // Skip "(u,v,w)="
            iss >> event.u >> event.v >> event.w;
            
            // Normalize direction vector (just in case)
            double norm = sqrt(event.u*event.u + event.v*event.v + event.w*event.w);
            if (norm > 0) {
                event.u /= norm;
                event.v /= norm;
                event.w /= norm;
            }
            
            events.push_back(event);
        }
    }
    return events;
}

void gen_cosmic_muons(const std::string& input_file,
                      const std::string& output_file,
                      float target_height,
                      float detector_x,
                      float detector_y,
                      float detector_z,
                      float z_off,
                      float E_cut){

    const double target_z = target_height * 1000.0;
    
    std::vector<MuonEvent> muon_events = readCRYMuonEvents(input_file);
    if (muon_events.empty()) {
        std::cerr << "No muon events found in input file!" << std::endl;
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

    int accepted_events = 0;

    for (size_t i = 0; i < muon_events.size(); i++) {
				
		const double start_x = detector_x * 1000.0;
		const double start_y = detector_y * 1000.0;
    	const double start_z = detector_z * 1000.0; 
        
    	const auto& muon = muon_events[i];
        
        // Apply geometric condition

        // Convert units and calculate kinematics
        // Note: We'll place the vertex at (x,y,50m) as specified
        const double x_mm = muon.x * 1000.0;
        const double y_mm = muon.y * 1000.0;
        const double z_mm = start_z; 
        const double total_energy = (muon.ke / 1000.0) + muon_mass; // Convert MeV to GeV
        if (total_energy < E_cut) continue;
        const double p_mag = sqrt(total_energy*total_energy - muon_mass*muon_mass);
        
        const double t = (target_z - start_z) / muon.w;
    
        // Calculate x,y position at z=0
        const double x_intersect = muon.x + t * muon.u;
        
        const double y_intersect = muon.y + t * muon.v;
        
        TVector3 direction(-1*muon.u, -1*muon.v, muon.w);
        const TVector3 momentum = direction * p_mag;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> z_dist(-z_off * 1000.0, z_off * 1000.0);
        
        const TVector3 position(-1*(x_intersect)+start_x, -1*(y_intersect)+start_y, target_z+z_dist(gen));

        // Create HepMC3 event
        GenEvent evt(Units::GEV, Units::MM);
        evt.set_event_number(accepted_events);

        // Create beam particles (electron and proton)
        GenParticlePtr p1 = std::make_shared<GenParticle>(
            FourVector(0.0, 0.0, 10.0, 10.0), 11, 4);
        GenParticlePtr p2 = std::make_shared<GenParticle>(
            FourVector(0.0, 0.0, 0.0, 0.938), 2212, 4);

        // Primary vertex with beam particles
        GenVertexPtr primary_vtx = std::make_shared<GenVertex>();
        primary_vtx->add_particle_in(p1);
        primary_vtx->add_particle_in(p2);
        evt.add_vertex(primary_vtx);

        // Muon particle and vertex at (x,y,50m)
        GenParticlePtr p_muon = std::make_shared<GenParticle>(
            FourVector(momentum.X(), momentum.Y(), momentum.Z(), total_energy), 
            muon_pdgID, 1);
        
        GenVertexPtr muon_vtx = std::make_shared<GenVertex>(
            FourVector(position.X(), position.Y(), position.Z(), 0.0));
        muon_vtx->add_particle_in(p1); // Connect to beam particle
        muon_vtx->add_particle_out(p_muon);
        evt.add_vertex(muon_vtx);

        // Write event
        hepmc_output.write_event(evt);
        accepted_events++;
        
        if (accepted_events == 1) {
            std::cout << "First accepted event: " << std::endl;
            Print::listing(evt);
            
            // Print some debugging info
            double t = -start_z / muon.w;
            double x_intersect = muon.x + t * muon.u;
            double y_intersect = muon.y + t * muon.v;
            std::cout << "Intersection at z=0: (" << x_intersect << ", " << y_intersect << ")" << std::endl;
        }
        
        if (accepted_events % 100 == 0) {
            std::cout << "Accepted events: " << accepted_events << std::endl;
        }
    }
    
    hepmc_output.close();
    std::cout << "Total altered muons: " << accepted_events << "/" << muon_events.size() 
              << " (" << (100.0*accepted_events/muon_events.size()) << "%)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file.txt> <output_file.hepmc>"
                  << " <muon_generation_height>"
                  << " <detector_position_x>"
                  << " <detector_position_y>"
                  << " <detector_position_z>" 
                  << " <z_offset>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    float target_height = std::stof(argv[3]);
    float detector_x     = std::stof(argv[4]);
    float detector_y     = std::stof(argv[5]);
    float detector_z     = std::stof(argv[6]);
    float z_off          = std::stof(argv[7]);
    float E_cut         = std::stof(argv[8]);

    gen_cosmic_muons(input_file, output_file, target_height, detector_x, detector_y, detector_z, z_off, E_cut);
    return 0;
}

