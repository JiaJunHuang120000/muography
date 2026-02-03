import numpy as np, uproot as ur, awkward as ak, pandas as pd
import matplotlib.pyplot as plt
import os, sys
import pickle
from scipy.optimize import curve_fit

def gaus(x, amp, mean, sigma):
    return amp * np.exp( -(x - mean)**2 / (2*sigma**2) ) 

def phi_reconstruct(x, y, z):
    return np.degrees(np.arctan2(y, x))
    
def theta_reconstruct(x, y, z):
    return np.degrees(np.arccos(abs(z)/np.sqrt(x**2+y**2+z**2)))
    
def theta_x(x, y, z):
    return x/z

def theta_y(x, y, z):
    return y/z
    
def vector_angle_reconstruct(x, y, z):
    data = np.concatenate((np.array(x)[:, np.newaxis], 
                           np.array(y)[:, np.newaxis], 
                           np.array(z)[:, np.newaxis]), 
                          axis=1)
    datamean = data.mean(axis=0)
    centered_data = data - datamean

    _, _, vv = np.linalg.svd(centered_data)
    direction_vector = vv[0]
    if direction_vector[2] > 0:
        direction_vector *= -1
        
    x_vec, y_vec, z_vec = direction_vector
    
    theta = theta_reconstruct(x_vec, y_vec, z_vec)
    phi = phi_reconstruct(x_vec, y_vec, z_vec)
    
    return theta, phi
    
def theta_x_y(x, y, z):
    data = np.concatenate((np.array(x)[:, np.newaxis], 
                           np.array(y)[:, np.newaxis], 
                           np.array(z)[:, np.newaxis]), 
                          axis=1)
    datamean = data.mean(axis=0)

    centered_data = data - datamean

    _, _, vv = np.linalg.svd(centered_data)
    direction_vector = vv[0]
    if direction_vector[2] > 0:
        direction_vector *= -1
        
    x_vec, y_vec, z_vec = direction_vector
    
    tanx = theta_x(x_vec, y_vec, z_vec)
    tany = theta_y(x_vec, y_vec, z_vec)
    
    return tanx, tany


path = os.getenv('output_file_path')+'/'
files = sorted(os.listdir(path))

posx = list(map(float, os.environ["detector_pos_x_env"].split()))
posy = list(map(float, os.environ["detector_pos_y_env"].split()))
posz = list(map(float, os.environ["detector_pos_z_env"].split()))

for config in ['free','target']:

    batches = pd.DataFrame()
    output_file = f'{config}_merge.pkl'
    for file in files:
        if config+'merge' in file: continue
        if config not in file: continue
    
        with ur.open(path+file+":events") as f:
            arrays = f.arrays(filter_name=["MuographyHits.energy", "MuographyHitsContributions.time", 
                                           "MuographyHits.position.x", "MuographyHits.position.y", "MuographyHits.position.z", 
                                           "MCParticles.PDG", "MCParticles.generatorStatus", 
                                           "MCParticles.momentum.x", "MCParticles.momentum.y", "MCParticles.momentum.z", 
                                           "MCParticles.vertex.x", "MCParticles.vertex.y", "MCParticles.vertex.z", 
                                           "MCParticles.mass"])
            
        y,x=np.histogram(ak.flatten(arrays["MuographyHits.energy"]), bins=100, range=(0, 0.004))
        bc=(x[1:]+x[:-1])/2
        MIP=list(bc[y==max(y[3:])])[0] 
        
        data_energy = arrays[f'MuographyHits.energy']
        
        sigma = 0.56
        
        # flatten to numpy
        flat = ak.to_numpy(data_energy.layout.content)
        noise = np.random.normal(0, sigma, size=len(flat))*MIP
        
        # add noise
        flat_smear = np.clip(flat + noise, a_min=1e-16, a_max=None)
        
        offsets = ak.to_numpy(data_energy.layout.offsets)  # convert Index64 → numpy
        lengths = offsets[1:] - offsets[:-1]
        data_energy_smear = ak.unflatten(flat_smear, lengths)
        
        data_MIP_cut = data_energy_smear > 0.2*MIP
        data_cell_cut = ak.num(arrays[f'MuographyHits.energy'], axis=1) >= 2
        
        data_energy = data_energy[data_cell_cut]
        data_energy_smear = data_energy_smear[data_cell_cut]
        data_x = arrays[f'MuographyHits.position.x'][data_cell_cut]
        data_y = arrays[f'MuographyHits.position.y'][data_cell_cut]
        data_z = arrays[f'MuographyHits.position.z'][data_cell_cut]    
        reco_data_angle = np.array([vector_angle_reconstruct(np.array(xi,dtype=float), np.array(yi,dtype=float), np.array(zi,dtype=float)) for xi, yi, zi in zip(data_x,data_y,data_z)])
        reco_data_angle_1 = np.array([theta_x_y(np.array(xi,dtype=float), np.array(yi,dtype=float), np.array(zi,dtype=float)) for xi, yi, zi in zip(data_x,data_y,data_z)])
        
        data_theta = ak.Array(reco_data_angle[:,0])
        data_phi = ak.Array(reco_data_angle[:,1])
        data_time = arrays[f'MuographyHitsContributions.time'][data_cell_cut]
        status = arrays["MCParticles.generatorStatus"]
        mc_px = arrays["MCParticles.momentum.x"][status==1][data_cell_cut]
        mc_py = arrays["MCParticles.momentum.y"][status==1][data_cell_cut]
        mc_pz = arrays["MCParticles.momentum.z"][status==1][data_cell_cut]
        mc_x = arrays["MCParticles.vertex.x"][status==1][data_cell_cut]
        mc_y = arrays["MCParticles.vertex.y"][status==1][data_cell_cut]
        mc_z = arrays["MCParticles.vertex.z"][status==1][data_cell_cut]
        mc_theta = theta_reconstruct(mc_px,mc_py,mc_pz)
        mc_phi = phi_reconstruct(mc_px,mc_py,mc_pz)
        mc_PDG = arrays["MCParticles.PDG"][status==1][data_cell_cut]
        mc_mass = arrays["MCParticles.mass"][status==1][data_cell_cut]
        status = status[status==1][data_cell_cut]
        detector = np.full(len(status), file.split('.edm4hep.root')[0].split('_')[-1], dtype=np.int32) #file.split('.edm4hep.root')[0].split('_')[-1]
        data_energy_sum = np.sum(data_energy_smear,axis=1)
        
        H, xedges, yedges = np.histogram2d(
            np.array(np.concatenate(mc_theta)),
            np.array(data_energy_sum),
            bins=(200, np.linspace(0,15/1000,200))
        )
        
        X, Y = np.meshgrid(xedges, yedges)
 
        peak_indices = np.argmax(H, axis=1)
        theta_centers = 0.5 * (xedges[:-1] + xedges[1:])
        energy_centers = 0.5 * (yedges[:-1] + yedges[1:])
        E_peaks = energy_centers[peak_indices]
        def exp_decay(theta, A, k, C):
            return A * np.exp(k * theta) + C
        # Remove NaNs or weird edges
        mask = np.isfinite(E_peaks)
        theta_fit = theta_centers[mask]
        E_fit = E_peaks[mask]
        
        # popt, pcov = curve_fit(exp_decay, theta_fit, E_fit, p0=(0.001, 0.05, 0.001))
        # A, k, C = popt
        # #print("Fit parameters: A=%.3f, k=%.3f, C=%.3f" % (A, k, C))
        # theta_smooth = np.linspace(theta_fit.min(), theta_fit.max(), 300)
        # plt.plot(theta_smooth, exp_decay(theta_smooth, *popt), '--', lw=2, label="Exp Fit: A=%.3f, B=%.3f, C=%.3f" % (A, k, C))
        # legend = plt.legend(facecolor="black", edgecolor="white")
        # for text in legend.get_texts():
        #     text.set_color("white")
        # def theta_from_energy(E, A, k, C):
        #     return (1.0 / k) * np.log((E - C) / A)
        # plt.close()
        
        # data_theta_engergy = theta_from_energy(np.array(data_energy_sum),*popt)
            
        branches = {
            "detector": ak.Array(detector),
            "theta_reco": ak.Array(data_theta),
            "phi_reco": ak.Array(data_phi),
            "theta_true": ak.Array(ak.flatten(mc_theta)),
            "phi_true": ak.Array(ak.flatten(mc_phi)),
            "event_energy": ak.Array(data_energy_sum),
            "theta_reco_x": ak.Array(ak.Array(reco_data_angle_1[:,0])),
            "theta_reco_y": ak.Array(ak.Array(reco_data_angle_1[:,1]))
            #"theta_energy": ak.Array(data_theta_engergy)
        }   
        batch = pd.DataFrame(branches)
        
        batches = pd.concat([batches, batch], ignore_index=True)
        
        num = 10000
        i = 0
        for i in range(int(len(mc_theta)/num)):
            if len(mc_theta) < num: continue
            print(f"Processing {file}: {i}/{int(len(mc_theta)/num)-1}", end='\r',flush=True)
            branches = {
                "MuographyHits.position.x": data_x[i*num:(i+1)*num],
                "MuographyHits.position.y": data_y[i*num:(i+1)*num],
                "MuographyHits.position.z": data_z[i*num:(i+1)*num],
                "MuographyHits.time": data_time[i*num:(i+1)*num],
                "MuographyHits.energy_nonsmear": data_energy[i*num:(i+1)*num],
                "MuographyHits.energy": data_energy_smear[i*num:(i+1)*num],
                "MCParticles.generatorStatus": status[i*num:(i+1)*num],
                "MCParticles.PDG": mc_PDG[i*num:(i+1)*num],
                "MCParticles.mass": mc_mass[i*num:(i+1)*num],
                "MCParticles.momentum.x": mc_px[i*num:(i+1)*num],
                "MCParticles.momentum.y": mc_py[i*num:(i+1)*num],
                "MCParticles.momentum.z": mc_pz[i*num:(i+1)*num]
            }
            with ur.recreate(path+f'split/{file}_{i:03d}.root') as fout:
                fout["events"] = branches
        if len(mc_theta)%10000 != 0:
            if len(mc_theta) < num: i -= 1
            print(f"Processing {file}: {i+1}/{int(len(mc_theta)/num)}", end='\r',flush=True)
            branches = {
                "MuographyHits.position.x": data_x[(i+1)*num:len(mc_theta)],
                "MuographyHits.position.y": data_y[(i+1)*num:len(mc_theta)],
                "MuographyHits.position.z": data_z[(i+1)*num:len(mc_theta)],
                "MuographyHits.time": data_time[(i+1)*num:len(mc_theta)],
                "MuographyHits.energy_nonsmear": data_energy[(i+1)*num:len(mc_theta)],
                "MuographyHits.energy": data_energy_smear[(i+1)*num:len(mc_theta)],
                "MCParticles.generatorStatus": status[(i+1)*num:len(mc_theta)],
                "MCParticles.PDG": mc_PDG[(i+1)*num:len(mc_theta)],
                "MCParticles.mass": mc_mass[(i+1)*num:len(mc_theta)],
                "MCParticles.momentum.x": mc_px[(i+1)*num:len(mc_theta)],
                "MCParticles.momentum.y": mc_py[(i+1)*num:len(mc_theta)],
                "MCParticles.momentum.z": mc_pz[(i+1)*num:len(mc_theta)]
            }
        
            with ur.recreate(path+f'split/{file}_{i+1:03d}.root') as fout:
                fout["events"] = branches
        
        
        
    with open(path+output_file, "wb") as fout:
        pickle.dump(batches, fout)
    print('/nDone' + f'for {config}' + '/n')

