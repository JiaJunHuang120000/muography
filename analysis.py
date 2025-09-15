import numpy as np, uproot as ur, awkward as ak, pandas as pd
import matplotlib.pyplot as plt
import os, sys
import pickle
import scipy.stats as st
from collections import defaultdict
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy.stats import poisson, norm
from scipy.spatial import ConvexHull


import mplhep as hep
plt.figure()
hep.style.use("CMS")
plt.close()

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

def phi_reconstruct(x, y, z):
    return np.degrees(np.arctan2(y, x))

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

def vector_angle_reconstruct(x, y, z): # Cell hit positions from detector, 
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

def poisson_to_Z(N, lam):
    N = np.asarray(N, dtype=float)
    lam = np.asarray(lam, dtype=float)
    # Poisson CDF
    p = np.where(N >= lam,
                 poisson.cdf(N, lam),
                 1.0 - poisson.cdf(N, lam))
    
    # clip to avoid 0 or 1
    #eps = 1e-16
    #p = np.clip(p, eps, 1 - eps)
    
    # sign convention
    s = np.sign(lam - N)
    
    return s * norm.ppf(p)

arrays = {}
num = 9
acceptance = 1.5
positions_x = [0,0,0,100,100,100,50,50,50]
positions_y = [0,0,0,0,0,0,50,50,50]
positions_z = [1,-25,-50,1,-25,-50,1,-25,-50]

simfile = '10x10_cm_detector_vacuum_sphere_merge.root'
arrays['muon_sim']=ur.open("data/"+simfile+":events").arrays()

datafile = '10x10_cm_detector_CarbonFiber_vacuum_sphere_merge.root'
arrays['muon_data']=ur.open("data/"+datafile+":events").arrays()

reco_data_angles = []
reco_sim_angles = []
energies = []

for i in range(num):
    data_energy = arrays['muon_data'][f'HcalFarForwardZDCHits{i}.energy']*1000
    
    y,x=np.histogram(ak.flatten(data_energy), bins=100, range=(0, 4))
    bc=(x[1:]+x[:-1])/2
    MIP=list(bc[y==max(y)])[0] 
    data_MIP_cut = data_energy > 0.5*MIP/1000
    data_cell_cut = [True if len(cells)>=2 else False for cells in arrays['muon_data'][f"HcalFarForwardZDCHits{i}.energy"][data_MIP_cut]]
    
    data_x = arrays['muon_data'][f'HcalFarForwardZDCHits{i}.position.x'][np.array(data_cell_cut)]
    data_y = arrays['muon_data'][f'HcalFarForwardZDCHits{i}.position.y'][np.array(data_cell_cut)]
    data_z = arrays['muon_data'][f'HcalFarForwardZDCHits{i}.position.z'][np.array(data_cell_cut)]    
    reco_data_angle = np.array([vector_angle_reconstruct(np.array(xi,dtype=float), np.array(yi,dtype=float), np.array(zi,dtype=float)) for xi, yi, zi in zip(data_x,data_y,data_z)])
    reco_data_angles.append(reco_data_angle)
    sim_energy = arrays['muon_sim'][f'HcalFarForwardZDCHits{i}.energy']*1000
    
    y,x=np.histogram(ak.flatten(sim_energy) ,bins=100, range=(0, 4))
    bc=(x[1:]+x[:-1])/2
    MIP=list(bc[y==max(y)])[0] 
    sim_MIP_cut = sim_energy > 0.5*MIP/1000
    sim_cell_cut = [True if len(cells)>=2 else False for cells in arrays['muon_sim'][f"HcalFarForwardZDCHits{i}.energy"][sim_MIP_cut]]
    
    sim_x = arrays['muon_sim'][f'HcalFarForwardZDCHits{i}.position.x'][np.array(sim_cell_cut)]
    sim_y = arrays['muon_sim'][f'HcalFarForwardZDCHits{i}.position.y'][np.array(sim_cell_cut)]
    sim_z = arrays['muon_sim'][f'HcalFarForwardZDCHits{i}.position.z'][np.array(sim_cell_cut)]    
    reco_sim_angle = np.array([vector_angle_reconstruct(np.array(xi,dtype=float), np.array(yi,dtype=float), np.array(zi,dtype=float)) for xi, yi, zi in zip(sim_x,sim_y,sim_z)])
    reco_sim_angles.append(reco_sim_angle)

    mc_px = arrays['muon_data']["MCParticles.momentum.x"][status==1][np.array(data_cell_cut)]
    mc_py = arrays['muon_data']["MCParticles.momentum.y"][status==1][np.array(data_cell_cut)]
    mc_pz = arrays['muon_data']["MCParticles.momentum.z"][status==1][np.array(data_cell_cut)]
    mc_mass = arrays['muon_data']["MCParticles.mass"][status==1][np.array(data_cell_cut)]
    mc_energy = np.sqrt(mc_px**2 + mc_py **2 + mc_pz**2 + mc_mass**2)
    energies.append(mc_energy)

def project_sphere_outline(ax, det_pos, target_pos, radius, n_points=200, **kwargs):
    xd, yd, zd = det_pos
    xt, yt, zt = target_pos
    
    # Parametrize sphere outline
    angles = np.linspace(0, 2*np.pi, n_points)
    circle_points = []
    for ang in angles:
        # Great circle in x-y plane relative to target
        x = xt + radius * np.cos(ang)
        y = yt + radius * np.sin(ang)
        z = zt
        vx, vy, vz = x - xd, y - yd, z - zd
        r = np.linalg.norm([vx, vy, vz])
        phi = np.mod(np.degrees(np.arctan2(vy, vx))+180,360)
        theta = np.degrees(np.arccos(vz / r))
        circle_points.append((phi, theta))
    
    circle_points = np.array(circle_points)
    ax.plot(circle_points[:,0], circle_points[:,1], **kwargs)

fig, ax = plt.subplots(3,num,figsize=(9*num,20),sharex=True,sharey=True)
fig.subplots_adjust(wspace=0.05,hspace=0.15)
plt.suptitle('World / Material: Vacuum/Rock ; Vacuum, Density: 2.6 / 0 g/cm3, Detector: Square 2x4x8 offset rotated, Target Position: 50,0,-20 m, Target Radius: 10 m',fontsize=60,y=1)
H_list = []
im_handles = [None, None, None]
theta_bin = np.logspace(0.69897,1.7708520116421442, 20)
phi_bin = np.linspace(0, 360, 10)

for i in range(num):
    det_pos = (positions_x[i], positions_y[i], positions_z[i])
    target_pos = (50, 0, 0)
    reco_data_angle = reco_data_angles[i]
    reco_sim_angle = reco_sim_angles[i]
    T, xedges, yedges = np.histogram2d(np.mod(reco_sim_angle[:,1], 360),reco_sim_angle[:,0],bins=[phi_bin,theta_bin], range=((0, 360),(10,90)))

    H, xedges, yedges = np.histogram2d(np.mod(reco_data_angle[:,1], 360),reco_data_angle[:,0],bins=[phi_bin,theta_bin], range=((0, 360),(10,90)))
    H_list.append(H)
    plt.sca(ax[0,i%num])

    #plt.gca().add_patch(rect)
    im1 = plt.imshow(1/np.sqrt(H.T)*100, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='vidiris', aspect='auto')
    im_handles[0] = im1
    #plt.colorbar(label='Counts')
    #plt.ylabel("Theta")
    plt.ylim(top=63)
    #plt.xlabel("Phi")
    plt.title(f'Detector {i}; {positions_x[i]},{positions_y[i]},{positions_z[i]} m')
    #plt.legend()
    
    # Plot the Z-map
    plt.sca(ax[1, i%num])


    #plt.gca().add_patch(rect)
    Z = poisson_to_Z(H, T)
    im2 = plt.imshow(Z.T, origin='lower',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='vidiris', aspect='auto', vmin=-3, vmax=3)
    im_handles[1] = im2
    #plt.colorbar(label='Significance Z')
    plt.ylim(top=63)
    #plt.ylabel("Theta")
    #plt.xlabel("Phi")
    plt.title(f'Tar/Free Tracks: {len(reco_data_angle)}/{len(reco_sim_angle)}')
    #plt.legend()
    
    plt.sca(ax[2,i%num])

    sim_H, sim_xedges, sim_yedges = np.histogram2d(
        np.mod(reco_sim_angle[:,1], 360), reco_sim_angle[:,0], bins=[phi_bin,theta_bin], range=((0, 360),(10,90))
    )
    
    # Data histogram (note: should use reco_data_angle for both axes, not mix with reco_sim_angle[:,0]?)
    data_H, _, _ = np.histogram2d(
        np.mod(reco_data_angle[:,1], 360), reco_data_angle[:,0], bins=[phi_bin,theta_bin], range=((0, 360),(10,90))
    )
    
    # Avoid divide-by-zero with masking
    #ratio = np.divide(data_H, sim_H, out=np.zeros_like(data_H, dtype=float), where=sim_H>0)
    ratio = np.divide(data_H, sim_H)
    # Plot the ratio
    im3 = plt.imshow(
        ratio.T,                      # transpose so x=first axis, y=second
        origin="lower", 
        extent=[sim_xedges[0], sim_xedges[-1], sim_yedges[0], sim_yedges[-1]],
        aspect="auto",
        cmap="vidiris",
        vmax=2
    )
    im_handles[2] = im3
    #plt.colorbar(label=r'Transmission')
    plt.ylim(top=63)
    #plt.ylabel("Theta")
    #plt.xlabel("Phi")
   # your sphere center

    #plt.legend()


ax[0, 0].set_ylabel('Theta (degrees)', fontsize=35, labelpad=20)
ax[-1, -1].set_xlabel('Phi (degrees)', fontsize=35, labelpad=20)
H_min = min(h.min() for h in H_list)
H_max = max(h.max() for h in H_list)
for row, im in enumerate(im_handles):
    fig.colorbar(im, ax=ax[row,:], orientation='vertical', label=['Error','Significance Z','Transmission'][row], pad=0.01)

plt.savefig('output/nine_detecotr.pdf',format='png')


fig, ax = plt.subplots(1,num,figsize=(7*num,5))
fig.subplots_adjust(wspace=0.4,hspace=0.4)
#plt.suptitle('World / Material: Vacuum / Steel235, Density: 0 / 7.85 g/cm3, Detector: Square 4x16 offset, Target Position: 50,0,-20 m, Target Radius: 20 m',fontsize=60,y=1)
for i in range(num):

    plt.sca(ax[0,i%num])
    plt.hist(energies[i],bins=int(np.sqrt(len(energies[i]))))

    plt.xlabel('Energy (GeV)')