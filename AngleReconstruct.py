import numpy as np, uproot as ur, awkward as ak, pandas as pd
import matplotlib.pyplot as plt
import os, sys
import pickle
import scipy.stats as st
from collections import defaultdict
from scipy.optimize import curve_fit

#import mplhep as hep
#plt.figure()
#hep.style.use("CMS")
#plt.close()

def gaus(x, amp, mean, sigma):
    return amp * np.exp( -(x - mean)**2 / (2*sigma**2) ) 
    
def theta_reconstruct(x, y, z):
    return np.degrees(np.arccos(abs(z)/np.sqrt(x**2+y**2+z**2)))

def phi_reconstruct(x, y, z):
    return np.degrees(np.arctan2(y, x))

def vector_angle_reconstruct(x, y, z, n_points=10, scale=1.0):
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
    
def get_xyz_board_mapping_fast(df, dg):
    channels = dg.columns
    board = np.array([x[0] if isinstance(x, list) and len(x) > 0 else np.nan 
                     for x in dg.loc['Board']])
    x = np.array([x[0] if isinstance(x, list) and len(x) > 0 else np.nan 
                for x in dg.loc['X']])
    y = np.array([x[0] if isinstance(x, list) and len(x) > 0 else np.nan 
                for x in dg.loc['Y']])
    z = np.array([x[0] if isinstance(x, list) and len(x) > 0 else np.nan 
                for x in dg.loc['Z']])
    
    xyz_board = np.column_stack((board, x, y, z))
    
    channel_to_idx = {ch: i for i, ch in enumerate(channels)}
    
    rows, cols = np.where(df)
    true_channels = df.columns[cols]
    
    row_dict = defaultdict(list)
    for row, ch in zip(rows, true_channels):
        if ch in channel_to_idx:
            row_dict[row].append(channel_to_idx[ch])
    
    results = []
    for i in range(len(df)):
        if i in row_dict:
            idxs = row_dict[i]
            results.append(xyz_board[idxs])
        else:
            results.append(np.empty((0, 4)))
    
    return results

def data_process(datafile, posfile, simfile="cosmic_muons.root", data = True, sim = False, output = "output/"):

    if sim:
        os.makedirs(output, exist_ok=True)
        arrays={}
        arrays['muon']=ur.open("data/"+simfile+":events").arrays()
        status = arrays['muon']["MCParticles.generatorStatus"]
        mc_px = arrays['muon']["MCParticles.momentum.x"][status==1]
        mc_py = arrays['muon']["MCParticles.momentum.y"][status==1]
        mc_pz = arrays['muon']["MCParticles.momentum.z"][status==1]
        mc_theta = theta_reconstruct(mc_px,mc_py,mc_pz)
        mc_phi = phi_reconstruct(mc_px,mc_py,mc_pz)

        bins, edges = np.histogram(ak.flatten(mc_theta),bins=20)
        centers = edges[1:]/2 + edges[:-1]/2
        plt.errorbar(centers,bins,yerr=np.sqrt(bins),fmt='-o',label=r"Theta")

        bins, edges = np.histogram(ak.flatten(mc_phi),bins=20)
        centers = edges[1:]/2 + edges[:-1]/2
        plt.errorbar(centers,bins,yerr=np.sqrt(bins),fmt='-o',label=r"Phi")
        
        plt.xlabel("Angle (Degrees)")
        plt.ylabel("Normalized Counts")
        plt.legend()
        plt.title("CRY Truth Level Muon Paricle")
        plt.savefig(output+'Truth Level Muon Angle.png',format='png')
        plt.close()

        sim_energy = arrays['muon']['HcalFarForwardZDCHits.energy']*1000
        
        y,x,_=plt.hist(ak.flatten(sim_energy),histtype='step', bins=100, range=(0, 2), density=True, label="")
        bc=(x[1:]+x[:-1])/2
        MIP=list(bc[y==max(y)])[0]
        plt.axvline(MIP)        
        plt.ylabel("Normalized Counts")
        plt.xlabel("Hit energy [MeV]")
        plt.title("Muon Energy Deposition per Cell")
        plt.savefig(output+'MIP per Cell.png', format='png')
        plt.close()     

        sim_MIP_cut = sim_energy > 0.5*MIP/1000
        sim_cell_cut = [True if len(cells)>=2 else False for cells in arrays['muon']["HcalFarForwardZDCHits.energy"][sim_MIP_cut]]
        
        sim_x = arrays['muon']["HcalFarForwardZDCHits.position.x"][np.array(sim_cell_cut)]+109.80000305175781
        sim_y = -1*(arrays['muon']["HcalFarForwardZDCHits.position.y"][np.array(sim_cell_cut)]+109.80000305175781)
        sim_z = arrays['muon']["HcalFarForwardZDCHits.position.z"][np.array(sim_cell_cut)]+317.2439880371094       

        # Temp solution to dead channels
        xx = []
        yy = []
        zz = []
        
        for x_sublist, y_sublist, z_sublist in zip(sim_x, sim_y, sim_z):
            xxx = []
            yyy = []
            zzz = []
            for x_val, y_val, z_val in zip(x_sublist, y_sublist, z_sublist):
                if abs(z_val)-310 > 1:
                    xxx.append(x_val)
                    yyy.append(y_val)
                    zzz.append(z_val)
                elif y_val < 0:
                    xxx.append(x_val)
                    yyy.append(y_val)
                    zzz.append(z_val)
                elif y_val < 60 and x_val < 50: 
                    xxx.append(x_val)
                    yyy.append(y_val)
                    zzz.append(z_val)
                     
        
            xx.append(xxx)
            yy.append(yyy)
            zz.append(zzz)
        sim_cell_cut_2 = [True if len(cells)>=2 else False for cells in xx]

        xx = [cells for cells in xx if len(cells) >= 2]
        yy = [cells for cells in yy if len(cells) >= 2]
        zz = [cells for cells in zz if len(cells) >= 2]
        #####
        
        sim_angle = np.array([vector_angle_reconstruct(np.array(xi,dtype=float), np.array(yi,dtype=float), np.array(zi,dtype=float)) for xi, yi, zi in zip(xx,yy,zz)])

        counts, bins = np.histogram(sim_angle[:,0]-ak.flatten(mc_theta[np.array(sim_cell_cut)][np.array(sim_cell_cut_2)]), bins = 100)
        bin_center = bins[1:]/2+bins[:-1]/2
        coeff, covar = curve_fit(gaus,np.array(bin_center),np.array(counts))
        plt.errorbar(bin_center,counts,fmt='-o',label=rf'$\Delta$ Theta; $\sigma$ = {coeff[2]:.2f}$\degree$', c='b')
        plt.errorbar(np.linspace(-20,20,100),gaus(np.linspace(-20,20,100),*coeff),fmt='--', c='b')
        
        counts, bins = np.histogram(sim_angle[:,1]-ak.flatten(mc_phi[np.array(sim_cell_cut)][np.array(sim_cell_cut_2)]), bins = 100)
        bin_center = bins[1:]/2+bins[:-1]/2
        coeff, covar = curve_fit(gaus,np.array(bin_center),np.array(counts))
        plt.errorbar(bin_center,counts,fmt='-o',label=rf'$\Delta$ Phi; $\sigma$ = {coeff[2]:.2f}$\degree$', c='r')
        plt.errorbar(np.linspace(-20,20,100),gaus(np.linspace(-20,20,100),*coeff),fmt='--', c='r')
        
        plt.xlabel(r'Sim Reco - Sim Truth(degrees)')
        plt.ylabel('Counts')
        plt.title('CRY Cosmic Muon, Hits per Events: >2, Cut: 0.5*MIP')
        #plt.xlim(-50,50)
        plt.legend()
        plt.savefig(output+"Simulation Angular Resolution.png", format='png')
        plt.close()

        counts, bins = np.histogram(sim_angle[:,0], bins = 100)
        bin_center = bins[1:]/2+bins[:-1]/2
        #coeff, covar = curve_fit(gaus,np.array(bin_center),np.array(counts))
        plt.errorbar(bin_center,counts,fmt='-o',label=rf'Theta', c='b')
        #plt.errorbar(np.linspace(-20,20,100),gaus(np.linspace(-20,20,100),*coeff),fmt='--', c='b')
        
        counts, bins = np.histogram(sim_angle[:,1], bins = 100)
        bin_center = bins[1:]/2+bins[:-1]/2
        #coeff, covar = curve_fit(gaus,np.array(bin_center),np.array(counts))
        plt.errorbar(bin_center,counts,fmt='-o',label=rf'Phi', c='r')
        #plt.errorbar(np.linspace(-20,20,100),gaus(np.linspace(-20,20,100),*coeff),fmt='--', c='r')
        
        plt.xlabel(r'Sim Reco (degrees)')
        plt.ylabel('Counts')
        plt.title('CRY Cosmic Muon, Hits per Events: >2, Cut: 0.5*MIP')
        #plt.xlim(-50,50)
        plt.legend()
        plt.savefig(output+"Simulation Angular Reconstruct.png", format='png')
        plt.close()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        detector_resolution = 10.0 * np.pi / 180  # degrees
        
        bandwidth = detector_resolution * 1.0  # Adjust between 0.5-1.0
        
        grid_step = detector_resolution / 2 
        
        values = np.vstack([sim_angle[:,1], sim_angle[:,0]])
        kernel = st.gaussian_kde(values)
        kernel.set_bandwidth(bw_method=bandwidth)
        
        phi_grid = np.linspace(-180, 180, 180)  
        theta_grid = np.linspace(0, 60, 100)    
        phi_rad = np.deg2rad(phi_grid)       
        
        phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)
        positions = np.vstack([phi_mesh.ravel(), theta_mesh.ravel()])
        
        density = np.reshape(kernel(positions).T, phi_mesh.shape)
        
        total_events = len(sim_angle)
        bin_area = (phi_grid[1]-phi_grid[0]) * (theta_grid[1]-theta_grid[0])
        density_counts = density * total_events * bin_area
        
        plot_data = density_counts / np.max(density_counts)
        label = "Normalized Counts"
        
        pc = ax.pcolormesh(
            np.deg2rad(phi_mesh),
            theta_mesh,           
            plot_data,      
            cmap='inferno',     
            shading='auto',
            alpha=0.9           
        )
        
        levels = np.linspace(plot_data.min(), plot_data.max(), 5)
        ax.contour(
            np.deg2rad(phi_mesh),
            theta_mesh,
            plot_data,
            levels=levels,
            colors='white',
            linewidths=0.5,
            alpha=0.7
        )
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_theta_zero_location('S')  # 0° at the top (North)
        ax.set_theta_direction(-1)     
        
        ax.set_title(f"Simulation Reco KDE: # Events {len(sim_angle)}", pad=20)
        cbar = plt.colorbar(pc, ax=ax, pad=0.1)
        cbar.set_label(label)
        
        ax.set_rgrids([10, 20, 30, 40, 50, 60], angle=45, color='lime')
        ax.set_ylim(0, 60)
        
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(['N', 'E', 'S', 'W'])
        
        plt.tight_layout()
        plt.savefig(output+f"Simulation Reconstructed KDE.png", format='png')
        plt.close()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        detector_resolution = 10.0 * np.pi / 180  # degrees
        
        bandwidth = detector_resolution * 1.0  # Adjust between 0.5-1.0
        
        grid_step = detector_resolution / 2 
        
        values = np.vstack([ak.flatten(mc_phi[np.array(sim_cell_cut)][np.array(sim_cell_cut_2)]), ak.flatten(mc_theta[np.array(sim_cell_cut)][np.array(sim_cell_cut_2)])])
        kernel = st.gaussian_kde(values)
        kernel.set_bandwidth(bw_method=bandwidth)
        
        phi_grid = np.linspace(-180, 180, 180)  
        theta_grid = np.linspace(0, 60, 100)    
        phi_rad = np.deg2rad(phi_grid)       
        
        phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)
        positions = np.vstack([phi_mesh.ravel(), theta_mesh.ravel()])
        
        density = np.reshape(kernel(positions).T, phi_mesh.shape)
        
        total_events = len(sim_angle)
        bin_area = (phi_grid[1]-phi_grid[0]) * (theta_grid[1]-theta_grid[0])
        density_counts = density * total_events * bin_area
        
        plot_data = density_counts / np.max(density_counts)
        label = "Normalized Counts"
        
        pc = ax.pcolormesh(
            np.deg2rad(phi_mesh),
            theta_mesh,           
            plot_data,      
            cmap='inferno',     
            shading='auto',
            alpha=0.9           
        )
        
        levels = np.linspace(plot_data.min(), plot_data.max(), 5)
        ax.contour(
            np.deg2rad(phi_mesh),
            theta_mesh,
            plot_data,
            levels=levels,
            colors='white',
            linewidths=0.5,
            alpha=0.7
        )
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_theta_zero_location('S')  # 0° at the top (North)
        ax.set_theta_direction(-1)     
        
        ax.set_title(f"Simulation Truth KDE: # Events {len(sim_angle)}", pad=20)
        cbar = plt.colorbar(pc, ax=ax, pad=0.1)
        cbar.set_label(label)
        
        ax.set_rgrids([10, 20, 30, 40, 50, 60], angle=45, color='lime')
        ax.set_ylim(0, 60)
        
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(['N', 'E', 'S', 'W'])
        
        plt.tight_layout()
        plt.savefig(output+f"Simulation Truth KDE.png", format='png')
        plt.close()
        
        print(f'Done! Number of simulation events reconstructed: {len(sim_angle)}')

    if data:
        with open('data/'+datafile) as f:
            lines = f.read().split('  0  00')[1:]
    
        df = {}
        for i in range(64):
            df[str(f'{i:02}')] = []
    
        keys = list(df.keys())
        for i in range(len(lines)):
            event = '  0  00' + lines[i]
            channels = [row.split() for row in event.split('\n')][:64]
            channel = [row[1] for row in channels]
            LG = [row[2] for row in channels]
            HG = [int(row[3]) for row in channels]
            for ch, gain in zip(channel,HG):
                df[ch].append(gain)
    
        fig, ax = plt.subplots(8,8,figsize=(30,30),sharex=True,sharey=True)
        plt.subplots_adjust(wspace=0,hspace=0)
        cuts = []
        for i in range(64):
            plt.sca(ax[i//8,i%8])
            bins, edges = np.histogram(np.array(df[keys[i]],dtype=int),bins=200)
            centers = edges[1:]/2+edges[:-1]/2
            plt.errorbar(centers,bins,fmt='-o',label=f'Channel {keys[i]} HG')
            baseline = 10*centers[np.argmax(bins)]
            plt.axvline(baseline)
            plt.yscale('log')
            cuts.append(baseline)
            plt.gca().set_ylim(bottom=1)
            plt.legend()
        ax[0, 0].set_ylabel('Counts', fontsize=35, labelpad=20)
        ax[-1, -1].set_xlabel('ADC Channels', fontsize=35, labelpad=20)
        plt.savefig(output+f"{datafile} Total Energy Deposition.png", format='png')
        plt.close()

        dd = pd.DataFrame(df)>cuts
        
        with open('data/'+posfile, 'rb') as h:
            dg =  pd.DataFrame(pd.DataFrame(pickle.load(h)))

        xyz_board_list = get_xyz_board_mapping_fast(dd, dg)

        data_angle = []
        for i in range(len(xyz_board_list)):
            xx = xyz_board_list[i][:,1]
            yy = xyz_board_list[i][:,2]
            zz = xyz_board_list[i][:,3]
            if np.isnan(xx).any() == False:
                data_angle.append(vector_angle_reconstruct(xx, yy, zz))
            
        counts, bins = np.histogram(np.array(data_angle)[:,0],bins=np.linspace(-180,180,360))
        bin_center = bins[1:]/2+bins[:-1]/2
        plt.errorbar(bin_center,counts,yerr=np.sqrt(counts),fmt='-o', label='theta')
        
        counts, bins = np.histogram(np.array(data_angle)[:,1],bins=np.linspace(-180,180,360))
        bin_center = bins[1:]/2+bins[:-1]/2
        plt.errorbar(bin_center,counts,yerr=np.sqrt(counts),fmt='-o', label='phi') 
        
         
        #plt.xlim(0,90)
        #plt.hist(theta_reco)
        plt.xlabel("Reconstructed Theta (Degrees)")
        plt.ylabel("Counts")
        plt.title(f"Data Reco: {datafile}")
        plt.legend()
        plt.savefig(output+f'{datafile} Angle Reconstruct.png',format='png')
        plt.close()
    
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        detector_resolution = 7.5 * np.pi / 180  # degrees
        
        bandwidth = detector_resolution * 1.0  # Adjust between 0.5-1.0
        
        grid_step = detector_resolution / 2 
        
        values = np.vstack([np.array(data_angle)[:,1][np.array(data_angle)[:,0]>5][:19665], np.array(data_angle)[:,0][np.array(data_angle)[:,0]>5][:19665]])
        kernel = st.gaussian_kde(values)
        kernel.set_bandwidth(bw_method=bandwidth)
        
        phi_grid = np.linspace(-180, 180, 180)  
        theta_grid = np.linspace(0, 60, 100)    
        phi_rad = np.deg2rad(phi_grid)       
        
        phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)
        positions = np.vstack([phi_mesh.ravel(), theta_mesh.ravel()])
        
        density = np.reshape(kernel(positions).T, phi_mesh.shape)
        
        total_events = len(data_angle)
        bin_area = (phi_grid[1]-phi_grid[0]) * (theta_grid[1]-theta_grid[0])
        density_counts = density * total_events * bin_area
        
        plot_data = density_counts / np.max(density_counts)
        label = "Normalized Counts"
        
        pc = ax.pcolormesh(
            np.deg2rad(phi_mesh),
            theta_mesh,           
            plot_data,      
            cmap='inferno',     
            shading='auto',
            alpha=0.9           
        )
        
        levels = np.linspace(plot_data.min(), plot_data.max(), 5)
        ax.contour(
            np.deg2rad(phi_mesh),
            theta_mesh,
            plot_data,
            levels=levels,
            colors='white',
            linewidths=0.5,
            alpha=0.7
        )
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_theta_zero_location('S')  # 0° at the top (North)
        ax.set_theta_direction(-1)     
        
        ax.set_title(f"Data Reco KDE: # Events {len(data_angle[:19665])}", pad=20)
        cbar = plt.colorbar(pc, ax=ax, pad=0.1)
        cbar.set_label(label)
        
        ax.set_rgrids([10, 20, 30, 40, 50, 60], angle=45, color='lime')
        ax.set_ylim(0, 60)
        
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(['N', 'E', 'S', 'W'])
        
        plt.tight_layout()
        plt.savefig(output+f"{datafile} Reconstructed KDE.png", format='png')
        plt.close()

        da = {
                'Theta': np.array(data_angle)[:,0],
                'Phi': np.array(data_angle)[:,1],
                }
        with open(f'data/{datafile}_angles.pkl', 'wb') as h:
            pickle.dump(da,h,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Done! Number of data events reconstructed: {len(data_angle)}')
 
data_process(sys.argv[1],sys.argv[2],data=False,simfile='metal.root',sim=True)


