import numpy as np, uproot as ur, awkward as ak, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import pickle
import os
from scipy.signal import find_peaks
import sys

def find_peaks_and_sigmas(x, y, distance=10, prominence=0.1, window=10):
    # Step 1: Detect peaks
    peaks, _ = find_peaks(y, distance=distance, prominence=prominence)
    results = []

    for peak_idx in peaks:
        # Step 2: Define a window around the peak
        left = max(0, peak_idx - window)
        right = min(len(x), peak_idx + window)
        x_fit = x[left:right]
        y_fit = y[left:right]

        # Step 3: Fit a Gaussian to the windowed data
        try:
            a_guess = y[peak_idx]
            x0_guess = x[peak_idx]
            sigma_guess = (x[right-1] - x[left]) / 6  # Approximate sigma
            popt, _ = curve_fit(gaus, x_fit, y_fit, p0=[a_guess, x0_guess, sigma_guess])
            a, x0, sigma = popt
            results.append([x0, abs(sigma), a])
        except RuntimeError:
            continue  # Skip if the fit fails

    return results

import mplhep as hep
plt.figure()
hep.style.use("CMS")
plt.close()

import uproot,glob

def gaus(x, amp, mean, sigma):
    return amp * np.exp( -(x - mean)**2 / (2*sigma**2) ) 
def linear(x, m, b):
    return m * x + b

def plot(infoFile, pedFile, UFile, dataFile, source_info, source, data=True, CH='00', detector='CsI', number = 9):
    arrays = {}
    source = str(source)
    with open(infoFile,'r') as f:
        lines = f.read()
        high_gain = lines.split('HG_Gain')[1].split()[0]
        low_gain = lines.split('LG_Gain')[1].split()[0]
        threshold = lines.split('TD_CoarseThreshold')[1].split()[0]
        shaper_time = lines.split('HG_ShapingTime')[1].split()[0]
        hold_delay = lines.split('HoldDelay ')[1].split()[0]
    title = f'{detector}: LG = {low_gain}, HG = {high_gain}, Threshold = {threshold}'+'\n'+f'Hold Delay = {hold_delay} ns, Shaper Time = {shaper_time} ns'
    
    with open(pedFile) as f:
        lines = f.read().split('  0  00')[1:]
    df = {}
    for i in range(64):
        df[str(f'{i:02}')] = []
    dg = {}
    for i in range(64):
        dg[str(f'{i:02}')] = []
    for i in range(len(lines)):
        event = '  0  00' + lines[i]
        channels = [row.split() for row in event.split('\n')][:64]
        channel = [row[1] for row in channels]
        LG = [int(row[2]) for row in channels]
        HG = [int(row[3]) for row in channels]
        for ch, gain in zip(channel,LG):
            dg[ch].append(gain)
        for ch, gain in zip(channel,HG):
            df[ch].append(gain)

    arrays['ped_lg'] = dg
    arrays['ped_hg'] = df

    keys = sorted(arrays['ped_lg'].keys())
    
    ped = []
    sig = []
    
    fig, ax = plt.subplots(8,8,figsize=(50,50))
    
    for i in range(64):
        plt.sca(ax[i//8,i%8])
        
        bins, edges = np.histogram(np.array(arrays['ped_lg'][str(f'{i:02}')],dtype=int)[np.array(arrays['ped_lg'][str(f'{i:02}')],dtype=int)<500],bins=35)
        centers = edges[1:]/2+edges[:-1]/2
        plt.errorbar(centers,bins,fmt='-o',label=f'{detector} LG',yerr=np.sqrt(bins))
        
        con1 = 100>centers
        con2 = centers>20
        cut = con1 & con2

        try:
            coeff, covar = curve_fit(gaus,centers[cut],bins[cut],p0=[2500,50,10])
        except: continue
        x = np.linspace(0,200,100)
        plt.errorbar(x,gaus(x,*coeff),fmt='--',label=
                     f'LG peak = {coeff[1]:.2f}'+r' $\pm$ '+f'{abs(coeff[2]):.2f} ADC'+'\n')
        ped.append(coeff[1])
        sig.append(abs(coeff[2]))
        
        bins, edges = np.histogram(np.array(arrays['ped_hg'][str(f'{i:02}')],dtype=int)[np.array(arrays['ped_hg'][str(f'{i:02}')],dtype=int)<1000],bins=35)
        centers = edges[1:]/2+edges[:-1]/2
        plt.errorbar(centers,bins,fmt='-o',label=f'{detector} HG',yerr=np.sqrt(bins))
        
        con1 = 1000>centers
        con2 = centers>20
        cut = con1 & con2
        
        try: coeff, covar = curve_fit(gaus,centers[cut],bins[cut],p0=[2500,200,100])
        except: continue
        x = np.linspace(0,1000,100)
        plt.errorbar(x,gaus(x,*coeff),fmt='--',label=
                     f'HG peak = {coeff[1]:.2f}'+r' $\pm$ '+f'{abs(coeff[2]):.2f} ADC'+'\n')
        
        ped.append(coeff[1])
        sig.append(abs(coeff[2]))

        #plt.yscale('log')
        plt.legend()
        plt.ylabel('Counts')
        plt.xlabel('ADC Channels')
        plt.ylim(bottom=1)
        #plt.title('LG = 1, HG = 20,  Hold Delay = 300 ns, Shaper Time = 87.5 ns')
        plt.title(f'Channel: {i:02}')
    plt.savefig('output/'+'pedestal.png',format='png')
    plt.close()

    with open(UFile) as f:
        lines = f.read().split('  0  00')[1:]
    df = {}
    for i in range(64):
        df[str(f'{i:02}')] = []
    dg = {}
    for i in range(64):
        dg[str(f'{i:02}')] = []
    for i in range(len(lines)):
        event = '  0  00' + lines[i]
        channels = [row.split() for row in event.split('\n')][:64]
        channel = [row[1] for row in channels]
        LG = [int(row[2]) for row in channels]
        HG = [int(row[3]) for row in channels]
        for ch, gain in zip(channel,LG):
            dg[ch].append(gain)
        for ch, gain in zip(channel,HG):
            df[ch].append(gain)

    arrays['U_lg'] = dg
    arrays['U_hg'] = df

    lg = {k: np.where(v - s >= 3*t, v - s, 0) for k, v, s, t in zip(keys, arrays['U_lg'].values(), ped[::2], sig[::2])}
    lg_sum = {}
    for i, start in enumerate(range(0, 64, number)):
        group_keys = keys[start:start + number]
        lg_sum[f"{i:02d}"] = np.sum([lg[k] for k in group_keys], axis=0)
    
    bins, edges = np.histogram(lg_sum['00'][lg_sum['00']<20000],bins=500)
    centers = edges[1:]/2+edges[:-1]/2
    plt.errorbar(centers,bins,fmt='-o',label=f'uraninite LG',yerr=np.sqrt(bins))

    peaks_lg = find_peaks_and_sigmas(centers, bins, distance=30, prominence=10, window=10) # CsI: distance=20, prominence=15, window=10
    for r in peaks_lg:
        plt.axvline(r[0], color='r', linestyle='--')

    plt.yscale('log')
    plt.legend()
    plt.ylabel('Counts')
    plt.xlabel('ADC Channels')
    plt.ylim(bottom=1)
    #plt.title('LG = 1, HG = 20,  Hold Delay = 300 ns, Shaper Time = 87.5 ns')
    plt.title(f'Channel: {CH:02}')
    plt.savefig('output/'+'uraninite.png',format='png')
    plt.close()

    energy = np.array([608.17, 1116.21, 1374.5])
    peaks = np.array([peaks_lg[3][0], peaks_lg[5][0], peaks_lg[6][0]])
    errors = np.array([peaks_lg[3][1], peaks_lg[5][1], peaks_lg[6][1]])

    plt.errorbar(energy,peaks,yerr=errors,fmt='o',label='calibration LG',c='b')
    
    m, p = np.polyfit(energy, peaks, 1, w=1/errors)
    plt.plot([-200,3500],p+m*np.array([-200,3500]),linestyle='dashed',label=f'linear fit; m={m:.2f} keV/ADC, y0={p:.2f} ADC')
    plt.axhline(0,linestyle='dashed',c='black')
    plt.axvline(0,linestyle='dashed',c='black')
    plt.ylabel('ADC - ped')
    plt.xlabel('Energy (keV)')
    plt.legend()
    plt.title(title)
    plt.savefig('output/'+'calibration.png',format='png')
    plt.close()


    fig, ax = plt.subplots(8,8,figsize=(50,50))
    mp = []
    for i in range(64):
        plt.sca(ax[i//8,i%8])
        lg = np.array(arrays['U_lg'][str(f'{i:02}')],dtype=int)-ped[i*2]
        hg = np.array(arrays['U_hg'][str(f'{i:02}')],dtype=int)-ped[1+i*2]
        con1 = hg<7500
        con2 = lg<7500
        con3 = 3*sig[i*2]<lg
        con4 = 3*sig[1+i*2]<hg
        cond = con1 & con2 & con3 & con4
    
        plt.xlabel('LG - ped (ADC)')
        plt.ylabel('HG - ped (ADC)')
        plt.title(f'Channel: {i:02}')
        
        try:
            bin = plt.hist2d(lg[cond],hg[cond],bins=(100,100),cmin=1)
            
            poeff, covar = curve_fit(linear,lg[cond],hg[cond])
            plt.plot([0,1000],linear(np.array([0,1000]),*poeff),label=f'linear fit; m={poeff[0]:.2f}, y0={poeff[1]:.2f} ADC'+'\n'+'3 sigma cut',c='r')
            #plt.colorbar(label='Counts')
            plt.legend()
            mp.append(poeff)
        except: 
            mp.append(np.array([ 1, 1]))
            continue
    plt.savefig('output/ratio.png',format='png')
    plt.close()

    lg_ped = {}
    hg_ped = {}
    lg_ = {k: np.where(True, v, v) for k, v, s, t in zip(keys, arrays['ped_lg'].values(), ped[::2], sig[::2])}
    hg_ = {k: np.where(True, (v -pm[1])/pm[0], (v -pm[1])/pm[0]) for k, v, s, t, pm in zip(keys, arrays['ped_hg'].values(), ped[1::2], sig[1::2], mp)}
    for i, start in enumerate(range(0, 64, number)):
        group_keys = keys[start:start + number]
        lg_ped[f"{i:02d}"] = np.sum([lg_[k] for k in group_keys], axis=0)
        hg_ped[f"{i:02d}"] = np.sum([hg_[k] for k in group_keys], axis=0)

    bins, edges = np.histogram(lg_ped[CH],bins=35)
    centers = edges[1:]/2+edges[:-1]/2
    plt.errorbar(centers,bins,fmt='-o',label=f'Csl LG',yerr=np.sqrt(bins))
    

    coeff, covar = curve_fit(gaus,centers,bins,p0=[2500,500,200])
    x = np.linspace(-1000,1000,100)
    plt.errorbar(x,gaus(x,*coeff),fmt='--',label=
                 f'LG pedestal width = {abs(coeff[2]):.2f} ADC / {abs(coeff[2]/m):.2f} keV')
    
    bins, edges = np.histogram(hg_ped[CH],bins=35)
    centers = edges[1:]/2+edges[:-1]/2
    plt.errorbar(centers,bins,fmt='-o',label=f'Csl HG',yerr=np.sqrt(bins))
    

    coeff, covar = curve_fit(gaus,centers,bins,p0=[2500,-500,50])
    x = np.linspace(-1000,1000,100)
    plt.errorbar(x,gaus(x,*coeff),fmt='--',label=
                 f'HG pedestal width = {abs(coeff[2]):.2f} (LG) ADC / {abs(coeff[2]/m):.2f} keV')
    
    #plt.yscale('log')
    plt.legend()
    plt.ylabel('Counts')
    plt.xlabel('ADC Channels')
    plt.ylim(bottom=1)
    plt.title(title)
    plt.savefig('output/'+'pedestal_calibrated.png',format='png')
    plt.close()


    
    if data:
        with open(source_info,'r') as f:
            lines = f.read()
            time = float(lines.split('Elapsed time =')[1].split()[0])
            threshold = lines.split('TD_CoarseThreshold')[1].split()[0]
        title = f'{detector}: LG = {low_gain}, HG = {high_gain}, Threshold = {threshold}'+'\n'+f'Hold Delay = {hold_delay} ns, Shaper Time = {shaper_time} ns'

        with open(dataFile) as f:
            lines = f.read().split('  0  00')[1:]
        df = {}
        for i in range(64):
            df[str(f'{i:02}')] = []
        dg = {}
        for i in range(64):
            dg[str(f'{i:02}')] = []
        for i in range(len(lines)):
            event = '  0  00' + lines[i]
            channels = [row.split() for row in event.split('\n')][:64]
            channel = [row[1] for row in channels]
            LG = [int(row[2]) for row in channels]# - np.array(ped[::2])
            #LG = [sum(np.array(LG[i:i+number])[LG[i:i+number]>3*np.array(sig[::2][i:i+number])]) for i in range(0, len(LG), number)]
            HG = [int(row[3]) for row in channels]# - np.array(ped[1::2])
            #HG = [sum(np.array(HG[i:i+number])[HG[i:i+number]>3*np.array(sig[1::2][i:i+number])]) for i in range(0, len(HG), number)]
            #channel = [str(f'{i:02}') for i in range(number)]
            for ch, gain in zip(channel,LG):
                dg[ch].append(gain)
            for ch, gain in zip(channel,HG):
                df[ch].append(gain)

        arrays[source+'_lg'] = dg
        arrays[source+'_hg'] = df

        _lg = {k: np.where(v - s >= 3*t, v - s, 0) for k, v, s, t in zip(keys, arrays[source+'_lg'].values(), ped[::2], sig[::2])}
        _hg = {k: np.where(v - s >= 3*t, ((v - s)+pm[1])/pm[0], 0) for k, v, s, t, pm in zip(keys, arrays[source+'_hg'].values(), ped[1::2], sig[1::2], mp)}

        lg_sum = {}
        hg_sum = {}
        for i, start in enumerate(range(0, 64, number)):
            group_keys = keys[start:start + number]
            lg_sum[f"{i:02d}"] = np.sum([_lg[k] for k in group_keys], axis=0)
            hg_sum[f"{i:02d}"] = np.sum([_hg[k] for k in group_keys], axis=0)
        
        bins, edges = np.histogram(((lg_sum[CH]-p)/m),bins=400,range=(0,3200))
        centers = edges[1:]/2+edges[:-1]/2
        plt.errorbar(centers,bins,fmt='-o',label=f'LG, {source}, Rate = {len(lg_sum[CH])/time:.2f} Hz',yerr=np.sqrt(bins))
        plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('energy (keV)')
        plt.title(title)
        plt.legend()
        plt.savefig('output/'+source+'_LG_calibrated.png',format='png')
        plt.close()
    
        bins, edges = np.histogram((hg_sum[CH]-p)/m,bins=400)
        centers = edges[1:]/2+edges[:-1]/2
        plt.errorbar(centers,bins,fmt='-o',label=f'HG, {source}, Rate = {len(lg_sum[CH])/time:.2f} Hz',yerr=np.sqrt(bins))
        plt.yscale('log')
        #plt.xscale('log')
        plt.xlabel('energy (keV)')
        plt.title(title)
        plt.legend()
        plt.savefig('output/'+source+'_HG_calibrated.png',format='png')
        plt.close()


if len(sys.argv) != 7:
    print('Correct formating must include 4 files with 1 source name.')
    print('python3 recolve.py {infoFile} {pedFile} {UFile} {dataFile} {source name}')
    print('Where;\ninfoFile = path to uraninite *_info.txt\npedFile = path to pedestal *_list.txt file\nUfile = path to uraninite *_list.txt\ndataFile = path to source data *_list.txt file for visualization in calibrated energy level\nsource_infoFile = path to source *_info.txt file\nsource name = display name for the dataFile')
elif len(sys.argv) == 7:
    plot(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],data=True,detector='NaI',CH='00',number=16)