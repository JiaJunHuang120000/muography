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

def plot(infoFile, pedFile, UFile, dataFile, source_info, source, data=True, CH='01', detector='CsI'):
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
    
    ped = []
    sig = []
    
    lg_ = np.array(arrays['ped_lg'][CH])
    hg_ = np.array(arrays['ped_hg'][CH])
    bins, edges = np.histogram(lg_[lg_<2000],bins=35)
    centers = edges[1:]/2+edges[:-1]/2
    plt.errorbar(centers,bins,fmt='-o',label=f'ped LG',yerr=np.sqrt(bins))
    
    con1 = 100>centers
    con2 = centers>0
    cut = con1 & con2
    coeff, covar = curve_fit(gaus,centers[cut],bins[cut],p0=[2500,50,300])
    x = np.linspace(0,100,100)
    plt.errorbar(x,gaus(x,*coeff),fmt='--',label=
                 f'LG peak = {coeff[1]:.2f}'+r' $\pm$ '+f'{abs(coeff[2]):.2f} ADC'+'\n')
    ped.append(coeff[1])
    sig.append(abs(coeff[2]))
    
    bins, edges = np.histogram(hg_[hg_<2000],bins=35)
    centers = edges[1:]/2+edges[:-1]/2
    plt.errorbar(centers,bins,fmt='-o',label=f'ped HG',yerr=np.sqrt(bins))
    
    con1 = 300>centers
    con2 = centers>20
    cut = con1 & con2
    coeff, covar = curve_fit(gaus,centers[cut],bins[cut],p0=[2500,150,300])
    x = np.linspace(0,300,100)
    plt.errorbar(x,gaus(x,*coeff),fmt='--',label=
                 f'HG peak = {coeff[1]:.2f}'+r' $\pm$ '+f'{abs(coeff[2]):.2f} ADC'+'\n')
    
    ped.append(coeff[1])
    sig.append(abs(coeff[2]))
    
    plt.legend()
    plt.ylabel('Counts')
    plt.xlabel('ADC Channels')
    plt.ylim(bottom=1)
    plt.title(title)
    plt.savefig('output/'+'pedestal.png',format='png')
    plt.close()
    
    lg = np.array(arrays['U_lg'][CH]) - ped[0]
    hg = np.array(arrays['U_hg'][CH]) - ped[1]
    bins, edges = np.histogram(lg,bins=400)
    centers = edges[1:]/2+edges[:-1]/2
    peaks_lg = find_peaks_and_sigmas(centers[centers>50], bins[centers>50], distance=20, prominence=15, window=10) # CsI: distance=20, prominence=15, window=10
    plt.errorbar(centers,bins,fmt='-o',label=f'uraninite LG',yerr=np.sqrt(bins),c='r')
    for r in peaks_lg:
        plt.axvline(r[0], color='r', linestyle='--')
    #plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('ADC - ped')
    plt.legend()
    plt.title(title)
    plt.savefig('output/'+'uraninite.png',format='png')
    plt.close()

    energy = np.array([608.17, 1116.21, 1374.5])
    peaks = np.array([peaks_lg[2][0], peaks_lg[3][0], peaks_lg[4][0]])
    errors = np.array([peaks_lg[2][1], peaks_lg[3][1], peaks_lg[4][1]])

    plt.errorbar(energy,peaks,yerr=errors,fmt='o',label='calibration LG',c='b')
    
    m, p = np.polyfit(energy, peaks, 1, w=1/errors)
    plt.plot([-200,3500],p+m*np.array([-200,3500]),linestyle='dashed',label=f'linear fit; m={m:.2f} keV/ADC, y0={p:.2f} ADC')
    plt.axhline(8192,label=f"Max ADC, {(8192-p)/m:.2f} keV",c='r',linestyle='dashed')
    plt.ylabel('ADC - ped')
    plt.xlabel('Energy (keV)')
    #plt.title('LG = 55, Hold Delay = 100 ns, Shaper Time = 25 ns')
    plt.ylim(0,8500)
    plt.legend()#bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.savefig('output/'+'calibration.png',format='png')
    plt.close()


    con1 = hg<8000
    con2 = 5*sig[1]<hg
    con3 = lg<800
    con4 = 5*sig[0]<lg
    cond = con1 & con2 & con3 & con4
    bin = plt.hist2d(lg[cond],hg[cond],bins=(100,100),cmin=1)
    
    poeff, covar = curve_fit(linear,lg[cond],hg[cond])
    plt.plot([0,1000],linear(np.array([0,1000]),*poeff),label=f'linear fit; m={poeff[0]:.2f}, y0={poeff[1]:.2f} ADC'+'\n'+'5 sigma pedestal width cut',c='r')
    
    plt.xlabel('LG - ped (ADC)')
    plt.ylabel('HG - ped (ADC)')
    plt.colorbar(label='Counts')
    plt.title(title)
    plt.legend()#bbox_to_anchor=(1, 1))
    plt.savefig('output/'+'ratio.png',format='png')
    plt.close()

    bins, edges = np.histogram(lg_[lg_<2000],bins=35)
    centers = edges[1:]/2+edges[:-1]/2
    plt.errorbar(centers,bins,fmt='-o',label=f'ped LG',yerr=np.sqrt(bins))
    
    con1 = 500>centers
    con2 = centers>0
    cut = con1 & con2
    coeff, covar = curve_fit(gaus,centers[cut],bins[cut],p0=[2500,50,300])
    x = np.linspace(0,100,100)
    plt.errorbar(x,gaus(x,*coeff),fmt='--',label=
                 f'LG Width = {abs(coeff[2]):.2f} ADC / {abs(coeff[2])*m:.2f} keV')
    
    
    bins, edges = np.histogram(hg_[hg_<2000],bins=35)
    centers = edges[1:]/2+edges[:-1]/2
    plt.errorbar(centers,bins,fmt='-o',label=f'ped HG',yerr=np.sqrt(bins))
    
    con1 = 300>centers
    con2 = centers>20
    cut = con1 & con2
    coeff, covar = curve_fit(gaus,centers[cut],bins[cut],p0=[2500,150,300])
    x = np.linspace(0,300,100)
    plt.errorbar(x,gaus(x,*coeff),fmt='--',label=
                 f'HG Width = {abs(coeff[2]):.2f} ADC / {abs(coeff[2]-poeff[1])/poeff[0]*m:.2f} keV')
    
    #plt.yscale('log')
    plt.legend()
    plt.ylabel('Counts')
    plt.xlabel('ADC Channels')
    plt.title(title)
    plt.savefig('output/'+'pedestal_calibrated.png',format='png')
    plt.close()

    bins, edges = np.histogram((lg-p)/m,bins=400)
    centers = edges[1:]/2+edges[:-1]/2
    plt.errorbar(centers,bins,fmt='-o',label=f'LG, uraninite',yerr=np.sqrt(bins))
    plt.yscale('log')
    plt.xlabel('energy (keV)')
    plt.title(title)
    plt.legend()
    plt.savefig('output/uraninite_LG_calibrated.png',format='png')
    plt.close()

    bins, edges = np.histogram(((hg-poeff[1])/poeff[0]-p)/m,bins=400)
    centers = edges[1:]/2+edges[:-1]/2
    plt.errorbar(centers,bins,fmt='-o',label=f'HG, uraninite',yerr=np.sqrt(bins))
    plt.yscale('log')
    plt.xlabel('energy (keV)')
    plt.title(title)
    plt.legend()
    plt.savefig('output/uraninite_HG_calibrated.png',format='png')
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
            LG = [int(row[2]) for row in channels]
            HG = [int(row[3]) for row in channels]
            for ch, gain in zip(channel,LG):
                dg[ch].append(gain)
            for ch, gain in zip(channel,HG):
                df[ch].append(gain)
    
        arrays[source+'_lg'] = dg
        arrays[source+'_hg'] = df
        _lg = (np.array(arrays[source+'_lg'][CH]) - ped[0] - p) / m
        _hg = ((np.array(arrays[source+'_hg'][CH]) - ped[1] - poeff[1]) / poeff[0] - p) / m
        
        bins, edges = np.histogram(_lg,bins=400)
        centers = edges[1:]/2+edges[:-1]/2
        plt.errorbar(centers,bins,fmt='-o',label=f'LG, {source}, Rate = {len(_lg)/time:.2f} Hz',yerr=np.sqrt(bins))
        plt.yscale('log')
        plt.xlabel('energy (keV)')
        plt.title(title)
        plt.legend()
        plt.savefig('output/'+source+'_LG_calibrated.png',format='png')
        plt.close()
    
        bins, edges = np.histogram(_hg,bins=400)
        centers = edges[1:]/2+edges[:-1]/2
        plt.errorbar(centers,bins,fmt='-o',label=f'HG, {source}, Rate = {len(_lg)/time:.2f} Hz',yerr=np.sqrt(bins))
        plt.yscale('log')
        plt.xlabel('energy (keV)')
        plt.title(title)
        plt.legend()
        plt.savefig('output/'+source+'_HG_calibrated.png',format='png')
        plt.close()

        con1 = (np.array(arrays[source+'_hg'][CH]) - ped[1])<8000
        con2 = 5*sig[1]<(np.array(arrays[source+'_hg'][CH]) - ped[1])
        con3 = (np.array(arrays[source+'_lg'][CH]) - ped[0])<1000
        con4 = 5*sig[0]<(np.array(arrays[source+'_lg'][CH]) - ped[0])
        cond = con1 & con2 & con3 & con4
        bin = plt.hist2d((np.array(arrays[source+'_lg'][CH]) - ped[0])[cond],(np.array(arrays[source+'_hg'][CH]) - ped[1])[cond],bins=(100,100),cmin=1)
        
        poeff, covar = curve_fit(linear,(np.array(arrays[source+'_lg'][CH]) - ped[0])[cond],(np.array(arrays[source+'_hg'][CH]) - ped[1])[cond])
        plt.plot([0,1000],linear(np.array([0,1000]),*poeff),label=f'linear fit; m={poeff[0]:.2f}, y0={poeff[1]:.2f} ADC'+'\n'+'5 sigma pedestal width cut'+'\n'+source+f', Rate = {len(_lg)/time:.2f} Hz',c='r')
        
        plt.xlabel('LG - ped (ADC)')
        plt.ylabel('HG - ped (ADC)')
        plt.colorbar(label='Counts')
        plt.title(title)
        plt.legend()#bbox_to_anchor=(1, 1))
        #plt.xlim(0,8000)
        #plt.ylim(0,8000)
        plt.savefig('output/'+source+'_ratio.png',format='png')
        plt.close()


if len(sys.argv) != 7:
    print('Correct formating must include 4 files with 1 source name.')
    print('python3 recolve.py {infoFile} {pedFile} {UFile} {dataFile} {source name}')
    print('Where;\ninfoFile = path to uraninite *_info.txt\npedFile = path to pedestal *_list.txt file\nUfile = path to uraninite *_list.txt\ndataFile = path to source data *_list.txt file for visualization in calibrated energy level\nsource_infoFile = path to source *_info.txt file\nsource name = display name for the dataFile')
elif len(sys.argv) == 7:
    plot(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],data=True,detector='CsI',CH='01')