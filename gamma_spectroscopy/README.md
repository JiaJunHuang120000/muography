### Analysis for gamma spectroscopy from CAEN DT5202 output
To use, simply run

``` python3 resolve.py ```

The output by running this command will point you to the data files that you need at the back of this command line. Please follow the format and give the file in the same order.

The files are obtain from running a pedestal run with PTRG, a run with radioactive source preferabling Uranitie for calibration (might be need to, but U gives you the most peaks), and a run for analyzing. All of the runs should have the same parameters such as HG, shaper time, etc.. while changing only the triggering logic. Minor adjustment might be needed from error due to fitting values may not agree. 

The calibration requires some observation and adjustment to ensure correct calibration. Look into `/output/uraninite.png/`, you should see the calibration spectrum with some red vertical lines. Those red vertical lines represent the fitting gaussian of the peaks from the spectrum. If there are no lines or too much, change the parameters inside the code, line 150

```
    peaks_lg = find_peaks_and_sigmas(centers[centers>50], bins[centers>50], distance=20, prominence=15, window=10)
``` 

such that you see only a few lines at the peaks you know the values for. After you will need to change the energy and peak locations in line 162-164 in order to calibration correctly.

```
    energy = np.array([608.17, 1116.21, 1374.5])
    peaks = np.array([peaks_lg[2][0], peaks_lg[3][0], peaks_lg[4][0]])
    errors = np.array([peaks_lg[2][1], peaks_lg[3][1], peaks_lg[4][1]])
```


## Visualizing the plots

All of the plots are in `output/`, following the order of; pedestal, calibration, HG/LG ratio, calibrated spectrum, calibration analysis.

