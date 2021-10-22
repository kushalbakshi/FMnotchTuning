from pypl2 import pl2_ad, pl2_events
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os


sr = 40000
b, a = butter(2, 500, btype='highpass', output='ba', fs=sr)


def plot_muanotch_raster(filelocation, bat, site, channel, poststim_window, savelocation):
    poststim_window *= sr

    data = pl2_ad(filelocation+bat+'/Plexon/{0}_FMnotch.pl2'.format(site), channel=channel)
    data = np.asarray(data.ad)
    data = filtfilt(b, a, data)
    thr = 4 * np.std(data)
    spikes, _ = find_peaks(data*-1, height=thr)

    _, ts, _ = pl2_events(filelocation+bat+'/Plexon/{0}_FMnotch.pl2'.format(site), channel='EVT01')
    ts = np.asarray(ts) * sr

    marker = np.asarray(loadmat(filelocation+bat+'/Data/{0}_{1}_marker_fm.mat'.format(bat, site)).get('marker_fm'))

    notch_freq = np.unique(marker)
    trial_num = len(marker)/len(notch_freq)

    raster_mat = np.zeros((len(marker), int(poststim_window)))
    for stim in range(0, len(ts)):
        stim_loc = (spikes[np.where((spikes > ts[stim]) & (spikes < ts[stim]+poststim_window))] - ts[stim]).astype('int')
        if np.mean(stim_loc) > 0:
            raster_mat[stim, stim_loc] = 1

    raster_mat = np.append(raster_mat, marker, axis=1)
    raster_mat = raster_mat[raster_mat[:, -1].argsort()]

    fig1, axs = plt.subplots(5, 2, figsize=(7, 10), sharey=True, sharex=True)
    params = {'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.labelsize': 14, 'figure.titlesize': 18,
              'figure.titleweight': 'bold', 'lines.linewidth': 3}
    plt.rcParams.update(params)

    iter = 0
    for y in range(0, 2):
        for x in range(0, 5):
            for n in range(0, int(trial_num)):
                axs[x, y].eventplot(np.asarray(np.where(raster_mat[int(n + (trial_num * iter)), :-2] == 1)) / sr * 1000,
                                    lineoffsets=0 + n, linelengths=0.4, colors='black')

            axs[x, y].set_title('{0} kHz notch'.format(notch_freq[iter]))
            iter += 1

    for ax in axs.flat:
        ax.set(xlabel='Time (ms)', ylabel='Trial #')
    for ax in axs.flat:
        ax.label_outer()

    fig1.suptitle('FM Notch Rasters')
    plt.tight_layout()

    if not os.path.exists(savelocation+bat+'/'):
        os.makedirs(savelocation+bat+'/')

    plt.savefig(savelocation+bat+'/Site_{0}_Chn{1}_FMnotch.jpg'.format(site, channel+1), dpi=300)
    plt.close()
