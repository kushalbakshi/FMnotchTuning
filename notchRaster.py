import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os


def plot_notch_raster(filelocation, bat, site, channel, poststim_window, savelocation):
    """
    Imports spike times from a csv/txt file and creates event raster plots.
    filelocation: absolute path of the spike times file.
    bat: Animal ID within filelocation subfolders
    site: Site in auditory cortex
    channel: Electrode channel being analyzed
    poststim_window: Length of post stimulus window to include in the raster plot
    savelocation: Absolute path of where the raster should be saved
    """
    sr = 40000
    poststim_window *= sr

    # Catches exception when some spike sorted files have a header and others do not
    try:
        spikes = np.loadtxt(filelocation+bat+'/Spiketimes/{0}_FMnotch_chn'.format(site)+str(channel)+'_times.txt',
                            delimiter=',')
    except ValueError:
        spikes = np.loadtxt(filelocation+bat+'/Spiketimes/{0}_FMnotch_chn'.format(site)+str(channel)+'_times.txt',
                            delimiter=',', skiprows=1)
    spikes = spikes[:, 0] * sr

    # Import stimulus timestamps
    ts = np.asarray(loadmat(filelocation+bat+'/Matfile/{0}_FMnotch/event.mat'.format(site)).get('ts'))
    ts *= sr

    # Import stimulus identity and other relevant information
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

    # Initiate and create figure
    fig1, axs = plt.subplots(5, 2, figsize=(7, 10), sharey=True, sharex=True)
    params = {'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.labelsize': 14, 'figure.titlesize': 18,
              'figure.titleweight': 'bold', 'lines.linewidth': 3}
    plt.rcParams.update(params)
    for n in range(0, int(trial_num)):
        axs[0, 0].eventplot(np.asarray(np.where(raster_mat[n, :-2] == 1)) / sr * 1000, lineoffsets=0 + n,
                            linelengths=0.5, colors='black')
        axs[1, 0].eventplot(np.asarray(np.where(raster_mat[n + 20, :-2] == 1)) / sr * 1000, lineoffsets=0 + n,
                            linelengths=0.5, colors='black')
        axs[2, 0].eventplot(np.asarray(np.where(raster_mat[n + 40, :-2] == 1)) / sr * 1000, lineoffsets=0 + n,
                            linelengths=0.5, colors='black')
        axs[3, 0].eventplot(np.asarray(np.where(raster_mat[n + 60, :-2] == 1)) / sr * 1000, lineoffsets=0 + n,
                            linelengths=0.5, colors='black')
        axs[4, 0].eventplot(np.asarray(np.where(raster_mat[n + 80, :-2] == 1)) / sr * 1000, lineoffsets=0 + n,
                            linelengths=0.5, colors='black')
        axs[0, 1].eventplot(np.asarray(np.where(raster_mat[n + 100, :-2] == 1)) / sr * 1000, lineoffsets=0 + n,
                            linelengths=0.5, colors='black')
        axs[1, 1].eventplot(np.asarray(np.where(raster_mat[n + 120, :-2] == 1)) / sr * 1000, lineoffsets=0 + n,
                            linelengths=0.5, colors='black')
        axs[2, 1].eventplot(np.asarray(np.where(raster_mat[n + 140, :-2] == 1)) / sr * 1000, lineoffsets=0 + n,
                            linelengths=0.5, colors='black')
        axs[3, 1].eventplot(np.asarray(np.where(raster_mat[n + 160, :-2] == 1)) / sr * 1000, lineoffsets=0 + n,
                            linelengths=0.5, colors='black')
        axs[4, 1].eventplot(np.asarray(np.where(raster_mat[n + 180, :-2] == 1)) / sr * 1000, lineoffsets=0 + n,
                            linelengths=0.5, colors='black')
    axs[0, 0].set_title('{0} notch'.format(notch_freq[0]))
    axs[1, 0].set_title('{0} kHz notch'.format(notch_freq[1]))
    axs[2, 0].set_title('{0} kHz notch'.format(notch_freq[2]))
    axs[3, 0].set_title('{0} kHz notch'.format(notch_freq[3]))
    axs[4, 0].set_title('{0} kHz notch'.format(notch_freq[4]))
    axs[0, 1].set_title('{0} kHz notch'.format(notch_freq[5]))
    axs[1, 1].set_title('{0} kHz notch'.format(notch_freq[6]))
    axs[2, 1].set_title('{0} kHz notch'.format(notch_freq[7]))
    axs[3, 1].set_title('{0} kHz notch'.format(notch_freq[8]))
    axs[4, 1].set_title('{0} kHz notch'.format(notch_freq[9]))
    for ax in axs.flat:
        ax.set(xlabel='Time (ms)', ylabel='Trial #')
    for ax in axs.flat:
        ax.label_outer()

    fig1.suptitle('FM Notch Rasters')
    plt.tight_layout()

    if not os.path.exists(savelocation+bat+'/'):
        os.makedirs(savelocation+bat+'/')

    plt.savefig(savelocation+bat+'/Site_{0}_Chn{1}_FMnotch.jpg'.format(site, channel), dpi=300)
    plt.close()

# Example
for site in range(5, 7):
    for channel in range(1, 33):
        try:
            plot_notch_raster(r'S:/Smotherman_Lab/Auditory cortex/', 'Tb106', site, channel, 0.1, 'C:/Users/kbakshi/Documents/Data/notches/')
        except IndexError:
            pass
