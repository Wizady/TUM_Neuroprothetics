# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 14:41:49 2018

@author: Nico Hertel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.signal import butter, lfilter, freqz
from scipy.io import wavfile
from scipy import fft

def create_corner_frequencies(n, interval=[200, 8000]):
    '''
    Returns the corncer frequencies in a logspace of a CI with
    n electrodes in the given frequency interval

    Parameters
    ----------
    n : int
        Number of Electrodes
    interval : list
        Start and End point of the interval

    Returns
    ------
    cf : ndarray
        Corner Frequencies of the CI
    '''
    if len(interval) != 2:
        print('Error: Interval has to be of length 2!')
        return None
    cf = np.logspace(np.log10(interval[0]), np.log10(interval[1]), n+1)
    return cf


def print_corner_frequencies(cf, title=None, savename=None):
    '''
    Plots and saves the corner frequencies

    Parameters
    ----------
    cf : array
        The corner frequencies of the CI
    title : str
        The title of the plot
    savename : str
        Path, name and type to store the plot
    '''  
    width = 8
    height =  (np.sqrt(5)-1.0)/2*width
    params = {
       'axes.labelsize': 25,
       'text.fontsize': 25,
       'legend.fontsize': 20,
       'xtick.labelsize': 20,
       'ytick.labelsize': 20,
       'text.usetex': False,
       'lines.linewidth' : 5,
       'figure.figsize': [width, height]
       }
    matplotlib.rcParams.update(params)
    fig, ax = plt.subplots(1, 1)
    ax.step(range(len(cf)), cf, where='mid')
    ax.set_yscale('log', basey=10)
    ax.grid(True, which='both')
    ax.set_ylim((100, 10000))
    if title is not None:
        fig.suptitle(title)
    ax.set_ylabel('f in Hz')
    ax.set_xlabel('Electrode')
    fig.subplots_adjust(top=0.85)
    if savename is not None:
        fig.savefig(savename, dpi=600)


def create_bandpass(low, high, fs, order=8):
    '''
    Creates a butterworth bandpass filter

    Parameters
    ----------
    low : float
        Lower cutoff frequency
    high : float
        Higher cutoff frequency
    fs : float
        Samplign Frequency
    order : int
        Order of the filter

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (b) and denominator (a) polynomials of the filter
    '''
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def plot_bandpass(low, high, fs, orders=[3, 5, 8]):
    width = 8
    height =  (np.sqrt(5)-1.0)/2*width
    params = {
       'axes.labelsize': 25,
       'text.fontsize': 25,
       'legend.fontsize': 20,
       'xtick.labelsize': 20,
       'ytick.labelsize': 20,
       'text.usetex': False,
       'lines.linewidth' : 3,
       'figure.figsize': [width, height]
       }
    matplotlib.rcParams.update(params)
    plt.figure()
    for order in orders:
        b, a = create_bandpass(low, high, fs, order)
        w, h = freqz(b, a)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

def create_filterbank(n, interval=[200, 8000], fs=25000, order=8):
    '''
    Creates a filter bank of bandpass filters

    Parameters
    ----------
    n : int
        Number of filters
    interval : array
        Start and Stop corner frequencies of the filterbank
    fs : float
        Sampling Frequency
    order : int
        Order of the filters

    Returns
    -------
    filterbank : list
        The collected filterbank
    '''
    cf = create_corner_frequencies(n, interval)
    if cf is None: # Check if cf is valid
        return None
    filterbank = [] # Initilize filterbank, then iterate over each filter
    for i in range(n):
        f_low = cf[i]
        f_high = cf[i+1]
        filter_i = create_bandpass(low=f_low, high=f_high, order=order, fs=fs)
        filterbank.append(filter_i)
    return filterbank


def plot_filterbank(filterbank, cf, fs=25000, title=None, savename=None):
    '''
    Plots and saves the frequency response of the filterbank

    Parameters
    ----------
    filterbank : list
        List of the filters
    cf : array
        The corner frequencies of the CI
    fs : float
        Sampling Frequency
    title : str
        The title of the plot
    savename : str
        Path, name and type to store the plot
    '''  
    width = 12
    height =  (np.sqrt(5)-1.0)/2*width
    params = {
       'axes.labelsize': 25,
       'text.fontsize': 25,
       'legend.fontsize': 20,
       'xtick.labelsize': 20,
       'ytick.labelsize': 20,
       'text.usetex': False,
       'lines.linewidth' : 3,
       'figure.figsize': [width, height]
       }
    matplotlib.rcParams.update(params)
    fig, ax = plt.subplots(1, 1)
    for i in range(len(filterbank)):
        b, a = filterbank[i]
        w, h = freqz(b, a)
        ax.plot((fs * 0.5 / np.pi) * w, 20*np.log10(abs(h)),
                label=r'Filter %i |$f_{low}$=%.2f, $f_{high}$=%.2f ' %(i, cf[i], cf[i+1]))
    ax.set_xscale('log')
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel('Gain in dB')
    ax.set_xlabel('f in Hz')
    ax.grid(True, which='both', axis='y')
    ax.set_xlim([100, 10000])
    ax.set_ylim([-40, 5])
    if savename is not None:
        plt.savefig(savename, dpi=600)


def filter_soundfile(file, fs, filterbank):
    '''
    Filters the soundfile through each filter in the filterbank

    Parameters
    ----------
    file : ndarray
        A soundfile
    fs : float
        Sampling rate of the soundfile
    filterbank : list
        List of all filters

    Returns
    -------
    file_filtered : list
        List with the filered soundfiles from first to last filter
    '''
    file_filtered = []
    for i, [b, a] in enumerate(filterbank):
        # b, a = filterbank[i]
        y = lfilter(b, a, file).astype('int16')
        file_filtered.append(y)
    return file_filtered


def save_sound(file_filtered, fs, savename):
    '''
    Saves each channel as an individual file

    Parameters
    ----------
    file_filtered : list
        List of the single channels
    fs : float
        Sampling frequency of original file
    savename : str
        Path and name to store the files as .wav

    Note
    ----
    The files will be stored as filename + '_%i.wav'
    with i being the channel number
    '''
    for i, channel in enumerate(file_filtered):
        wavfile.write(data=channel, rate=fs, filename=savename+'_%i.wav' %i)
        print('Saved %s_%i.wav' %(savename, i))


def plot_file_filtered(file_filtered, cf, title=None, savename=None, cols=2):
    '''
    Plots each channel of the filetered soundfile as a single plot

    Parameters
    ----------
    file_filtered : list
        List of each channel of the file
    cf : array
        The corner frequencies of the filters
    title : str
        The title of the plot
    savename : str
        Path, name and type to store the plot
    cols : int
        Number of columns in the plot
    '''
    n_plots = len(file_filtered)
    rows = int(np.ceil(n_plots/2))

    width = 8 * cols
    height =  (np.sqrt(5)-1.0)/4*width * rows
    params = {
       'axes.labelsize': 25,
       'text.fontsize': 23,
       'legend.fontsize': 15,
       'xtick.labelsize': 20,
       'ytick.labelsize': 20,
       'text.usetex': False,
       'lines.linewidth' : 3,
       'figure.figsize': [width, height]
       }
    matplotlib.rcParams.update(params)
    fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
    for i, channel in enumerate(file_filtered):
        axis = ax.flat[i]
        t = np.arange(0, len(channel))/fs
        axis.plot(t, channel/10000,
                          label=r'$f_{low}$=%.0fHz | $f_{high}$=%.0fHz' %(cf[i], cf[i+1]))
        axis.set_title('Channel %i' %(i+1))
        axis.set_xlabel('t in s')
        axis.set_ylabel('Amplitude')
        axis.legend()
    fig.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=600)

def reconstruct_sound(file_filtered):
    file_reconstructed = np.zeros(file_filtered[0].shape)
    for channel in file_filtered:
        file_reconstructed += channel
    return file_reconstructed


def plot_spectrum(y, fs, title=None, savename=None):
    '''
    Plots the Spectrum of the signal

    Parameters
    ----------
    y : ndarray
        The signal to analyse
    Fs : float
        The sampling frequency of the signal
    title : str
        The title of the plot
    savename : str
        Path, name and type to store the plot
    '''
    sns.set_style('darkgrid')
    width = 8
    height =  (np.sqrt(5)-1.0)/2*width
    params = {
       'axes.labelsize': 25,
       'text.fontsize': 23,
       'legend.fontsize': 15,
       'xtick.labelsize': 20,
       'ytick.labelsize': 20,
       'text.usetex': False,
       'lines.linewidth' : 2,
       'figure.figsize': [width, height]
       }
    matplotlib.rcParams.update(params)
    fig, ax = plt.subplots(1,1)

    n = len(y)
    T = n/fs
    frq = np.arange(n)/T
    frq = frq[range(int(n/2))] # one sided frequency range

    Y = fft(y)/n # Norm Y by length
    Y = Y[range(int(n/2))] # one sided spectrum
    Y = abs(Y)

    ax.loglog(frq, Y)
    ax.set_xlabel('Frequency in Hz')
    ax.set_ylabel('|Y|')
    fig.tight_layout()
    if title is not None:
        ax.set_title(title)
    if savename is not None:
        fig.savefig(savename, dpi=600, bbox_inches="tight")

def plot_spectrogram(file_recostructed, fs, t_window=10e3, t_overlap=5e-3,
                     title=None, savename=None):
    '''
    Plots the spectrogram of the reconstructed soundfile

    Parameters
    ----------
    file_reconstructed : ndarray
        The reconstructed soundfile
    fs : float
        Sampling frequency of the soundfile
    t_windows : float
        Duration of the FFT-window [s]
    t_overlap: float
        Duration of the overlap [s]
    title : str
        The title of the plot
    savename : str
        Path, name and type to store the plot
    '''
    n_window = int(t_window * fs_file) # Number of samples of window
    n_overlap = int(t_overlap * fs_file) # Number of samples of overlap

    sns.set_style('darkgrid')
    width = 8
    height =  (np.sqrt(5)-1.0)/2*width
    params = {
       'axes.labelsize': 25,
       'text.fontsize': 23,
       'legend.fontsize': 15,
       'xtick.labelsize': 20,
       'ytick.labelsize': 20,
       'text.usetex': False,
       'lines.linewidth' : 2,
       'figure.figsize': [width, height]
       }
    matplotlib.rcParams.update(params)
    fig, ax = plt.subplots(1,1)

    [spec, freq, t, ax2] = ax.specgram(file_recostructed, Fs=fs_file,
                                       noverlap=n_overlap, NFFT=n_window)
    ax.set_xlim([0.2, 1.0])
    ax.set_xlabel('t in s')
    ax.set_ylabel('f in Hz')
    fig.colorbar(ax2).set_label('Intensity [dB]')
    if title is not None:
        ax.set_title(title)
    if savename is not None:
        fig.savefig(savename, dpi=600, bbox_inches="tight")

#==============================================================================
#                       Neuroprothetics Exercise 7
#                             CI Filter Banks
#==============================================================================


'''
 1.1 Electrode corner frequencies

 The band-pass filters in CIs range from about 200 Hz for the most apical electrode to
 8 kHz for the most basal electrode. Use logspace to calculate the corner frequencies of
 the filters in CIs with 3, 6, 12 and 22 electrodes. Plot the result for the 22-electrode CI.
 (solution for 3 electrodes: [200, 680, 2340, 8000])
'''

cf = create_corner_frequencies(22)
print_corner_frequencies(cf, 'Corner Frequencies for 22 Electrodes', 
                         savename='.\Exercise 7\plots\cornerfreq.pdf')

'''
 1.2 Implement a filter bank

 Implement an eighth-order band-pass filter bank with the corner frequencies from 1.1
 Plot the frequency response of the filter bank on a double-logarithmic scale for 3 and 22
 electrodes (don’t forget the units).
'''

fs = 25000
filterbank_22 = create_filterbank(22, interval=[200, 8000], fs=fs, order=3)
plot_filterbank(filterbank_22, cf, fs=25000,
                title='Frequency Response for 22 Filters',
                savename='.\Exercise 7\plots\\filterbank_22.pdf')

filterbank_3 = create_filterbank(3, interval=[200, 8000], fs=fs, order=3)
plot_filterbank(filterbank_3, cf, fs=25000,
                title='Frequency Response for 3 Filters',
                savename='.\Exercise 7\plots\\filterbank_3.pdf')

'''
 Record an acoustically interesting word with a microphone and filter it with the filter banks.
 Plot and listen to the output (time signal) of each filter channel of a 12-electrode CI.
'''
[fs_file, file] = wavfile.read('.\Exercise 7\\sounds\\Nuss.wav')
cf = create_corner_frequencies(12)
filterbank = create_filterbank(12, order=3)
file_filtered = filter_soundfile(file, fs, filterbank)
plot_file_filtered(file_filtered, cf, savename='.\Exercise 7\plots\\filtered_signal.pdf')

'''
 1.3 Join the channels

 Sum the channel outputs and listen to the result. Plot the spectra (loglog)
 and spectrograms (spectrogram) of joint signal for all given CI types.
 Use a time window of 10 ms with an overlap of 5 ms for the spectrograms.
'''
file_recostructed = reconstruct_sound(file_filtered)
plot_spectrum(file_recostructed, fs=fs, title='Reconstructed Soundfile',
              savename='.\Exercise 7\plots\\reconstructed_signal.pdf')

# Iterate over all CIs
for n in [3, 6, 12, 22]:
    filterbank = create_filterbank(n, order=3, fs=fs_file)
    file_filtered = filter_soundfile(file, fs, filterbank)
    file_recostructed = reconstruct_sound(file_filtered)
    plot_spectrum(file_recostructed, fs=fs,
                  title='Reconstructed Soundfile with %i Electrodes' %n,
                  savename='.\Exercise 7\plots\\reconstructed_signal_%i.pdf' %n)
plot_spectrum(file, fs=fs_file, title='Original Soundfile',
              savename='.\Exercise 7\plots\\original_signal.pdf')
plot_spectrogram(file_recostructed, fs_file, title='Spectrogram for 22 Electrodes')

