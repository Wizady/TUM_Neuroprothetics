# -*- coding: utf-8 -*-
"""
Neuroprothetics - Exercise 1 - Introduction

Created on Fri Apr 20 14:14:14 2018

@author: Nico Hertel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


def plot_setup(width=None, height=None):
    if width is None and height is None:
        width = 3.39
        height = width * (np.sqrt(5)-1.0)/2
    elif height is None:
        height = width * (np.sqrt(5)-1.0)/2
    elif width is None:
        width = height / (np.sqrt(5)-1.0)/2

    params = {'axes.labelsize': 16,
              'axes.titlesize': 16,
              'text.fontsize': 16,
              'legend.fontsize': 16,
              'xtick.labelsize': 16,
              'ytick.labelsize':16,
              'figure.figsize': [width, height]}
    matplotlib.rcParams.update(params)
    

def create_signal(freq, amp, duration, f_s):
    '''
    Creates a signal vector consisting of the sum of all sin-function
    described in freq, amp for the given duration and sampling frequency

    Parameters:
    -----------
    freq : list
        List of the frequencies of the sin-signals
        Has to be the same length as amp
    amp : list
        List of the amplituedes of the sin-signals
        Has to be the same length as freq
    duration : int
        Duration of the final signal in seconds
    f_s : int
        Sampling frequency of the final signal in Hz

    Output:
    -------
    t : list
        Time-vector
    signal : list
        The created signal

    Notes:
    ------
    The signal is computed as the sum of all described signals:
        signal_i = amp[i] * sin(2*pi*freq[i]*t) + amp[0]
        signal = sum(signal_[i])
    '''
    #Check if freq and amp are same dimension
    if len(freq) != len(amp):
        print('Error: Frequency and Amplitude-Vector have to have same length')
        return None

    t = np.arange(0,duration,1/f_s)
    signal = np.zeros(t.shape)
    for i in range(len(freq)):
        if freq[i] == 0 and i == 0:
            signal = signal + np.ones(signal.shape) * amp[i]
        else:
            signal = signal + np.sin(2 * np.pi * freq[i] * t) * amp[i]

    return t, signal


def plot_signal(t, signal, title='Plot of Exercise 1.1(b)', savename=None):
    '''
    Plots t and signal with title and saves plot at savename if given
    '''
    plot_setup(width=8)
    plt.plot(t, signal, linewidth=0.3)

    plt.xlabel('t in s')
    plt.ylabel('Amplitude')
    plt.title(title)
    if savename is not None:
        plt.savefig(savename, dpi=600)
    plt.show()


def plot_spectrums(signals, savename=None):
    n_plots = len(signals)
    plot_setup(8,8)
    for i, key in enumerate(signals):
        plt.subplot(n_plots, 1, i+1)
        spec = np.abs(np.fft.fft(signals[key][0])/int(len(signals[key][0])/2))
        f = np.fft.fftfreq(signals[key][0].size, 1/signals[key][1])
        f = f[0:int(len(f)/2)]
        spec = spec[0:int(len(spec)/2)]
        spec[0] = spec[0]/2
        plt.plot(f, spec, label='f_s = ' + key)
        plt.xlabel('f in Hz')
        plt.ylabel('|A|')
        plt.legend()
    plt.tight_layout()
    plt.suptitle('Plots for Exercise 2.1(a)', fontsize=16)
    plt.subplots_adjust(top=0.95)
    if savename is not None:
        plt.savefig(savename, dpi=600)
    plt.show()


freq = [0, 100, 600, 9000]
amp = [3, 1, 1.5, 2]
f_s = [10000, 20000, 100000]

t_10kHz, x_10kHz = create_signal(freq, amp, 0.1, f_s[0])
t_20kHz, x_20kHz = create_signal(freq, amp, 0.1, f_s[1])
t_100kHz, x_100kHz = create_signal(freq, amp, 0.1, f_s[2])

t1, x1 = create_signal([0, 100], [3, 1], 0.1, 100000)
t2, x2 = create_signal([0, 600], [3, 1.5], 0.1, 100000)
t3, x3 = create_signal([0, 9000], [3, 2], 0.1, 100000)
plot_signal(t1, x1, 'x1')
plot_signal(t2, x2, 'x2')
plot_signal(t3, x3, 'x3')
plot_signal(t1, x1+x2+x3, 'sum')
plot_signal(t_100kHz, x_100kHz)
plot_signal(t_10kHz, x_10kHz, '10Khz')
plot_signal(t_20kHz, x_20kHz, '20Khz')
plot_signal(t_100kHz, x_100kHz, 'Plot of Exercise 1.1(b)', '200418_exercise1b.png')


signals = {'10 kHz': [x_10kHz, f_s[0]], '20 kHz': [x_20kHz, f_s[1]],
          '100 kHz': [x_100kHz, f_s[2]]}


plot_spectrums(signals, '.\\Exercise 1\\200418_exercise21b.png')
