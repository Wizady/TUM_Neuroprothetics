# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:45:54 2018

@author: Nico Hertel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import pandas as pd

os.chdir('Z:\\Studiumsunterlagen\\Master Elektro- und Informationstechnik\\SS 2018\\Neuroprothetics\\Neuroprothetics - Exercise\\Exercise 5')
from Exercise5 import HH_current, setup

def plot_potential_field(savename=None):
    rho = 3 # {Ohm m}
    I = 0.001 # {A}
    d = 10e-6 # {m}

    x = np.linspace(-25, 25, 101) * 1e-6 
    y = np.linspace(-25, 25, 101) * 1e-6 
    X, Y = np.meshgrid(x, y)

    pot = lambda x, y: rho*I/(4*np.pi) / (np.sqrt(d**2 + x**2 + y**2))

    P = pot(X, Y)

    sns.set_style('darkgrid')
    plt.figure()
    width = 8
    height =  (np.sqrt(5)-1.0)/2*width
    params = {
       'axes.labelsize': 25,
       'text.fontsize': 25,
       'legend.fontsize': 20,
       'xtick.labelsize': 20,
       'ytick.labelsize': 20,
       'text.usetex': False,
       'figure.figsize': [width, height]
       }
    matplotlib.rcParams.update(params)
    
    P = pot(X, Y)
    x_marks = np.array([x[0], x[20], x[40], x[60], x[80], x[100]]) * 1e6
    y_marks = np.array([y[0], y[20], y[40], y[60], y[80], y[100]]) * 1e6
    plt.xticks([0, 20, 40, 60, 80, 100], x_marks)
    plt.yticks([0, 20, 40, 60, 80, 100], y_marks)
    plt.xlabel(r'x in $\mu m$')
    plt.ylabel(r'y in $\mu m$')
    plt.pcolor(P, cmap='inferno')
    plt.colorbar().set_label(r'Potential $\Phi$ in V')

    if savename is not None:
        plt.savefig(savename+'.pdf', dpi=600)
        plt.savefig(savename+'.eps', dpi=600)


def plot_potential_axon(polarity=1, savename=None):
    ''' 
    Problems: 
        - E-Field is 1e6 order to big
        - F is qualitative correct, but wrong scale
    '''
    rho = 3 # [Ohm m]
    I = 0.001 * polarity # [A]
    d = 10e-6 # [m]
    cm = 0.01
    ra = 14.1 *1e6

    x = np.linspace(-25, 25, 101) * 1e-6
    r = np.sqrt(d**2 + x**2)

    pot = lambda r: rho*I/(4*np.pi) / r
    E = lambda x: rho*I/(4*np.pi) * (x/(x**2 + d**2)**1.5)
    F = lambda x: - rho*I/(4*np.pi*cm*ra) * (1/(x**2 + d**2)**1.5 - 3*x**2/(x**2 + d**2)**2.5)

    P = pot(r)
    Efield = E(x)
    Ffunc = F(x)

    sns.set_style('darkgrid')
    fig, ax = plt.subplots(3,1)
    width = 8
    height =  3/2 * (np.sqrt(5)-1.0)*width
    params = {
       'axes.labelsize': 17,
       'text.fontsize': 17,
       'legend.fontsize': 15,
       'xtick.labelsize': 15,
       'ytick.labelsize': 15,
       'text.usetex': False,
       'figure.figsize': [width, height]
       }
    matplotlib.rcParams.update(params)
    ax[0].plot(x*1e6, P)
    ax[0].set_title('Potential')
    ax[0].set_xlabel(r'x in $\mu m$')
    ax[0].set_ylabel(r'$\Phi$ in V')
    ax[1].plot(x*1e6, Efield)
    ax[1].set_title('Electric Field')
    ax[1].set_xlabel(r'x in $\mu m$')
    ax[1].set_ylabel(r'E in $\frac{V}{m}$')
    ax[2].plot(x*1e6, Ffunc)
    ax[2].set_title('Activation Function')
    ax[2].set_xlabel(r'x in $\mu m$')
    ax[2].set_ylabel(r'$f_a$ in $\frac{V}{m^2}$')

    fig.tight_layout()
    if savename is not None:
        plt.savefig(savename+'.pdf', dpi=600)
        plt.savefig(savename+'.eps', dpi=600)


def MultiCompartmentModel(stimulation, Ra=1, Cm= 1.26, N=100, dt=0.025, d=30,
                          savename=None, amp=1):

    D = int(d/dt)
    time = np.linspace(0, d-dt, D)

    # Create arrays
    if type(stimulation) == int: 
        C, Vm, Is, Ihh = setup(N, D)
        Ve = CreateStimuliPatter(pattern=stimulation, N=N, dt=dt, dur=d,
                                 amp=amp)
        if Ve is False:
            return False
    else:
        print('Error, no other stimulation currently possible!')
        # return False

    # Current gating values for all compartments
    m = np.ones((N)) * 0.053
    n = np.ones((N)) * 0.318
    h = np.ones((N)) * 0.596

    # Setting Parameters - Update these later

    A = np.identity(N)-dt/(Cm*Ra) * C

    for t, t_s in enumerate(time):
        if t_s < time[-1]:
            # Calculate Ihh for every compartment
            for i in range(N):
                Ihh[i][t+1], m[i], n[i], h[i] = HH_current(Vm[i, t], m[i],
                                                           n[i], h[i], dt)
            if stimulation == 4 or stimulation == 6:
                if t_s < 20.0:
                    for i in range(5):
                        Ihh[i][t+1] = Ihh[i+5][t+1]
                        m[i] = m[i+5]
                        n[i] = n[i+5]
                        h[i] = h[i+5]
                        Ihh[-i][t+1] = Ihh[-i-5][t+1]
                        m[-i] = m[-i-5]
                        n[-i] = n[-i-5]
                        h[-i] = h[-i-5]
            b = Vm[:, t] + dt/Cm * (-Ihh[:, t+1]+1/Ra*np.dot(C, Ve[:,t+1]))
            Vm[:, t+1] = np.linalg.solve(A, b)

    df = pd.DataFrame(Vm, columns=time)
    df.index.name = 'Compartment'
    if stimulation != 4 and stimulation != 6:
        n = 5
        for i in range(n):
            df.iloc[i] = df.iloc[n+1]
            df.iloc[-i] = df.iloc[-n-1]

    if stimulation == 6:
        df.drop(np.arange(0,25), inplace=True)
        df.drop(np.arange(125,150), inplace=True)

    plt.figure()
    width = 8
    height =  (np.sqrt(5)-1.0)/2*width
    params = {
       'axes.labelsize': 15,
       'text.fontsize': 15,
       'legend.fontsize': 15,
       'xtick.labelsize': 15,
       'ytick.labelsize': 15,
       'text.usetex': False,
       'figure.figsize': [width, height]
       }
    matplotlib.rcParams.update(params)
    sns.heatmap(df, yticklabels=10, xticklabels=200,
                cbar_kws={'label': 'V in mV'})
    plt.xlabel(r't in $ms$')

    
    if savename is not None:
        plt.savefig(savename+'.pdf', dpi=600)
        plt.savefig(savename+'.eps', dpi=600)
        plt.savefig(savename+'.jpg', dpi=600)

    return df, Ve


def CreateStimuliPatter(amp=1, pattern=1, N=100, dt=0.025, dur=100):
    '''
    Creates the stimuli pattern for the experiments
    '''
    D = int(dur/dt)
    Ve = np.zeros((N,D))
    d = 10e-6 # Distance Neuron from Electrode [m]
    dl = 0.5 * 1e-6 # Length of compartment [m]
    L = N*dl # Length of axon [m]
    x0 = L/2 # Position of Electrode [m]
    rho = 3 # [Ohm m]
    t0 = 5 # Start of pulse [ms]
    l0 = int(t0/dt) # Start index of pulse
    x = np.linspace(0, N*dl, N+1) # Start-Positions of the compartments [m]
    r = np.sqrt(d**2 + (x-x0)**2) # Distance from the electrode of the compartments [m]
    pot = lambda r: rho*I/(4*np.pi) / r # Potential Function [V]
    Ve = np.zeros((N,D))

    if pattern == 0:
        Ve = np.zeros((N, D))

    elif pattern == 1:
        # Stimulation by a mono-phasic current pulse, phase duration = 1 ms, current = -0.25 mA
        I = -0.25 * 1e-6 * amp # Stimulation current [A]
        l = int(1.0/dt) # Number of samples for duration
        V = pot(r[:-1]).reshape((N,1))
        Ve[:,l0:l0+l] = np.repeat(V, l, axis=1)

    elif pattern == 2:
        # Stimulation by a mono-phasic current pulse, phase duration = 1 ms, current = -1 mA
        I = -1.0 * 1e-6 * amp # Stimulation current [A]
        l = int(1.0/dt) # Number of samples for duration
        V = pot(r[:-1]).reshape((N,1))
        Ve[:,l0:l0+l] = np.repeat(V, l, axis=1)

    elif pattern == 3:
        # Stimulation by a bi-phasic current pulse, phase duration = 1 ms, current = -0.25 mA
        I = 0.5 * 1e-6 * amp # Stimulation current [A]
        l = int(1.0/dt) # Number of samples for duration
        V = pot(r[:-1]).reshape((N,1))
        Ve[:,l0:l0+l] = -np.repeat(V, l, axis=1) # negative phase first
        Ve[:,l0+l:l0+2*l] = np.repeat(V, l, axis=1)

    elif pattern == 4:
        # Stimulation by a bi-phasic current pulse, phase duration = 1 ms, current = -0.25 mA
        I = 2.0 * 1e-6 * amp # Stimulation current [A]
        l = int(1.0/dt) # Number of samples for duration
        V = pot(r[:-1]).reshape((N,1))
        Ve[:,l0:l0+l] = -np.repeat(V, l, axis=1) # negative phase first
        Ve[:,l0+l:l0+2*l] = np.repeat(V, l, axis=1)

    elif pattern == 5:
        # Stimulation by a mono-phasic current pulse, phase duration = 1 ms, current = 0.25 mA
        I = 0.25 * 1e-6 * amp # Stimulation current [A]
        l = int(1.0/dt) # Number of samples for duration
        V = pot(r[:-1]).reshape((N,1))
        Ve[:,l0:l0+l] = np.repeat(V, l, axis=1)

    elif pattern == 6:
        # Stimulation by a mono-phasic current pulse, phase duration = 1 ms, current = 5 mA
        I = 5 * 1e-6 * amp # Stimulation current [A]
        l = int(1.0/dt) # Number of samples for duration
        V = pot(r[:-1]).reshape((N,1))
        Ve[:,l0:l0+l] = np.repeat(V, l, axis=1)
    
    else:
        print('Error: Pattern unkown! Only pattern 1 and 2 available')
        return False
    return Ve

plot_potential_field('Exercise6_potential')

plot_potential_axon(1, 'Exercise6_potential_axon_1')
plot_potential_axon(-1, 'Exercise6_potential_axon_2')

df1, Ve1 = MultiCompartmentModel(stimulation=1, amp=2*1e5,
                                 savename='Exercise6_stim1_2')
df2, Ve2 = MultiCompartmentModel(stimulation=2, amp=2.5*1e5, Cm=1.6,
                                 savename='Exercise6_stim2_2')
df3, Ve3 = MultiCompartmentModel(stimulation=3, amp=2*1e5,
                                 savename='Exercise6_stim3_2')
df4, Ve4 = MultiCompartmentModel(stimulation=4, amp=2.5*1e5, Cm=2.1,
                                 savename='Exercise6_stim4_2')
df5, Ve5 = MultiCompartmentModel(stimulation=5, amp=2*1e5,
                                 savename='Exercise6_stim5_2')
df6, Ve6 = MultiCompartmentModel(stimulation=6, amp=1.5*1e5, N=150,
                                 savename='Exercise6_stim6_2')