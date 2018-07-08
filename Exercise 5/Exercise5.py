# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 09:31:10 2018

@author: Nico Hertel
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def setup(N=100, D=100):
    '''
    Sets up the connection-matrix C and empty arrays for the membrane potential
    Vm, stimulation current Is and the internal current Ihh.

    Parameters
    ----------
    N : int
        Number of compartments
    D : int
        Number of timesteps

    Returns
    -------
    C : array
        NxN-sized array with the standard connection matrix
    Vm : array
        NxD-sized zero-array
    Is : array
        NxD-sized zero-array
    Ihh : array
        NxD-sized zero-array
    '''
    Vm = np.zeros((N,D))
    Is = np.zeros((N,D))
    Ihh = np.zeros((N,D))

    x = np.ones((N))
    C = -2*np.diag(x)
    C[0,0] = -1
    C[-1, -1] = -1
    C1 = np.diag(x[0:-1], 1)
    C2 = np.diag(x[0:-1], -1)
    C = C + C1 + C2

    return C, Vm, Is, Ihh


def HH_current(Vm, m, n, h, dt, T=6.3):
    '''
    Calculates the next value of the internal current Ihh

    Parameters
    ----------
    Vm : float
        Current membrane potential in mV
    m : float
        Current m-value for this compartment
    n : float
        Current n-value for this compartment
    h : float
        Current h-value for this compartment
    dt : float
        Timestep

    Return
    ------
    Ihh : float
        Internal current for this compartment
    m_next/n_next/h_next : float
        Next m/n/h-values for this compartment
    '''
    g_Na = 120
    g_K = 36
    g_L = 0.3
    V_Na = 115
    V_K = -12
    V_L = 10.6
    k = 3**(0.1*(T-6.3))

    # Define functions
    alpha_m = lambda V: (2.5-0.1*V)/(np.exp(2.5-0.1*V)-1)
    alpha_n = lambda V: (0.1-0.01*V)/(np.exp(1-0.1*V)-1)
    alpha_h = lambda V: 0.07*np.exp(-V/20)
    beta_m = lambda V: 4*np.exp(-V/18)
    beta_n = lambda V: 0.125*np.exp(-V/80)
    beta_h = lambda V: 1/(np.exp(3-0.1*V)+1)

    F_m = lambda a_m, b_m, m: (a_m*(1-m)-b_m*m)*k
    F_n = lambda a_n, b_n, n: (a_n*(1-n)-b_n*n)*k
    F_h = lambda a_h, b_h, h: (a_h*(1-h)-b_h*h)*k

    I_Na = lambda m, h, V: g_Na*m**3*h*(V-V_Na)
    I_K = lambda n, V: g_K*n**4*(V-V_K)
    I_L = lambda V: g_L*(V-V_L)
    
    # Calculate values
    a_m = alpha_m(Vm)
    b_m = beta_m(Vm)
    a_n = alpha_n(Vm)
    b_n = beta_n(Vm)
    a_h = alpha_h(Vm)
    b_h = beta_h(Vm)

    m_next = m + F_m(a_m, b_m, m) * dt 
    n_next = n + F_n(a_n, b_n, n) * dt
    h_next = h + F_h(a_h, b_h, h) * dt

    i_Na = I_Na(m, h, Vm)
    i_K = I_K(n, Vm)
    i_L = I_L(Vm)

    Ihh = i_Na + i_K + i_L

    return Ihh, m_next, n_next, h_next


def MultiCompartmentModel(Ra=1, Cm= 1.26, amp=10e-8, N=100, dt=0.025, d=100,
                          stimulation=None, savename=None):

    D = int(d/dt)
    time = np.linspace(0, d-dt, D)

    # Create arrays
    if stimulation is None:
        C, Vm, Is, Ihh = setup(N, D)
    elif type(stimulation) == int: 
        C, Vm, Is, Ihh = setup(N, D)
        Is = CreateStimuliPatter(amp=amp, pattern=stimulation, N=N, dt=dt, d=d)
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
            b = Vm[:, t] + dt/Cm * (-Ihh[:, t+1]+Is[:,t+1])
            Vm[:, t+1] = np.linalg.solve(A, b)

    df = pd.DataFrame(Vm, columns=time)
    df.index.name = 'Compartment'
    if savename is not None:
        plt.figure()
        width = 8
        height =  (np.sqrt(5)-1.0)/2*width
        params = {
           'axes.labelsize': 12,
           'text.fontsize': 12,
           'legend.fontsize': 10,
           'xtick.labelsize': 10,
           'ytick.labelsize': 10,
           'text.usetex': False,
           'figure.figsize': [width, height]
           }
        matplotlib.rcParams.update(params)
        sns.heatmap(df, yticklabels=10, xticklabels=False)
        plt.xlabel('t in s')
    
    
        plt.savefig(savename+'.pdf', dpi=600)
        plt.savefig(savename+'.eps', dpi=600)
    else:
        plt.figure(figsize=(20,20))
        sns.heatmap(Vm)
    return df, Is


def CreateStimuliPatter(amp=10e-8, pattern=1, N=100, dt=0.025, d=100):
    '''
    Creates the stimuli pattern for the experiments
    '''
    D = int(d/dt)
    Is = np.zeros((N,D))
    if pattern == 1:
        # Patter 1: 5ms rect puls at first compartment with 1uA/cm²
        d_1 = int(5/dt)
        Is[0,0:d_1] = amp
    elif pattern == 2:
        # Patter 2: 5ms rect puls at compartment 20 and 80
        d_2 = int(5/dt)
        Is[19, 0:d_2] = amp
        Is[79, 0:d_2] = amp
    elif pattern == 3:
        Is[0, :] = A
    elif pattern == 4:
        d_4 = int(5/dt)
        Is[0, 8*d_4:9*d_4] = amp
    elif pattern == 5:
        d_4 = int(5/dt)
        Is[0, 2*d_4:3*d_4] = amp
        Is[0, 12*d_4:13*d_4] = amp
    else:
        print('Error: Pattern unkown! Only pattern 1 and 2 available')
        return False
    return Is







































