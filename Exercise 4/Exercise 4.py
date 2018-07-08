# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 07:29:52 2018

@author: Nico Hertel
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from plot_setup import plot_setup
from ODE_Solvers import heun_step, run_solver


def exercise4_timeconstants(T=6.3, savename=None):
    '''
    Will plot the time constants and steady state value for the gating
    variables m, n and h of the Hodgkins-Huxley-Model at two
    temperatures (6.3°C and 28°C) for a membrane potential between
    [-100mV, 100mV]
    
    Parameters
    ----------
    savename : None or str
        If give, will save the plot here
    '''

    Vm = np.linspace(-0.1, 0.1, 1001)
    k = 3**(0.1*(T-6.3))
    # Gating Variable: m
    alpha_m = (0.025-Vm)/(0.01*(np.exp((0.025-Vm)/0.01)-1))
    beta_m = 4*np.exp(-Vm/0.018)

    tau_m = 1/(k*(alpha_m + beta_m)) # time constant
    m_ss = alpha_m/(alpha_m + beta_m) # steady state value
    
    # Gating Variable: n
    alpha_n = (0.001 - 0.1*Vm)/(0.01*(np.exp((0.01-Vm)/0.01)-1))
    beta_n = 0.125*np.exp(-Vm/0.08)

    tau_n = 1/(k*(alpha_n + beta_n)) # time constant
    n_ss = alpha_n/(alpha_n + beta_n) # steady state value

    # Gating Variable: h
    alpha_h = 0.07*np.exp(-Vm/0.02)
    beta_h = 1/(np.exp((0.03-Vm)/0.01)+1)

    tau_h = 1/(k*(alpha_h + beta_h)) # time constant
    h_ss = alpha_h/(alpha_h + beta_h) # steady state value

    print(r'm($V_m=0V$)=%.3f' %m_ss[500])
    print(r'n($V_m=0V$)=%.3f' %n_ss[500])
    print(r'h($V_m=0V$)=%.3f' %h_ss[500])

    # Plotting
    xticks = [-0.100, -0.050, 0, 0.050, 0.100]
    fig, ax = plt.subplots(1,2, sharex=True)
    width = 4.5
    height =  (np.sqrt(5)-1.0)/2*width
    params = {
       'axes.labelsize': 8,
       'text.fontsize': 8,
       'legend.fontsize': 10,
       'xtick.labelsize': 10,
       'ytick.labelsize': 10,
       'text.usetex': False,
       'figure.figsize': [width, height]
       }
    matplotlib.rcParams.update(params)
    sns.set_style('darkgrid')
    sns.set_palette(sns.color_palette("muted"))
    ax[0].plot(Vm, tau_m, label=r'$\tau_m$')
    ax[0].plot(Vm, tau_n, label=r'$\tau_n$')
    ax[0].plot(Vm, tau_h, label=r'$\tau_h$')
    ax[0].legend()
    ax[0].set_xticks(xticks)
    ax[0].set_xlabel(r'$V_m$/mV')
    ax[0].set_ylabel(r'$\tau_x$/ms')
    ax[0].set_title('Time Constant at %.1f °C' %T)

    ax[1].plot(Vm, m_ss, label=r'$m_{\infty}$')
    ax[1].plot(Vm, n_ss, label=r'$n_{\infty}$')
    ax[1].plot(Vm, h_ss, label=r'$h_{\infty}$')
    ax[1].legend()
    ax[1].set_xticks(xticks)
    ax[1].set_xlabel(r'$V_m$/mV')
    ax[1].set_ylabel(r'$x_{\infty}$/ms')
    ax[1].set_title('Steady State Value at %.1f °C' %T)
    fig.tight_layout()

    if savename is not None:
        fig.savefig(savename+'.pdf', dpi=600)
        fig.savefig(savename+'.eps', dpi=600)


def HH_model(i_in, T=6.3, savename=None, phase_title=None):
    # Define Parameters as given in lecture slides
    # Always use SI-Units!
    g_Na = 120
    g_K = 36
    g_L = 0.3
    V_Na = 115
    V_K = -12
    V_L = 10.6
    C = 1
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

    F_V = lambda i_Na, i_K, i_L, i_in: 1/C*(i_in-i_Na-i_K-i_L)

    # Set simulation conditions
    d = 100                 # Duration {s}
    dt = 0.01            # Timestep {s}
    V_0 = 0                 # Resting Voltage {V}
    m_0 = 0.053             # Start Gating Value (Steady State at Vm=0V)
    n_0 = 0.318             # Start Gating Value (Steady State at Vm=0V)
    h_0 = 0.596             # Start Gating Value (Steady State at Vm=0V)

    time = np.linspace(0, d, int(d/dt))  # Time Vector
    Vm = np.zeros(time.shape)
    a_m = np.zeros(time.shape)
    b_m = np.zeros(time.shape)
    a_n = np.zeros(time.shape)
    b_n = np.zeros(time.shape)
    a_h = np.zeros(time.shape)
    b_h = np.zeros(time.shape)
    m = np.zeros(time.shape)
    n = np.zeros(time.shape)
    h = np.zeros(time.shape)
    i_Na = np.zeros(time.shape)
    i_K = np.zeros(time.shape)
    i_L = np.zeros(time.shape)
    
    # Fill with start conditions
    Vm[0] = V_0
    m[0] = m_0
    n[0] = n_0
    h[0] = h_0
    a_m[0] = alpha_m(Vm[0])
    b_m[0] = beta_m(Vm[0])
    a_n[0] = alpha_n(Vm[0])
    b_n[0] = beta_n(Vm[0])
    a_h[0] = alpha_h(Vm[0])
    b_h[0] = beta_h(Vm[0])
    i_Na[0] = I_Na(m[0], h[0], Vm[0])
    i_K[0] = I_K(n[0], Vm[0])
    i_L[0] = I_L(Vm[0])

    for i,t in enumerate(time):
        if i != 0:
            a_m[i] = alpha_m(Vm[i-1])
            b_m[i] = beta_m(Vm[i-1])
            a_n[i] = alpha_n(Vm[i-1])
            b_n[i] = beta_n(Vm[i-1])
            a_h[i] = alpha_h(Vm[i-1])
            b_h[i] = beta_h(Vm[i-1])

            m[i] = m[i-1] + F_m(a_m[i-1], b_m[i-1], m[i-1]) * dt
            n[i] = n[i-1] + F_n(a_n[i-1], b_n[i-1], n[i-1]) * dt
            h[i] = h[i-1] + F_h(a_h[i-1], b_h[i-1], h[i-1]) * dt

            i_Na[i] = I_Na(m[i-1], h[i-1], Vm[i-1])
            i_K[i] = I_K(n[i-1], Vm[i-1])
            i_L[i] = I_L(Vm[i-1])

            Vm[i] = Vm[i-1] + F_V(i_Na[i-1], i_K[i-1], i_L[i-1], i_in[i-1])*dt

    fig, ax = plt.subplots(4,1, sharex=True)
#    params = {'axes.labelsize': 18,
#              'axes.titlesize': 22,
#              'text.fontsize': 18,
#              'legend.fontsize': 16,
#              'xtick.labelsize': 16,
#              'ytick.labelsize':16,
#              'lines.linewidth': 3
#              }
#    matplotlib.rcParams.update(params)
    width = 4.5
    height =  (np.sqrt(5)-1.0)/2*width*2
    params = {
       'axes.labelsize': 8,
       'text.fontsize': 8,
       'legend.fontsize': 10,
       'xtick.labelsize': 10,
       'ytick.labelsize': 10,
       'text.usetex': False,
       'figure.figsize': [width, height]
       }
    matplotlib.rcParams.update(params)
    sns.set_style('darkgrid')
    sns.set_palette(sns.color_palette("muted"))
    
    ax[0].plot(time, Vm)
    ax[0].set_xlabel('t/ms')
    ax[0].set_ylabel(r'$V_m/mV$')
    ax[0].set_title('Membrane Potential')
    ax[0].set_yticks([-25, 0, 25, 50, 75, 100, 125])
    
    ax[1].plot(time, m, label='m')
    ax[1].plot(time, n, label='n')
    ax[1].plot(time, h, label='h')
    ax[1].set_xlabel('t/ms')
    ax[1].set_ylabel(r'$P_x(s)$')
    ax[1].set_title('Gating Variables')
    ax[1].set_yticks([0.0, 0.5, 1.0])
    ax[1].legend()

    ax[2].plot(time, i_Na, label=r'$I_{Na}$')
    ax[2].plot(time, i_K, label=r'$I_K$')
    ax[2].set_xlabel('t/ms')
    ax[2].set_ylabel(r'$i(t)$ $\frac{\mu A} {cm^2}$')
    ax[2].set_title('Current Densities')
    ax[2].set_yticks([-1000, -500, 0, 500, 1000])
    ax[2].legend()

    ax[3].plot(time, i_in, label=r'$I_{In}$')
    ax[3].set_xlabel('t/ms')
    ax[3].set_ylabel(r'$i(t)$ $\frac{\mu A} {cm^2}$')
    ax[3].set_title('Input Current')
    
    fig.tight_layout()
    if savename is not None:
        fig.savefig(savename+'timeplot.eps', dpi=600)
        fig.savefig(savename+'timeplot.pdf', dpi=600)

    plt.figure()
    params = {
       'axes.labelsize': 12,
       'text.fontsize': 12,
       'legend.fontsize': 14,
       'xtick.labelsize': 14,
       'ytick.labelsize': 14,
       'text.usetex': False,
       'figure.figsize': [width, height]
       }
    matplotlib.rcParams.update(params)

    plt.plot(Vm, i_Na, label=r'$I_{Na}$')
    plt.plot(Vm, i_K, label=r'$I_{K}$')
    plt.plot(Vm, i_L, label=r'$I_{leek}$')
    plt.xlabel('V/mV')
    plt.ylabel(r'$i(t)$ $\frac{\mu A} {cm^2}$')
    if phase_title is not None:
        plt.title(phase_title)
    plt.legend()
    if savename is not None:
        plt.savefig(savename+'phaseplot.eps', dpi=600)
        plt.savefig(savename+'phaseplot.pdf', dpi=600)
        
d = 100                 # Duration {s}
dt = 0.01            # Timestep {s}


time = np.linspace(0, d, int(d/dt))  # Time Vector       

I_in1 = np.zeros(time.shape)
I_in2 = np.zeros(time.shape)

for i in range(5):
    I_in1[i*1500:i*1500+500] = (i+1)
    I_in2[i*1500:i*1500+500] = (2**(i+1))

HH_model(i_in=I_in1, T=6.3, savename='Exercise 4\Exercise4_A_63C_2', phase_title='Phase Plot for (A) at 6.3°C')
HH_model(i_in=I_in2, T=6.3, savename='Exercise 4\Exercise4_B_63C_2', phase_title='Phase Plot for (B) at 6.3°C')
HH_model(i_in=I_in1, T=28, savename='Exercise 4\Exercise4_A_28C_2', phase_title='Phase Plot for (A) at 28°C')
HH_model(i_in=I_in2, T=28, savename='Exercise 4\Exercise4_B_28C_2', phase_title='Phase Plot for (B) at 28°C')


exercise4_timeconstants(T=6.3, savename='Exercise 4\Exercie4_timeconst_63C')
exercise4_timeconstants(T=28, savename='Exercise 4\Exercie4_timeconst_28C')












