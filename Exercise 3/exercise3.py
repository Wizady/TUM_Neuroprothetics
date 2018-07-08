# -*- coding: utf-8 -*-
"""
Created on Sun May 20 14:50:05 2018

@author: Nico Hertel
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from ODE_Solvers import explicit_euler_step, heun_step, exponential_euler_step
from ODE_Solvers import run_solver
from plot_setup import plot_setup


def exercise3_SolveFunctions(savename=None):
    '''
    Plots the differential equation
        dV/dt = 1-V-t
    With the start conditions
        V0 = -4
        t0 = -4.5
    Using the three solvers Explicit Euler, Heun Method and Exponential Euler
    with varing stepsizes (1s, 0.5s, 0.1s and 0.012s)
    
    Parameters
    ----------
    Savename : None or str
        If not Non, saves the resulting plot as savename
    '''

    # Define functions and start conditions
    f = lambda V, t: 1-V-t
    A = lambda t: -1
    B = lambda V, t: 1-t
    V0 = -4.0
    t0 = -4.5
    d = 10

    stepsize = [1, 0.5, 0.1, 0.012]

    sns.set_style('darkgrid')
    sns.set_palette('husl')
    fig, axes = plt.subplots(3, 1)
    plot_setup(15, 12)
    # Explicit Euler
    ax = axes[0]
    for dt in stepsize:
        V, t = run_solver(f, V0, t0, dt, d, solver='Explicit Euler')
        ax.plot(t, V, label= r'$\Delta t= %g s$' %dt)
    ax.legend()
    ax.set_xlabel('time in s')
    ax.set_ylabel('V in V')
    ax.set_ylim([-6, 6])
    ax.set_title('Explicit Euler')

    ax = axes[1]
    for dt in stepsize:
        V, t = run_solver(f, V0, t0, dt, d, solver='Heun Method')
        ax.plot(t, V, label= r'$\Delta t= %g s$' %dt)
    ax.legend()
    ax.set_xlabel('time in s')
    ax.set_ylim([-6, 6])
    ax.set_ylabel('V in V')
    ax.set_title('Heun Method')

    ax = axes[2]
    for dt in stepsize:
        V, t = run_solver([A, B], V0, t0, dt, d, solver='Exponential Euler')
        ax.plot(t, V, label= r'$\Delta t= %g s$' %dt)
    ax.legend()
    ax.set_ylim([-6, 6])
    ax.set_xlabel('time in s')
    ax.set_ylabel('V in V')
    ax.set_title('Exponential Euler')

    fig.tight_layout()
    if savename is not None:
        fig.savefig(savename, dpi=600)


def LIF_Model(I, d=0.05, dt=25e-6, g_leak=100e-6, V_rest=-0.06, V_th=-0.02,
              V_spike=0.02, Cm=1e-6):
    '''
    Models a Leaky Integrate and Fire Neuron

    Parameters
    ----------
    I : array
        The input current of the neuron. Has to have d/dt entries
    d : float
        Duration in s
    dt : float
        Timestep in s
    g_leak : float
        Leak conductivity in S
    V_rest : float
        Resting potential in V
    V_th : float
        Threshold potential in V
    V_spike : float
        Spiking potential in V
    Cm : float
        Membrane potential in F

    Output
    ------
    V : array
        The output voltage of the model
    t : array
        The corresponding time-vector

    Notes
    -----
    THe neuron is modeled using this equation

            V_n + dt/Cm(-g_leak(V_n-V_rest)+I_n)    if V_n < V_th
    V_n+1 = V_spike                                 if V_n = V_th
            V_rest                                  if V_n = V_spike
    '''
    t = [0]
    V = [V_rest] # start at resting potential

    for n in range(int(d/dt)-1):
        V_n = V[n]
        t_n = t[n]
        V_n1 = 0
        if V_n < V_th:
            V_n1 = V_n + dt/Cm*(-g_leak*(V_n-V_rest) + I[n])
        elif V_n == V_spike:
            V_n1 = V_rest
        elif V_n >= V_th:
            V_n1 = V_spike
        
        V.append(V_n1)
        t.append(t_n + dt)

    return np.array(V), np.array(t)


def exercise3_LeakyIntegrateAndFireNeuron(savename=None):
    '''
    Plots the LIF-Neuron for four different inputs:
        - Constant 10muA
        - Constant 20muA
        - rectified 50Hz sinus with 10muA amplitude
        - rectified 50hz sinus with 30muA amplitude
    '''
    d = 0.05
    dt = 25e-6
    n_entries = int(d/dt)
    t = np.linspace(0, d, n_entries)
    I1 = 10e-6 * np.ones(n_entries)
    I2 = 20e-6 * np.ones(n_entries)
    I3 = 10e-6 * np.abs(np.sin(50*2*np.pi*t))
    I4 = 30e-6 * np.abs(np.sin(50*2*np.pi*t))

    V1, t1 = LIF_Model(I1)
    V2, t2 = LIF_Model(I2)
    V3, t3 = LIF_Model(I3)
    V4, t4 = LIF_Model(I4)

    fig, axis = plt.subplots(2,2)
    plot_setup(15, 15)

    ax = axis[0][0]
    ax.plot(t1, V1, 'b')
    ax.set_title(r'Constant Input with $I_{max}=10\mu A$')
    ax.set_xlabel('t in s')
    ax.set_ylabel('V in V')

    ax = axis[0][1]
    ax.plot(t2, V2, 'b')
    ax.set_title(r'Constant Input with $I_{max}=20\mu A$')
    ax.set_xlabel('t in s', color='b')
    ax.set_ylabel('V in V', color='b')

    ax = axis[1][0]
    ax.plot(t3, V3, 'b')
    ax.set_title(r'50Hz Rectified Sine Input with $I_{max}=10\mu A$')
    ax.set_xlabel('t in s', color='b')
    ax.set_ylabel('V in V', color='b')
    ax2 = ax.twinx()
    ax2.plot(t3, I3*1000000, 'r')
    ax2.set_ylabel(r'$I_{input}$ in $\mu A$', color='r')

    ax = axis[1][1]
    ax.plot(t4, V4, 'b')
    ax.set_title(r'50Hz Rectified Sine Input with $I_{max}=30\mu A$')
    ax.set_xlabel('t in s', color='b')
    ax.set_ylabel('V in V', color='b')
    ax2 = ax.twinx()
    ax2.plot(t3, I4*1000000, 'r')
    ax2.set_ylabel(r'$I_{input}$ in $\mu A$', color='r')

    fig.tight_layout()
    if savename is not None:
        fig.savefig(savename, dpi=600)

