# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 07:30:34 2018

@author: Nico Hertel
"""

import numpy as np

def explicit_euler_step(f, V0, t0, dt):
    '''
    Calculates the next step of the explicid euler method

    Parameters
    ----------
    f : function
        The differential equation to process, a lambda-function depending on
        V and t
    V0 : float
        Current V-Value
    t0 : float
        Current t-Value
    dt : float
        Step size in s

    Returns
    ------
    V1 : float
        Next V-Value calculated using the explicid euler method

    Notes
    -----
    V1 is calculated like the following:
        V1 = V0 + f(V0,t0)*dt
    '''
    return V0 + f(V0, t0) * dt


def heun_step(f, V0, t0, dt):
    '''
    Calculates the next step of the heun method

    Parameters
    ----------
    f : function
        The differential equation to process, a lambda-function depending on
        V and t
    V0 : float
        Current V-Value
    t0 : float
        Current t-Value
    dt : float
        Step size in s

    Returns
    ------
    V1 : float
        Next V-Value calculated using the heun method

    Notes
    -----
    V1 is calculated like the following:
        V1 = V0 + (A+B)/2*dt
    with
        A = f(V0, t0)
        B = f(V0+A*dt, t0+dt)
    '''
    A = f(V0, t0)
    B = f(V0+A*dt, t0+dt)
    return V0 + (A+B)/2 * dt


def exponential_euler_step(A, B, V0, t0, dt):
    '''
    Calculates the next step of the exponential euler method

    Parameters
    ----------
    A : function
        The differential equation to process, a lambda-function depending on t
    A : function
        The differential equation to process, a lambda-function depending on V and t
    V0 : float
        Current V-Value
    t0 : float
        Current t-Value
    dt : float
        Step size in s

    Returns
    ------
    V1 : float
        Next V-Value calculated using the heun method

    Notes
    -----
    V1 is calculated like the following:
        V1 = V0*exp(A(t0)*dt) + B(V0, t0)/A(t0)*(exp(A(t0)*dt)-1)
    '''
    return V0*np.exp(A(t0)*dt) + B(V0, t0)/A(t0)*(np.exp(A(t0)*dt)-1)


def run_solver(f, V0, t0, dt, d, solver='Explicit Euler'):
    '''
    Solves the differential equation f using one of three solvers

    Parameters
    ----------
    f : function
        The differential equation to process, a lambda-function depending on
        V and t
    V0 : float
        Current V-Value
    t0 : float
        Current t-Value
    dt : float
        Step size in s
    d : float
        Duration of the signal in s
    solver : str
        Which solver to use, 'Explicit Euler', 'Heun Method' or 'Expolential Euler'

    Returns
    -------
    V : array
        The signal described by the differential equation f
    t : array
        The time vector to plot V

    Notes
    ----
    For the exponential euler method, the function f describing the differential
    equation has to look like this:
        dV/dt = A(t)V(t) + B(V,t)
    Where A(t) only depends on t and B(V,t) can depend on both V and t
    If using the exponential euler method, f has to be a list of functions instead
    of only a function:
        f : list [A, B]
    
    '''
    n_steps = int(d/dt) # Nuper of steps calculated
    V = [V0]
    t = [t0]

    if solver == 'Explicit Euler':
        for i in range(n_steps):
            V_next = explicit_euler_step(f, V[i], t[i], dt)
            t_next = t[i] + dt
            V.append(V_next)
            t.append(t_next)
    elif solver == 'Heun Method':
        for i in range(n_steps):
            V_next = heun_step(f, V[i], t[i], dt)
            t_next = t[i] + dt
            V.append(V_next)
            t.append(t_next)
    elif solver == 'Exponential Euler':
        if len(f) != 2:
            print('Error, for the exponential euler to work, f has to be a list of functions! See notes for more information')
            return False
        A = f[0]
        B = f[1]
        for i in range(n_steps):
            V_next = exponential_euler_step(A, B, V[i], t[i], dt)
            t_next = t[i] + dt
            V.append(V_next)
            t.append(t_next)
    
    V = np.array(V)
    t = np.array(t)
    return V, t
