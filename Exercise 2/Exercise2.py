# -*- coding: utf-8 -*-
"""
Created on Tue May  8 10:17:38 2018

Neuroprothetik Exercise 2

@author: Nico Hertel
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_slope_field(function, iso_funct = None, iso_val=None,
                     t=np.linspace(-5, 5, 30),
                     V=np.linspace(-5, 5, 30), savename=None,
                     title=None, text=None, ax=None):
    t_grid, V_grid = np.meshgrid(t, V)
    lng = np.sqrt(function(V_grid,  t_grid )**2+1)
    dt = 1/lng;
    dv = function(V_grid, t_grid)/lng;

    if ax==None:
        plt.figure(figsize=(10,10))
        Q = plt.quiver(t_grid, V_grid, dt, dv, cmap='inferno')
        if iso_funct is not None and iso_val is not None:
            for x in iso_val:
                x_val = iso_funct(t, x)
                plt.plot(t, x_val, label=r'Isocline for %i $\frac{v}{s}$' %x)
            plt.legend()
        plt.xlim([-5,5])
        plt.ylim([-5,5])
        if text is not None:
            plt.quiverkey(Q, 0.8, 0.9, 1, text, labelpos='W', coordinates='figure')
        if title is not None:
            plt.title(title, loc='left')
        if savename is not None:
            plt.savefig(savename, dpi=600)
            plt.show()
    else:
        Q = ax.quiver(t_grid, V_grid, dt, dv, cmap='inferno')
        if text is not None:
            ax.quiverkey(Q, 0.9, 1.02, 1, text, labelpos='W', coordinates='axes')
        if title is not None:
            ax.set_title(title, loc='left')
    


#==============================================================================
#       Plot slope fields and isocline
#==============================================================================


func1 = lambda V, t: 1-V-t
iso1 = lambda t, x: 1-t-x
func2 = lambda V, t: np.sin(t)-V/1.5
iso2 = lambda t, x: 1.5*(np.sin(t)-x)
text1 = r'$\frac{dt}{dV}=1-V-t$'
text2 = r'$\frac{dt}{dV}=sin(t)-\frac{1}{1.5}V$'
iso_val = [-2, 1, 0, 1, 2]

plot_slope_field(func1, iso_funct=iso1, iso_val=iso_val, text=text1, title='Slope Field for Function 1', savename='exercise21a.pdf')
plot_slope_field(func2, iso_funct=iso2, iso_val=iso_val, text=text2, title='Slope Field for Function 2', savename='exercise21b.pdf')

#==============================================================================
#       Differential equations of a simple cell model
#==============================================================================

sns.set_style('darkgrid')


Imax_1 = 0.0
Imax_2 = 1.0
func3 = lambda V, t: Imax_1*np.sin(t)-V
func4 = lambda V, t: Imax_2*np.sin(t)-V
func5 = lambda V, t: Imax_1*np.sin(t)-V+2.0
func6 = lambda V, t: Imax_2*np.sin(t)-V+2.0

text3 = r'$\frac{dV}{dt}=\frac{1}{C_{m}}*(I_{max}*sin(t)+D)-\frac{1}{R_{l}C_{m}}V$'
text4 = r'$\frac{dV}{dt}=\frac{1}{C_{m}}*(I_{max}*sin(t)+D)-\frac{1}{R_{l}C_{m}}V$'

fig, ax = plt.subplots(2,2, figsize=(20,20))
plot_slope_field(func3, title=r'Slope Field with: $I_{max}=0A, D=0A$', text=text3, ax=ax[0][0])
plot_slope_field(func4, title=r'Slope Field with: $I_{max}=1A, D=0A$', text=text4, ax=ax[1][0])
plot_slope_field(func5, title=r'Slope Field with: $I_{max}=0A, D=2A$', text=text3, ax=ax[0][1])
plot_slope_field(func6, title=r'Slope Field with: $I_{max}=1A, D=2A$', text=text4, ax=ax[1][1])
plt.tight_layout()
plt.savefig('.\\Exercise 2\\exercise22.pdf', dpi=600)
