# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 07:58:16 2018

@author: Nico Hertel
"""
import matplotlib
import numpy as np

def plot_setup(width=None, height=None):
    if width is None and height is None:
        width = 3.39
        height = width * (np.sqrt(5)-1.0)/2
    elif height is None:
        height = width * (np.sqrt(5)-1.0)/2
    elif width is None:
        width = height / (np.sqrt(5)-1.0)/2

    params = {'axes.labelsize': 16,
              'axes.titlesize': 22,
              'text.fontsize': 16,
              'legend.fontsize': 14,
              'xtick.labelsize': 14,
              'ytick.labelsize':14,
              'figure.figsize': [width, height]}
    matplotlib.rcParams.update(params)