#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 08:43:27 2025

@author: taufiqmurtadho
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from classes.fdm_poisson_1d_solver import fdm_poisson_1d_solver
from classes.phase_correlation_functions import phase_correlation_functions
from plotting_funcs.fast_cmap import fast_cmap
import matplotlib
#from matplotlib.ticker import ScalarFormatter
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#Physical parameters in SI unit
atomic_mass_m = 86.9091835*1.66054*(10**(-27)) #mass of Rb-87 (kg)
hbar = 1.054571817*(10**(-34))     #Reduced Planck constant (SI)
expansion_time = 15.6e-3
kB = 1.380649*(10**(-23))          #Boltzmann constant
lt = np.sqrt(hbar*expansion_time/(2*atomic_mass_m))*1e6

data = np.load('recurrence_data/scan5100_recurrence.npy', allow_pickle=True).item()

density_ripples_all = data['density_ripples_all']*1e-6
evol_time = data['evol_time']
z_grid = data['z_grid']*1e6

length = z_grid[-1] - z_grid[0]
z_grid = z_grid - z_grid[0] - length/2
pixel_size = abs(z_grid[1] - z_grid[0])

ext_phases_all = []
for i in range(len(evol_time)):
    dr_set = density_ripples_all[i]
    mean_dr = np.mean(dr_set, 0)
    dr_set = np.array([(sum(mean_dr)/sum(dr))*dr for dr in dr_set])
    sources = np.array([(1-dr/mean_dr)/(lt**2) for dr in dr_set])
    ext_phases = np.zeros(np.shape(sources))

    num_shots = np.shape(sources)[0]
    for j in range(num_shots):
        extraction_class = fdm_poisson_1d_solver(sources[j], z_grid)
        ext_phases[j,:] = extraction_class.solve_poisson()
    ext_phases_all.append(ext_phases)
    

correlation_class = phase_correlation_functions(ext_phases_all[2])
full_four_point_func = correlation_class.calculate_four_point_function(z_grid, -10, 10)
wick_four_point_func = correlation_class.calculate_disconnected_four_point_function(z_grid, -10, 10)

plt.subplot(1,2,1)
plt.pcolormesh(full_four_point_func, cmap = fast_cmap)
plt.colorbar()
plt.clim([-20,20])

plt.subplot(1,2,2)
plt.pcolormesh(wick_four_point_func, cmap = fast_cmap)
plt.colorbar()
plt.clim([-20,20])

plt.gcf().set_size_inches(5, 2)
