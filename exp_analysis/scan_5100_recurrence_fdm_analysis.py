#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:12:45 2025

@author: taufiqmurtadho
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

example = np.load('recurrence_data/interference_example.npy', allow_pickle=True). item()
ex_z_grid = example['z_grid']
ex_x_grid = example['x_grid']
rho_tof = example['rho_tof']
n_tof = example['density_ripple']


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

def exponential_model(x,A,Lambda):
    return A*np.exp(-x/Lambda)

coh_lengths = []
delta_coh_lengths = []
vertex_corr = []

for i in range(len(evol_time)):
    correlation_class = phase_correlation_functions(ext_phases_all[i])
    vertex_1d = correlation_class.calculate_vertex_corr_func_1d()
    vertex_corr.append(vertex_1d)
    fit_idx_num = 15
    distance_z = np.arange(0, fit_idx_num*pixel_size, pixel_size)

    param, cv = curve_fit(exponential_model, distance_z, vertex_1d[0:fit_idx_num])
    coh_lengths.append(param[1])
    delta_coh_lengths.append(np.diag(cv)[1])


#%%
#Plotting

Dred = np.array([200,43,80])/238
Dblue = np.array([33,82,135])/238

plt.subplot(2,2,1)
plt.pcolormesh(ex_x_grid, ex_z_grid, rho_tof, cmap = fast_cmap, rasterized = True)
plt.xticks([-60,-30,0,30,60], fontsize = 18)
plt.yticks([-40,-20,0, 20,40], fontsize = 18)
plt.xlabel(r'$x\; \rm (\mu m)$', fontsize = 20)
plt.ylabel(r'$z\; \rm (\mu m)$', fontsize = 20)
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{a}$', transform=ax.transAxes, 
            fontsize=18, va='top', ha='right', color = 'white')

plt.subplot(2,2,2)
plt.plot(n_tof, ex_z_grid, color = Dred)
plt.plot(np.mean(density_ripples_all[0], 0), z_grid, linestyle ='--', color='black')
plt.yticks([])
plt.xticks([50,100,150], fontsize = 18)
plt.xlabel(r'$n_{\rm tof}\; \rm (\mu m^{-1})$', fontsize = 20)
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{b}$', transform=ax.transAxes, 
            fontsize=18, va='top', ha='right', color = 'black')


plt.subplot(2,2,3)
plt.plot(distance_z, vertex_corr[1][0:fit_idx_num], 'o', color = 'green', markersize = 6)
plt.plot(distance_z,vertex_corr[3][0:fit_idx_num], 'x', color = Dred, markersize = 8)
plt.plot(distance_z,vertex_corr[5][0:fit_idx_num], '^', color = Dblue)
plt.plot(distance_z, vertex_corr[7][0:fit_idx_num], 's', color = 'orange', markersize = 5, fillstyle = 'none')
plt.legend([r'$t = 0\; \rm ms$', r'$t = 6\; \rm ms$', r'$t = 12\; \rm ms$', r'$t = 18\; \rm ms$'],
           fontsize = 12, loc = 'lower left')
plt.ylabel(r'$C_+(\Delta z, t)$', fontsize = 20)
plt.xlabel(r'$\Delta z\; \rm (\mu m)$',fontsize =20)
plt.xticks([0,5,10,15,20,25], fontsize = 18)
plt.yticks([0,0.5,1], fontsize = 18 )
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{c}$', transform=ax.transAxes, 
            fontsize=18, va='top', ha='right', color = 'black')

plt.subplot(2,2,4)
plt.axvline(0, color = 'black', linestyle='--')
plt.axhline(11, color = 'black', linestyle = '-.')
plt.errorbar(evol_time, coh_lengths, yerr=delta_coh_lengths, linestyle = 'none', marker = 'o',
             markersize = 6, capsize =4, color = Dred)
plt.ylabel(r'$\lambda_{+}\; \rm (\mu m)$', fontsize = 20)
plt.xlabel(r'$t\; \rm (ms)$', fontsize = 20)
plt.xticks([0,6,12,18], fontsize = 18)
plt.yticks([10, 11,12,13], fontsize = 18)
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{d}$', transform=ax.transAxes, 
            fontsize=18, va='top', ha='right', color = 'black')

plt.subplots_adjust(wspace = 0.25, hspace = 0.3)
plt.gcf().set_size_inches(10,8.5)

plt.savefig('figures/scan_5100_fdm_result.pdf', format='pdf', dpi=1200)
    

