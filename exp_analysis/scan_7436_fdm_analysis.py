#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 06:16:24 2025

@author: taufiqmurtadho
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from classes.fdm_poisson_1d_solver import fdm_poisson_1d_solver
from classes.phase_correlation_functions import phase_correlation_functions
import matplotlib
from matplotlib.ticker import ScalarFormatter
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from scipy.interpolate import interp1d


#Physical parameters in SI unit
atomic_mass_m = 86.9091835*1.66054*(10**(-27)) #mass of Rb-87 (kg)
hbar = 1.054571817*(10**(-34))     #Reduced Planck constant (SI)
expansion_time = 11.2e-3
kB = 1.380649*(10**(-23))          #Boltzmann constant
lt = np.sqrt(hbar*expansion_time/(2*atomic_mass_m))*1e6

data = np.load('shaking_data/scan_7436.npy', allow_pickle=True).item()

density_ripples_all = data['density_ripples']*1e-6
evol_time = data['evol_time']


#Analyze equilibrium data
init_ripples = density_ripples_all[0]
mean_density = np.mean(init_ripples,1)

bulk_idx_start = 20
bulk_idx_end = 64
dz = 1.0492
bulk_length = dz*(bulk_idx_end-bulk_idx_start)
bulk_z_grid = np.arange(0, bulk_length, dz)

bulk_mean_density = mean_density[bulk_idx_start:bulk_idx_end]


bulk_density_ripples_all = []
sources_all = []
extracted_phases_all = []
for i in range(len(evol_time)):
    density_ripples = np.transpose(density_ripples_all[i])
    num_shots = np.shape(density_ripples)[0]
    bulk_density_ripples = [dr[bulk_idx_start:bulk_idx_end] for dr in density_ripples]
    bulk_density_ripples = np.array([(sum(bulk_mean_density)/sum(dr))*dr for dr in bulk_density_ripples])
    
    bulk_density_ripples_all.append(bulk_density_ripples)
    
    sources = np.array([(1-dr/bulk_mean_density)/(lt**2) for dr in bulk_density_ripples])
    sources_all.append(sources)
    ext_phases = np.zeros(np.shape(sources))
    for j in range(num_shots):
        ext_class = fdm_poisson_1d_solver(sources[j], bulk_z_grid)
        ext_phases[j,:] = ext_class.solve_poisson()
    extracted_phases_all.append(ext_phases)

# %%Analyzing initial thermal correlation
init_ext_com_phases = extracted_phases_all[0]
correlation_class = phase_correlation_functions(init_ext_com_phases)
vertex_1d = correlation_class.calculate_vertex_corr_func_1d()

fit_idx_start = 0
fit_idx_end = 21
log_vertex_fit = np.log(vertex_1d[fit_idx_start:fit_idx_end])
distance_z_fit = np.arange(fit_idx_start, fit_idx_end, dz)

p, cov = np.polyfit(distance_z_fit, log_vertex_fit, 1, cov = True)
sigma = np.sqrt(np.diag(cov))
m,b = p

coh_length = (-1/m)*1e-6
delta_coh_length = (sigma[0]/(m**2))*1e-6
n0 = np.mean(bulk_mean_density)*1e6

T = (hbar**2)*n0/(2*atomic_mass_m*kB*coh_length)*1e9
delta_T = T*(delta_coh_length/coh_length)

def exponential_model(x, m,b):
    coh_length = (-1/m)
    amplitude = np.exp(b)
    return amplitude*np.exp(-abs(x)/coh_length)

#10 ms, 40 ms, 70ms extracted profiles
ext_phases_10ms = extracted_phases_all[8]
ext_phases_40ms = extracted_phases_all[14]
ext_phases_70ms = extracted_phases_all[20]

#Define cosine transform
def cosine_transform(phase_profile, z_grid, fourier_cutoff):
    dz = abs(z_grid[1] - z_grid[0])
    condensate_length = abs(z_grid[-1] - z_grid[0])
    dim_z = len(z_grid)
    dct_coeffs = np.zeros(fourier_cutoff)
    for i in range(fourier_cutoff):
        coeff = 0
        for j in range(dim_z):
            coeff += phase_profile[j]*np.cos(i*np.pi*z_grid[j]/condensate_length)*dz
        dct_coeffs[i] = coeff 
    
    dct_coeffs = (2/condensate_length)*dct_coeffs
    return dct_coeffs


#Compute mean profiles
mean_profiles = [np.mean(extracted_phases_all[i], 0) for i in range(len(evol_time))]
mean_dct = [cosine_transform(mean, bulk_z_grid, 10) for mean in mean_profiles]
mean_dct = np.array(mean_dct)

mean_profile_10ms = mean_profiles[8]
mean_profile_40ms = mean_profiles[14]
mean_profile_70ms = mean_profiles[20]

#%%

#Plotting
idx_1 = 0
idx_2 = 83
length = (idx_2 - idx_1)*dz
z_grid = np.arange(0, length, dz)
z_grid = z_grid - z_grid[bulk_idx_start]
Dred = np.array([200,43,80])/238


subplot  = plt.subplot(3,2,1)
plt.plot(z_grid, init_ripples[idx_1:idx_2], color = 'lightgrey')
plt.plot(z_grid, mean_density[idx_1:idx_2], color = Dred)
plt.axvline(z_grid[bulk_idx_start], linestyle = '--', color = 'black')
plt.axvline(z_grid[bulk_idx_end], linestyle = '--', color = 'black')
plt.yticks([0,100,200], fontsize = 16)
plt.xticks([-20,0,20,40, 60], fontsize = 16)
plt.axhline(128, linewidth = 0.6, color = 'black')
plt.xlim([-20,60])
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))  # Forces scientific notation
subplot.yaxis.set_major_formatter(formatter)
subplot.yaxis.get_offset_text().set_fontsize(16)
plt.ylabel(r'$n_{\rm tof}\; \rm (\mu m^{-1})$', fontsize = 18)
plt.xlabel(r'$z\; \rm (\mu m)$', fontsize = 18)
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{a}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')


plt.subplot(3,2,2)
plt.plot(bulk_z_grid, np.transpose(init_ext_com_phases), color = 'lightgrey', linewidth = 0.6)
plt.plot(bulk_z_grid, np.mean(init_ext_com_phases, 0), color = Dred)
plt.ylim([-3.5,3.5])
plt.xlabel(r'$z\; \rm (\mu m)$', fontsize = 16)
plt.ylabel(r'$\phi_+\; \rm (rad)$', fontsize = 16)
plt.yticks([-3,0,3], fontsize = 16)
plt.xticks([0,10,20,30,40], fontsize = 16)
plt.xlim([0,45])
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{b}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')

plt.subplot(3,2,3)
plt.plot(distance_z_fit, vertex_1d[fit_idx_start:fit_idx_end], 'x', color = 'black', markersize = 4)
plt.plot(distance_z_fit, exponential_model(distance_z_fit, m, b), color = 'black')
plt.ylabel(r'$C_+$', fontsize = 18)
plt.xlabel(r'$\Delta z\; \rm (\mu m)$', fontsize = 16)
plt.yticks([0,0.5,1], fontsize = 16)
plt.xticks([0,5,10,15,20], fontsize = 16)
plt.xlim([0,20])
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{c}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')

plt.subplot(3,2,4)
plt.plot(bulk_z_grid, np.transpose(ext_phases_10ms), color = 'lightgrey', linewidth = 0.6)
plt.plot(bulk_z_grid, mean_profile_10ms, color = Dred)
plt.xlabel(r'$z\; \rm (\mu m)$', fontsize = 18)
plt.ylabel(r'$\phi_+\; \rm (rad)$', fontsize = 18)
plt.yticks([-5,0,5], fontsize = 16)
plt.xticks([0,10,20,30,40], fontsize = 16)
plt.xlim([0,45])
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{d}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')

plt.subplot(3,2,5)
plt.plot(bulk_z_grid, mean_profile_10ms, color = Dred)
plt.plot(bulk_z_grid, mean_profile_40ms, linestyle = '--', color = Dred)
plt.plot(bulk_z_grid, mean_profile_70ms, linestyle = '-.', color = Dred)
plt.xlabel(r'$z\; \rm (\mu m)$', fontsize = 18)
plt.ylabel(r'$\langle\phi_+\rangle$', fontsize = 18)
plt.yticks([-2,0,2], fontsize = 16)
plt.xticks([0,10,20,30,40], fontsize = 16)
plt.xlim([0,45])
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{e}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')

evol_time = evol_time*1e3
refined_evol_time = np.arange(evol_time[0],evol_time[-1])
interp1 = interp1d(evol_time, mean_dct[:,1], kind = 'quadratic')
interp2 = interp1d(evol_time, mean_dct[:,2], kind = 'quadratic')
interp3 = interp1d(evol_time, mean_dct[:,3], kind = 'quadratic')
plt.subplot(3,2,6)
plt.plot(evol_time, mean_dct[:,2], 'o', color = Dred, markersize = 6)
#plt.plot(evol_time, mean_dct[:,1],'x', color = 'black', linewidth = 0.6)
#plt.plot(evol_time, mean_dct[:,3], '^', color = 'black', linewidth = 0.6)

plt.plot(refined_evol_time, interp2(refined_evol_time), color = Dred)
plt.plot(refined_evol_time, interp1(refined_evol_time), color = 'grey',linestyle = '--')
plt.plot(refined_evol_time, interp3(refined_evol_time), color = 'grey',linestyle = '-.')
#plt.axhline(0, color = 'black', linewidth = 0.6)
plt.xlim([-30,100])
plt.yticks([-2,0,2], fontsize = 16)
plt.xticks([0,50,100], fontsize = 16)
plt.axvline(0, color = 'black', linewidth = 0.6)
plt.axvline(10, color = 'black', linewidth = 0.6)
plt.axvline(40, color = 'black', linewidth = 0.6)
plt.axvline(70, color = 'black', linewidth = 0.6)
plt.ylabel(r'$\langle B_k \rangle$', fontsize = 18)
plt.xlabel(r'$t\; \rm (ms)$', fontsize = 18)
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{f}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')

plt.subplots_adjust(wspace = 0.3, hspace = 0.4)
plt.gcf().set_size_inches(8, 8)
#plt.savefig('figures/scan_7436_result.pdf', format='pdf', dpi=1200)
