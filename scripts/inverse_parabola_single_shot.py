#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 07:07:46 2025

@author: taufiqmurtadho
"""
import sys
sys.path.append('..')
import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
#from plotting_funcs.fast_cmap import fast_cmap
from classes.tof_expansion_1d import tof_expansion_1d
from classes.thermal_fluctuations_sampling import thermal_fluctuations_sampling
from classes.fdm_poisson_1d_solver import fdm_poisson_1d_solver
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
#%%

#Setting gas parameters
condensate_length = 100e-6 #100 microns gas
x_span = 100e-6
pixel_size = 2e-6
gaussian_convolution_width = 3e-6
z_grid = np.arange(-condensate_length/2, condensate_length/2, pixel_size)
x_grid = np.arange(-x_span/2, x_span/2, pixel_size)

#%%
#sampling fluctuation
temperature_T1 = 50e-9 #20 nK
temperature_T2 = 30e-9


peak_density = 75e6
mean_density = np.array([peak_density*(1-(2*z/(1.2*condensate_length))**2) for z in z_grid])
fourier_cutoff = int(condensate_length/(2*pixel_size))
sampling_class_1 = thermal_fluctuations_sampling(peak_density, temperature_T1)
sampling_class_2 = thermal_fluctuations_sampling(peak_density, temperature_T1)
density_fluct_1, phase_fluct_1 = sampling_class_1.generate_fluct_samples(fourier_cutoff, z_grid)
density_fluct_2, phase_fluct_2 = sampling_class_2.generate_fluct_samples(fourier_cutoff, z_grid)

com_phase_in = phase_fluct_1[0] + phase_fluct_2[0]
rel_phase_in = phase_fluct_2[0] - phase_fluct_1[0]

#Computing the fields
field_1 = np.sqrt(mean_density+density_fluct_1[0])*np.exp(1j*phase_fluct_1[0])
field_2 = np.sqrt(mean_density+density_fluct_2[0])*np.exp(1j*phase_fluct_2[0])

#%%
#TOF simulation
t_tof = 16e-3
expansion_class = tof_expansion_1d([field_1, field_2], z_grid, x_grid, t_tof)
density = expansion_class.calc_evolved_density()
density_convolved = gaussian_filter(density, gaussian_convolution_width/pixel_size)
lt = expansion_class.expansion_length_scale*1e6


#%%
#Extracting common phase from density ripple
density_ripple = np.sum(density, axis = 1)*pixel_size
density_ripple_convolved = np.sum(density_convolved, axis = 1)*pixel_size


#Defining bulk region
bulk_start = -40e-6
bulk_end = 40e-6
bulk_idx_start = np.argmin(abs(z_grid-bulk_start))
bulk_idx_end = np.argmin(abs(z_grid-bulk_end))


density_ripple_bulk = density_ripple[bulk_idx_start:bulk_idx_end]
density_ripple_convolved_bulk = density_ripple_convolved[bulk_idx_start:bulk_idx_end]

mean_density_bulk = mean_density[bulk_idx_start:bulk_idx_end]
bulk_z_grid = z_grid[bulk_idx_start:bulk_idx_end]*1e6

source= (1 - density_ripple_bulk/(2*mean_density_bulk))/(lt**2)
source_convolved = (1-density_ripple_convolved_bulk/(2*mean_density_bulk))/(lt**2)

com_extraction_class = fdm_poisson_1d_solver(source, bulk_z_grid)
com_extraction_class_convolved = fdm_poisson_1d_solver(source_convolved, bulk_z_grid)

com_phase = com_extraction_class.solve_poisson()
com_phase_convolved = com_extraction_class_convolved.solve_poisson()


#%%
#Plotting
Dred = np.array([151,43,80])/238
Dblue = np.array([33,82,135])/238
z_grid_microns = z_grid*1e6
plt.subplot(1,2,1)
plt.plot(z_grid_microns, density_ripple*1e-6, color = Dblue)
plt.plot(z_grid_microns, density_ripple_convolved*1e-6, color = Dred)
plt.plot(z_grid_microns, 2*mean_density*1e-6, color = 'black', linestyle = '-.')
plt.xlabel(r'$z\; \rm \mu m$', fontsize = 14)
plt.xticks([-50,-25,0,25, 50], fontsize = 12)
plt.yticks([50,150,250], fontsize = 14)
plt.ylabel(r'$n_{\rm tof}\; \rm (\mu m^{-1})$', fontsize = 14)
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{a}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')

plt.subplot(1,2,2)
plt.plot(bulk_z_grid, com_phase, color = Dblue)
plt.plot(bulk_z_grid, com_phase_convolved, color = Dred)
plt.plot(z_grid_microns, com_phase_in, color = 'black', linestyle = '-.')
plt.xlabel(r'$z\; \rm \mu m$', fontsize = 14)
plt.ylabel(r'$\phi_+\; \rm (rad)$', fontsize = 14)
plt.xticks([-50,-25,0,25, 50], fontsize = 12)
plt.yticks([-3,0,3], fontsize = 12)
plt.subplots_adjust(wspace=0.3)
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{b}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')

plt.gcf().set_size_inches(8, 3)
plt.tight_layout(pad = 0.1)
#plt.savefig('main_figs/apdx_inverse_parabola_convolution.pdf', format='pdf', dpi=1200)
