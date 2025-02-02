#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:53:38 2025

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
from classes.relative_phase_extraction import relative_phase_extraction
from classes.fdm_poisson_1d_solver import fdm_poisson_1d_solver
#%%

#Setting gas parameters
condensate_length = 100e-6 #100 microns gas
x_span = 100e-6
pixel_size = 1.5e-6
gaussian_convolution_width = 2.5e-6
z_grid = np.arange(-condensate_length/2, condensate_length/2, pixel_size)
x_grid = np.arange(-x_span/2, x_span/2, pixel_size)

#%%
#sampling fluctuation
temperature_T1 = 50e-9 #20 nK
temperature_T2 = 50e-9
mean_density = 30e6
fourier_cutoff = int(condensate_length/(2*np.pi*pixel_size))
sampling_class_1 = thermal_fluctuations_sampling(mean_density, temperature_T1)
sampling_class_2 = thermal_fluctuations_sampling(mean_density, temperature_T1)
density_fluct_1, phase_fluct_1 = sampling_class_1.generate_fluct_samples(fourier_cutoff, z_grid)
density_fluct_2, phase_fluct_2 = sampling_class_2.generate_fluct_samples(fourier_cutoff, z_grid)

com_phase_in = phase_fluct_1[0] + phase_fluct_2[0]
rel_phase_in = phase_fluct_2[0] - phase_fluct_1[0]

#Computing the fields
field_1 = np.sqrt(mean_density+density_fluct_1[0])*np.exp(1j*phase_fluct_1[0])
field_2 = np.sqrt(mean_density+density_fluct_2[0])*np.exp(1j*phase_fluct_2[0])

#%%
#TOF simulation
t_tof = 15e-3
expansion_class = tof_expansion_1d([field_1, field_2], z_grid, x_grid, t_tof)
density = expansion_class.calc_evolved_density()
density = gaussian_filter(density, gaussian_convolution_width/pixel_size)
lt = expansion_class.expansion_length_scale*1e6

#%%
#Extracting relative phase
rel_extraction_class = relative_phase_extraction(density, z_grid, x_grid, t_tof)
init_guess = rel_extraction_class.init_guess()
rel_phase = rel_extraction_class.relative_phase_fitting(init_guess)

#%%
#Extracting common phase from density ripple
density_ripple = np.sum(density, axis = 1)*pixel_size

#Defining bulk region
bulk_start = -40e-6
bulk_end = 40e-6
bulk_idx_start = np.argmin(abs(z_grid-bulk_start))
bulk_idx_end = np.argmin(abs(z_grid-bulk_end))


density_ripple_bulk = density_ripple[bulk_idx_start:bulk_idx_end]
bulk_z_grid = z_grid[bulk_idx_start:bulk_idx_end]*1e6
source= (1 - density_ripple_bulk/(2*mean_density))/(lt**2)

com_extraction_class = fdm_poisson_1d_solver(source, bulk_z_grid)
com_phase = com_extraction_class.solve_poisson()

#%%
#Calculating individual phases
rel_phase_bulk = rel_phase[bulk_idx_start:bulk_idx_end]
phi_1 = (com_phase - rel_phase_bulk)/2
phi_2 = (com_phase + rel_phase_bulk)/2


#%%
#Plotting the extracted phases

Dred = np.array([151,43,80])/238
Dblue = np.array([33,82,135])/238
z_grid = z_grid*1e6
x_grid = x_grid*1e6

ax1 = plt.subplot(3, 2, 1)
pcol = ax1.pcolormesh(x_grid, z_grid, np.transpose(density * 1e-12))

# Manually add colorbar above the first subplot
cbar_ax = plt.gcf().add_axes([ax1.get_position().x0, ax1.get_position().y1 + 0.02,
                              ax1.get_position().width, 0.02])
plt.colorbar(pcol, cax=cbar_ax, orientation='horizontal', label=r'$\rho_{\rm tof} \; (\mu m^{-2})$')
cbar_ax.xaxis.set_ticks_position('top')
cbar_ax.xaxis.set_label_position('top')
ax1.set_xticks([])
ax1.set_ylabel(r'$x\; \rm (\mu m)$')

plt.subplot(3,2,2)
plt.plot(z_grid, density_ripple*1e-6, color = 'black')
plt.xticks([])
plt.ylabel(r'$n_{\rm tof}\; \rm (\mu m^{-1})$')
plt.axhline(60, linestyle = '-.', color = 'black')
plt.axvline(-40, linestyle = '--', color = 'black', linewidth = 0.8)
plt.axvline(40, linestyle = '--', color = 'black', linewidth = 0.8)
plt.title('Density Ripple (15 ms)')

plt.subplot(3,2,3)
plt.plot(z_grid, rel_phase, 'o', markersize = 4, color = Dred)
plt.plot(z_grid, rel_phase_in, color = Dblue)
plt.xticks([])
plt.axvline(-40, linestyle = '--', color = 'black', linewidth = 0.8)
plt.axvline(40, linestyle = '--', color = 'black', linewidth = 0.8)
plt.ylabel(r'$\phi_-\; \rm (rad)$')
plt.title('Relative Phase')
plt.legend(['Extracted', 'Input'], loc = 'lower right')

plt.subplot(3,2,4)
plt.plot(bulk_z_grid, com_phase, 'o', markersize = 4, color = Dred)
plt.plot(z_grid, com_phase_in, color = Dblue)
plt.ylabel(r'$\phi_+\; \rm (rad)$')
plt.xticks([])
plt.axvline(-40, linestyle = '--', color = 'black', linewidth = 0.8)
plt.axvline(40, linestyle = '--', color = 'black', linewidth = 0.8)
plt.title('Common Phase')

plt.subplot(3,2,5)
plt.plot(bulk_z_grid, phi_1, 'o', markersize = 4, color = Dred)
plt.plot(z_grid, phase_fluct_1[0], color = Dblue)
plt.ylabel(r'$\phi_1\; (\rm rad)$')
plt.xlabel(r'$z\; \rm (\mu m)$')
plt.axvline(-40, linestyle = '--', color = 'black', linewidth = 0.8)
plt.axvline(40, linestyle = '--', color = 'black', linewidth = 0.8)
plt.title('Phase 1')

plt.subplot(3,2,6)
plt.plot(bulk_z_grid, phi_2, 'o', markersize = 4, color = Dred)
plt.plot(z_grid, phase_fluct_2[0], color =Dblue)
plt.ylabel(r'$\phi_2\; (\rm rad)$')
plt.xlabel(r'$z\; \rm (\mu m)$')
plt.axvline(-40, linestyle = '--', color = 'black', linewidth = 0.8)
plt.axvline(40, linestyle = '--', color = 'black', linewidth = 0.8)
plt.title('Phase 2')

plt.gcf().set_size_inches(10, 8)
#plt.savefig('phases_extraction_2.png', dpi=300, bbox_inches='tight')

