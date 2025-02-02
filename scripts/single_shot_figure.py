#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:53:51 2025

@author: taufiqmurtadho
"""
import sys
sys.path.append('..')
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from plotting_funcs.fast_cmap import fast_cmap
from classes.tof_expansion_1d import tof_expansion_1d
from classes.thermal_fluctuations_sampling import thermal_fluctuations_sampling
from classes.fdm_poisson_1d_solver import fdm_poisson_1d_solver
from mpl_toolkits.mplot3d import Axes3D
#%%

#Setting gas parameters
condensate_length = 100e-6 #100 microns gas
x_span = 100e-6
pixel_size = 1e-6
z_grid = np.arange(-condensate_length/2, condensate_length/2, pixel_size)
x_grid = np.arange(-x_span/2, x_span/2, pixel_size)

#%%
#sampling fluctuation
temperature_T1 = 50e-9 #20 nK
temperature_T2 = 50e-9
mean_density = 75e6
fourier_cutoff = 20
sampling_class_1 = thermal_fluctuations_sampling(mean_density, temperature_T1)
sampling_class_2 = thermal_fluctuations_sampling(mean_density, temperature_T1)
density_fluct_1, phase_fluct_1 = sampling_class_1.generate_fluct_samples(fourier_cutoff, z_grid)
density_fluct_2, phase_fluct_2 = sampling_class_2.generate_fluct_samples(fourier_cutoff, z_grid)

com_phase_in = phase_fluct_1[0] + phase_fluct_2[0]
rel_phase_in = phase_fluct_2[0] - phase_fluct_1[0]

#Computing the fields
field_1 = np.sqrt(mean_density+density_fluct_1[0])*np.exp(1j*phase_fluct_1[0])
field_2 = np.sqrt(mean_density+density_fluct_2[0])*np.exp(1j*phase_fluct_2[0])
fields = {'field_1':field_1, 'field_2':field_2}

#np.save('data/single_shot_fields.npy', field_1, field_2)
#%%
#TOF simulation
t_tof = 11e-3
expansion_class = tof_expansion_1d([field_1, field_2], z_grid, x_grid, t_tof)
density = expansion_class.calc_evolved_density()
lt = expansion_class.expansion_length_scale*1e6

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
"""
#Plotting the interference pattern as a surface in 3D
# Create sample 2D data for the image
x, y = np.meshgrid(z_grid, z_grid)
z = np.zeros_like(x)  # Place the plane at z=0

# Normalize the density values to [0, 1] for colormap
norm_density = (density - np.min(density)) / (np.max(density) - np.min(density))

# Plot the surface (2D image as a flat plane in 3D)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, facecolors=fast_cmap(norm_density), rstride=1, cstride=1, shade=False)

# Optional: Set the view and hide axes for better visualization
ax.view_init(elev= 20, azim=-45)  # Adjust elevation and azimuth to suit your preference
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Set limits if necessary
ax.axis('off')
plt.savefig('main_figs/vertical_imaging_example.pdf', format='pdf', dpi=1200)

"""

#%%
"""
#Plotting the image in side imaging
y_grid = x_grid
sigma_y = 30e-6
y_func = np.array([np.exp(-abs(y_grid)**2/sigma_y**2)])
dr_z = np.array([density_ripple])
density_ripple_2d = density_ripple*np.transpose(y_func)
plt.yticks([])
plt.xticks([])
plt.pcolormesh(density_ripple_2d, cmap = fast_cmap, rasterized = True) 
plt.savefig('main_figs/side_imaging_example.pdf', format='pdf', dpi=1200)
"""


#%%Plotting extracted phases
Dred = np.array([200,43,80])/238
Dblue = np.array([33,82,135])/238
subplot = plt.subplot(2,1,1)
plt.plot(z_grid*1e6, density_ripple*1e-6, color = Dred)
# %%
plt.axhline(150, color = 'black', linestyle = '-.')

plt.axvline(-40, color = 'black', linestyle = '--')
plt.axvline(40, color = 'black', linestyle = '--')
plt.xticks([])
plt.yticks([50,150, 250], fontsize =20)
plt.ylabel(r'$n_{\rm tof}\; \rm (\mu m^{-1})$', fontsize = 22)
# Set scientific notation for y-axis
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))  # Forces scientific notation
subplot.yaxis.set_major_formatter(formatter)
subplot.yaxis.get_offset_text().set_fontsize(20)  

plt.subplot(2,1,2)
plt.plot(bulk_z_grid, com_phase, color = Dred)
plt.plot(z_grid*1e6, com_phase_in, color = 'black', linestyle = '-.')
plt.axvline(-40, color = 'black', linestyle = '--')
plt.axvline(40, color = 'black', linestyle = '--')
plt.ylabel(r'$\phi_+ \; (\rm rad)$', fontsize = 22)
plt.xlabel(r'$z\; \rm (\mu m)$', fontsize = 22)
plt.xticks([-40,-20,0,20,40], fontsize = 20)
plt.yticks([-2,0,2], fontsize = 20)
plt.gcf().set_size_inches(7, 6)

#plt.savefig('main_figs/single_shot_example.pdf', format='pdf', dpi=1200)