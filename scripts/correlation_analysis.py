#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 06:16:29 2025

@author: taufiqmurtadho
"""
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from classes.phase_correlation_functions import phase_correlation_functions
from plotting_funcs.fast_cmap import fast_cmap
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#Physical parameters in SI unit
atomic_mass_m = 86.9091835*1.66054*(10**(-27)) #mass of Rb-87 (kg)
hbar = 1.054571817*(10**(-34))     #Reduced Planck constant (SI)
kB = 1.380649*(10**(-23))          #Boltzmann constant

data1 = np.load('data/extracted_com_phases_11ms.npy', allow_pickle=True).item()
in_com_phases = data1['input_com_phases']
out_com_phases_1= data1['output_com_phases']
z_grid = data1['input_z_grid']
mean_density = data1['mean_density']
pixel_size = abs(z_grid[1]-z_grid[0])*1e6
bulk_z_grid = data1['bulk_z_grid']

data2 = np.load('data/extracted_com_phases_16ms.npy', allow_pickle=True).item()
out_com_phases_2 = data2['output_com_phases']

in_corr_class = phase_correlation_functions(in_com_phases)
out_corr_class_1 = phase_correlation_functions(out_com_phases_1)
out_corr_class_2 = phase_correlation_functions(out_com_phases_2)

#Vertex correlation functions

in_vertex = in_corr_class.calculate_vertex_corr_func_1d()
out_vertex_1 = out_corr_class_1.calculate_vertex_corr_func_1d()
out_vertex_2 = out_corr_class_2.calculate_vertex_corr_func_1d()

in_vertex_slice = in_corr_class.calculate_vertex_corr_func_slice_1d(50)
vertex_slice_1 = out_corr_class_1.calculate_vertex_corr_func_slice_1d(40)
vertex_slice_2 = out_corr_class_2.calculate_vertex_corr_func_slice_1d(40)

fit_idx_start = 3
fit_idx_end = 15

log_in_vertex_fit = np.log(in_vertex[fit_idx_start:fit_idx_end+1])
log_out_vertex_1_fit = np.log(out_vertex_1[fit_idx_start:fit_idx_end+1])
log_out_vertex_2_fit = np.log(out_vertex_2[fit_idx_start:fit_idx_end+1])

distance_z_fit = np.arange(fit_idx_start, fit_idx_end, pixel_size)

m_in, b_in = np.polyfit(distance_z_fit, log_in_vertex_fit, 1)
m_1, b_1 = np.polyfit(distance_z_fit, log_out_vertex_1_fit, 1)
m_2, b_2 = np.polyfit(distance_z_fit, log_out_vertex_2_fit, 1)

coh_length_in = (-1/m_in)*1e-6
coh_length_1 = (-1/m_1)*1e-6
coh_length_2 = (-1/m_2)*1e-6

T_in = (hbar**2)*mean_density/(atomic_mass_m*kB*coh_length_in)*1e9
T_1 = (hbar**2)*mean_density/(atomic_mass_m*kB*coh_length_1)*1e9
T_2 = (hbar**2)*mean_density/(atomic_mass_m*kB*coh_length_2)*1e9



#Two-point function
in_tp_func = in_corr_class.calculate_two_point_function()
#out_tp_func_1 = out_corr_class_1.calculate_two_point_function()
out_tp_func_2 = out_corr_class_2.calculate_two_point_function()

#Plotting
Dred = np.array([200,43,80])/238
Dblue = np.array([33,82,135])/238
bulk_z_grid_n = bulk_z_grid*1e6

plt.subplot(2,2,1)
plt.plot(bulk_z_grid_n, in_vertex_slice[10:90], color = 'black', linestyle = '--')
plt.plot(bulk_z_grid_n, vertex_slice_1, 'x', markersize = 4, color = Dred)
plt.plot(bulk_z_grid_n, vertex_slice_2, '^', markersize = 4, color = Dblue)
plt.ylabel(r'$C_+(z,0)$', fontsize = 18)
plt.xlabel(r'$z\;\rm (\mu m)$', fontsize = 18)
plt.yticks([0,0.5,1], fontsize = 14)
plt.xticks([-40, -20, 0, 20, 40], fontsize=16)
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{a}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')

plt.subplot(2,2,2)
plt.plot(distance_z_fit, log_in_vertex_fit, 'o', color = 'black')
plt.plot(distance_z_fit, m_in*distance_z_fit+b_in, color = 'black', linestyle = '--')
plt.plot(distance_z_fit, log_out_vertex_1_fit, 'x', color = Dred)
plt.plot(distance_z_fit, m_1*distance_z_fit+b_1, color = Dred)
plt.plot(distance_z_fit, log_out_vertex_2_fit, '^', color = Dblue)
plt.plot(distance_z_fit, m_2*distance_z_fit+b_2, color = Dblue)
plt.ylabel(r'$\ln C_+$', fontsize = 18)
plt.xlabel(r'$\Delta z\;\rm (\mu m)$', fontsize = 18)
plt.yticks([-1.5,-1,-0.5, 0], fontsize = 16)
plt.xticks([3,6,9,12,15], fontsize = 16)
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{b}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')


# Third subplot
plt.subplot(2, 2, 3)
mesh3 = plt.pcolormesh(bulk_z_grid_n, bulk_z_grid_n, in_tp_func[10:90, 10:90], cmap=fast_cmap,
                       rasterized = True)
plt.clim(0, 8)
plt.xticks([-40, -20, 0, 20, 40], fontsize=16)
plt.yticks([-40, -20, 0, 20, 40], fontsize=16)
plt.xlabel(r'$z\; \rm (\mu m)$', fontsize=18)
plt.ylabel(r'$z^\prime\; \rm (\mu m)$', fontsize=18)
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{c}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')

# Fourth subplot
plt.subplot(2, 2, 4)
mesh4 = plt.pcolormesh(bulk_z_grid_n, bulk_z_grid_n, out_tp_func_2, cmap=fast_cmap,
                       rasterized = True)
plt.yticks([])
plt.clim(0, 8)
plt.xticks([-40, -20, 0, 20, 40], fontsize=16)
plt.xlabel(r'$z\; \rm (\mu m)$', fontsize=18)
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{d}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')

# Add a horizontal colorbar below subplots 3 and 4
cbar_ax = plt.gcf().add_axes([0.35, 0.15, 0.4, 0.02])  # Adjust position and size as needed
cbar = plt.colorbar(mesh3, cax=cbar_ax, orientation='horizontal', ticks = [0,2,4,6, 8])
cbar.set_label(label=r'$G_+(z,z^\prime)$',fontsize = 18, labelpad= 10)
cbar.ax.tick_params(labelsize=16)
cbar.ax.xaxis.set_label_position('top')
plt.tight_layout(pad = 0.01, rect=[0, 0.12, 1, 1])  # Adjust the layout to make space for the colorbar
plt.gcf().set_size_inches(7, 7)
plt.subplots_adjust(hspace=0.32)

#plt.savefig('main_figs/numerical_results_corr.pdf', format='pdf', dpi=1200)
