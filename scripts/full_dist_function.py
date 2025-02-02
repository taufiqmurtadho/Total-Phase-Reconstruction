#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:09:28 2025

@author: taufiqmurtadho
"""
import sys
sys.path.append('..')
import numpy as np
from classes.phase_correlation_functions import phase_correlation_functions
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt


data1 = np.load('data/extracted_com_phases_11ms_10000samples.npy', allow_pickle=True).item()
in_com_phases = data1['input_com_phases']
out_com_phases_1= data1['output_com_phases']
z_grid = data1['input_z_grid']
mean_density = data1['mean_density']
pixel_size = abs(z_grid[1]-z_grid[0])*1e6
bulk_z_grid = data1['bulk_z_grid']

data2 = np.load('data/extracted_com_phases_16ms_10000samples.npy', allow_pickle=True).item()
out_com_phases_2 = data2['output_com_phases']

in_corr_class = phase_correlation_functions(in_com_phases)
out_corr_class_1 = phase_correlation_functions(out_com_phases_1)
out_corr_class_2 = phase_correlation_functions(out_com_phases_2)

short_length = 20e-6
long_length = 65e-6

in_fdf_short = in_corr_class.calculate_full_distribution_function(z_grid, short_length)
in_fdf_long = in_corr_class.calculate_full_distribution_function(z_grid,  long_length)

out_fdf_1_short = out_corr_class_1.calculate_full_distribution_function(bulk_z_grid, short_length)
out_fdf_1_long = out_corr_class_1.calculate_full_distribution_function(bulk_z_grid, long_length)

out_fdf_2_short = out_corr_class_2.calculate_full_distribution_function(bulk_z_grid, short_length)
out_fdf_2_long = out_corr_class_2.calculate_full_distribution_function(bulk_z_grid, long_length)


#Plotting
Dred = np.array([200,43,80])/238
Dblue = np.array([33,82,135])/238
plt.subplot(1,2,1)
plt.hist(in_fdf_short, density = True, histtype='step', bins = 15, color = 'black',
         linestyle = '--')
plt.hist(out_fdf_1_short, density = True, histtype='step', bins = 15, color = Dred)
plt.hist(out_fdf_2_short, density = True, histtype='step', bins = 15, color = Dblue)
plt.yticks([0,0.5,1,1.5], fontsize = 16)
plt.ylabel(r'$P(\xi_+)$', fontsize = 18)
plt.xticks([0,0.5,1,1.5], fontsize = 16)
plt.xlabel(r'$\xi_+$', fontsize = 18)
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{a}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')

plt.subplot(1,2,2)
plt.hist(in_fdf_long, density = True, histtype='step', bins = 15, color = 'black',
         linestyle = '--')
plt.hist(out_fdf_1_long, density = True, histtype='step', bins = 15, color= Dred)
plt.hist(out_fdf_2_long, density = True, histtype='step', bins = 15, color=Dblue)
plt.yticks([0,0.4,0.8], fontsize = 16)
plt.xticks([0,1,2,3], fontsize = 16)
plt.xlabel(r'$\xi_+$', fontsize = 18)
ax = plt.gca()
ax.text(0.95, 0.95, r'$\mathbf{b}$', transform=ax.transAxes, 
            fontsize=16, va='top', ha='right', color = 'black')

plt.gcf().set_size_inches([8,4])
plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.subplots_adjust(wspace=0.32)
#plt.savefig('main_figs/fdf_plot.pdf', format='pdf', dpi=1200)