#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 05:56:30 2025

@author: taufiqmurtadho
"""

import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
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

#Plotting
z_grid_microns = z_grid*1e6
bulk_z_grid_microns = bulk_z_grid*1e6

idx_1 = 1
idx_2 = 18
idx_3 = 42
idx_4 = 68
idx_5 = 120
idx_6 = 238
idx_7 = 368
idx_8 = 178
idx_9 = 254

Dred = np.array([200,43,80])/238
Dblue = np.array([33,82,135])/238
plt.subplot(3,3,1)
plt.plot(bulk_z_grid_microns, out_com_phases_1[idx_1], color = Dblue)
plt.plot(bulk_z_grid_microns, out_com_phases_2[idx_1], color = Dred)
plt.plot(z_grid_microns, in_com_phases[idx_1], linestyle = '--', color = 'black')
plt.ylim([-3.5,3.5])
plt.xticks([])
plt.yticks([-2,0,2], fontsize = 14)
plt.ylabel(r'$\phi_+\; \rm (rad)$',fontsize =16)



plt.subplot(3,3,2)
plt.plot(bulk_z_grid_microns, out_com_phases_1[idx_2], color = Dblue)
plt.plot(bulk_z_grid_microns, out_com_phases_2[idx_2], color = Dred)
plt.plot(z_grid_microns, in_com_phases[idx_2], linestyle = '--', color = 'black')
plt.ylim([-3.5,3.5])
plt.xticks([])
plt.yticks([])


plt.subplot(3,3,3)
plt.plot(bulk_z_grid_microns, out_com_phases_1[idx_3], color = Dblue)
plt.plot(bulk_z_grid_microns, out_com_phases_2[idx_3], color = Dred)
plt.plot(z_grid_microns, in_com_phases[idx_3], linestyle = '--', color = 'black')
plt.ylim([-3.5,3.5])
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,4)
plt.plot(bulk_z_grid_microns, out_com_phases_1[idx_4], color = Dblue)
plt.plot(bulk_z_grid_microns, out_com_phases_2[idx_4], color = Dred)
plt.plot(z_grid_microns, in_com_phases[idx_4], linestyle = '--', color = 'black')
plt.ylim([-3.5,3.5])
plt.xticks([])
plt.yticks([-2,0,2], fontsize = 14)
plt.ylabel(r'$\phi_+\; \rm (rad)$', fontsize = 16)

plt.subplot(3,3,5)
plt.plot(bulk_z_grid_microns, out_com_phases_1[idx_5], color = Dblue)
plt.plot(bulk_z_grid_microns, out_com_phases_2[idx_5], color = Dred)
plt.plot(z_grid_microns, in_com_phases[idx_5], linestyle = '--', color = 'black')
plt.xticks([])
plt.yticks([])
plt.ylim([-3.5,3.5])

plt.subplot(3,3,6)
plt.plot(bulk_z_grid_microns, out_com_phases_1[idx_6], color = Dblue)
plt.plot(bulk_z_grid_microns, out_com_phases_2[idx_6], color = Dred)
plt.plot(z_grid_microns, in_com_phases[idx_6], linestyle = '--', color = 'black')
plt.yticks([])
plt.xticks([])
plt.ylim([-3.5,3.5])


plt.subplot(3,3,7)
plt.plot(bulk_z_grid_microns, out_com_phases_1[idx_7], color = Dblue)
plt.plot(bulk_z_grid_microns, out_com_phases_2[idx_7], color = Dred)
plt.plot(z_grid_microns, in_com_phases[idx_7], linestyle = '--', color = 'black')
plt.ylim([-3.5,3.5])
plt.xticks([-50,0,50], fontsize = 14)
plt.xlabel(r'$z\; \rm (\mu m)$', fontsize = 16)
plt.ylabel(r'$\phi_+\; \rm (rad)$', fontsize = 16)
plt.yticks([-2,0,2], fontsize = 14)

plt.subplot(3,3,8)
plt.plot(bulk_z_grid_microns, out_com_phases_1[idx_8], color = Dblue)
plt.plot(bulk_z_grid_microns, out_com_phases_2[idx_8], color = Dred)
plt.plot(z_grid_microns, in_com_phases[idx_8], linestyle = '--', color = 'black')
plt.yticks([])
plt.ylim([-3.5,3.5])
plt.xticks([-50,0,50], fontsize = 14)
plt.xlabel(r'$z\; \rm (\mu m)$', fontsize = 16)

plt.subplot(3,3,9)
plt.plot(bulk_z_grid_microns, out_com_phases_1[idx_9], color = Dblue)
plt.plot(bulk_z_grid_microns, out_com_phases_2[idx_9], color = Dred)
plt.plot(z_grid_microns, in_com_phases[idx_9], linestyle = '--', color = 'black')
plt.yticks([])
plt.ylim([-3.5,3.5])
plt.xticks([-50,0,50], fontsize = 14)
plt.xlabel(r'$z\; \rm (\mu m)$', fontsize = 16)

plt.subplots_adjust(wspace=0.1)
plt.gcf().set_size_inches(9, 6)
#plt.savefig('main_figs/apdx_single_shots.pdf', format='pdf', dpi=1200)