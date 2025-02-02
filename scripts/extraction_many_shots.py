#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 22:52:15 2025

@author: taufiqmurtadho
"""

import sys
sys.path.append('..')
import numpy as np
from classes.tof_expansion_1d import tof_expansion_1d
from classes.fdm_poisson_1d_solver import fdm_poisson_1d_solver

data = np.load('data/fluctuations_thermal_10000_samples.npy', allow_pickle=True).item()

#Loading initial fields
mean_density = data['mean_density']
rel_density_flucts = data['density_flucts_rel']
com_density_flucts = data['density_flucts_com']
rel_phases = data['phase_flucts_rel']
com_phases = data['phase_flucts_com']
input_z_grid = data['z_grid']

num_shots = np.shape(com_phases)[0]

phases_1 = (com_phases - rel_phases)/2
phases_2 = (com_phases + rel_phases)/2
density_flucts_1 = (com_density_flucts - rel_density_flucts)/2
density_flucts_2 = (com_density_flucts + rel_density_flucts)/2

fields_1 = np.array([np.sqrt(mean_density+density_flucts_1[i,:])*np.exp(1j*phases_1[i,:]) for i in range(num_shots)])
fields_2 = np.array([np.sqrt(mean_density+density_flucts_2[i,:])*np.exp(1j*phases_2[i,:]) for i in range(num_shots)])

#Simulating TOF dynamics
expansion_time = 16e-3 
input_x_grid = input_z_grid
pixel_size = abs(input_z_grid[1] - input_z_grid[0])
#Defining bulk region
bulk_start = -40e-6
bulk_end = 40e-6
bulk_idx_start = np.argmin(abs(input_z_grid-bulk_start))
bulk_idx_end = np.argmin(abs(input_z_grid-bulk_end))
bulk_z_grid = input_z_grid[bulk_idx_start:bulk_idx_end]

extracted_com_phases = np.zeros((num_shots, len(bulk_z_grid)))
count = 0
for i in range(num_shots):
    expansion_class = tof_expansion_1d([fields_1[i], fields_2[i]], input_z_grid, input_x_grid, expansion_time)
    density_tof = expansion_class.calc_evolved_density()
    lt = expansion_class.expansion_length_scale*1e6
    density_ripple = np.sum(density_tof, 1)*pixel_size
    bulk_density_ripple = density_ripple[bulk_idx_start:bulk_idx_end]
    source = (1 - bulk_density_ripple/(2*mean_density))/(lt**2)
    
    #Solving Poisson equation
    extraction_class = fdm_poisson_1d_solver(source, bulk_z_grid*1e6)
    com_phase = extraction_class.solve_poisson()
    extracted_com_phases[i,:]=com_phase
    count = count+1
    print(count)

data = {'input_com_phases': com_phases, 'output_com_phases':extracted_com_phases, 
        'input_z_grid':input_z_grid, 'bulk_z_grid':bulk_z_grid, 'mean_density':mean_density}

#np.save('data/extracted_com_phases_16ms_10000samples.npy', data)
    
    
    
    