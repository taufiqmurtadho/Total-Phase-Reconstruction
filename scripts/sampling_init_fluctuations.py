#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:28:04 2025

@author: taufiqmurtadho
"""

import sys
sys.path.append('..')
import numpy as np
from classes.thermal_fluctuations_sampling import thermal_fluctuations_sampling

mean_density = 75e6
temperature_com =  50e-9
temperature_rel = 30e-9
condensate_length = 100e-6
pixel_size = 1e-6
z_grid = np.arange(-condensate_length/2, condensate_length/2+pixel_size, pixel_size)
fourier_cutoff = 50
num_shots = 10000

#sampling class
sampling_class_com = thermal_fluctuations_sampling(mean_density, temperature_com, 
                                               scaling_factor = np.sqrt(2))
sampling_class_rel = thermal_fluctuations_sampling(mean_density, temperature_rel, 
                                               scaling_factor = np.sqrt(2))

density_flucts_com, phase_flucts_com = sampling_class_com.generate_fluct_samples(fourier_cutoff, z_grid,
                                                                     num_samples=num_shots)

density_flucts_rel, phase_flucts_rel = sampling_class_rel.generate_fluct_samples(fourier_cutoff, z_grid,
                                                                     num_samples=num_shots)

data = {'mean_density': mean_density, 'temperature_com': temperature_com, 
        'temperature_rel':temperature_rel, 'cutoff': fourier_cutoff, 'z_grid':z_grid, 
        'density_flucts_rel':density_flucts_rel, 'phase_flucts_rel':phase_flucts_rel,
        'density_flucts_com':density_flucts_com, 'phase_flucts_com':phase_flucts_com}

#np.save('data/fluctuations_thermal_10000_samples.npy', data)
        