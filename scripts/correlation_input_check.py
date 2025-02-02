#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 11:04:09 2025

@author: taufiqmurtadho
"""

import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from classes.thermal_fluctuations_sampling import thermal_fluctuations_sampling
from classes.phase_correlation_functions import phase_correlation_functions
#from plotting_funcs.fast_cmap import fast_cmap
from scipy.optimize import curve_fit

#Physical parameters in SI unit
atomic_mass_m = 86.9091835*1.66054*(10**(-27)) #mass of Rb-87 (kg)
hbar = 1.054571817*(10**(-34))     #Reduced Planck constant (SI)
kB = 1.380649*(10**(-23))          #Boltzmann constant


#Setting up the parameters
temperature = 50e-9
mean_density = 30e6
condensate_length = 100e-6
pixel_size = 1e-6
midpoint = round(condensate_length/(2*pixel_size))
fourier_cutoff = 50
z_grid = np.arange(-condensate_length/2, condensate_length/2, pixel_size)

#Defining fitting function
def exponential_func(x,A, Lambda_T):
    return A*np.exp(-abs(x)/Lambda_T)

#Initiate sampling
sampling_class = thermal_fluctuations_sampling(mean_density, temperature)
density_flucts, phase_flucts = sampling_class.generate_fluct_samples(fourier_cutoff, z_grid, 
                                                                     num_samples = 1000)

#Calculating vertex correlation function
correlation_class = phase_correlation_functions(phase_flucts)
vertex_corr_1d = correlation_class.calculate_vertex_corr_func_1d()

distance_z = np.arange(0, condensate_length/2, pixel_size)*1e6

pixnum_fit = 20
vertex_corr_1d_fit = vertex_corr_1d[0:pixnum_fit]
distance_z_fit = distance_z[0:pixnum_fit]

popt, pcov = curve_fit(exponential_func, distance_z_fit, vertex_corr_1d_fit, p0 = [1,10])

LambdaT = popt[1]*1e-6

T = 2*(hbar**2)*mean_density*1e9/(atomic_mass_m*kB*LambdaT)
plt.plot(distance_z_fit, vertex_corr_1d_fit, 'o')
plt.plot(distance_z_fit, exponential_func(distance_z_fit, popt[0], popt[1]))