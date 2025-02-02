#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:53:45 2025

@author: taufiqmurtadho
"""

import numpy as np
from scipy.optimize import curve_fit


class relative_phase_extraction:
    def __init__(self, interference_pattern_tof, input_z_grid, input_x_grid, expansion_time, 
                 separation_distance = 2e-6, omega_perp = 2*np.pi*1400, length_scale_factor = 1e6):
        
        self.length_scale_factor = length_scale_factor
        self.interference_pattern = interference_pattern_tof
        self.normalized_interference_pattern = self.interference_pattern/(self.length_scale_factor**2)
        self.dim_z = np.shape(interference_pattern_tof)[0]
        self.dim_x = np.shape(interference_pattern_tof)[1]
        self.input_x_grid = input_x_grid
        self.input_z_grid = input_z_grid
        self.Lx = abs(self.input_x_grid[-1] - self.input_x_grid[0])*self.length_scale_factor
        self.Lz = abs(self.input_z_grid[-1] - self.input_z_grid[0])*self.length_scale_factor
        self.expansion_time = expansion_time
        
        
        self.normalized_x_grid = input_x_grid*self.length_scale_factor   #expressing length in microns
        self.normalized_z_grid = input_z_grid*self.length_scale_factor   #expressing length in microns
        
        #physical constants in SI unit
        self.atomic_mass_m = 86.9091835*1.66054*(10**(-27)) #mass of Rb-87 (kg)
        self.hbar = 1.054571817*(10**(-34))     #Reduced Planck constant (SI)
        self.omega_perp = omega_perp
        self.separation_distance = separation_distance
        width = np.sqrt(self.hbar/(self.atomic_mass_m*self.omega_perp))*np.sqrt(1+(self.omega_perp*self.expansion_time)**2)
        self.width = width*self.length_scale_factor
        fringe_spacing = (2*np.pi*self.hbar*self.expansion_time)/(self.atomic_mass_m*self.separation_distance)
        self.fringe_spacing = fringe_spacing*self.length_scale_factor
    
    def init_guess(self):
        xmax = np.array([self.input_x_grid[np.argmax(self.interference_pattern[i,:])] for i in range(self.dim_z)])
        prefactor = -2*np.pi/(self.fringe_spacing*1e-6)
        return prefactor*xmax
        
    def relative_phase_fitting(self, guess_phi):
        
        def fitting_func(x, A, sigma, C, Lambda, phi):
            return A*np.exp(-x**2/(sigma**2))*(1+C*np.cos(2*np.pi*x/Lambda+phi))
        
        relative_phases = np.zeros(self.dim_z)
        for i in range(self.dim_z):
            init_guess = [max(self.interference_pattern[i,:]), self.width, 1, self.fringe_spacing,
                          guess_phi[i]]
            param_bounds = ([0, 0, 0, -np.inf, -np.pi],  # Lower bounds
                            [np.inf, self.Lx, 1, self.Lx, np.pi]) 
            popt, pcov = curve_fit(fitting_func, self.normalized_x_grid, self.interference_pattern[i,:], 
                                            p0=init_guess, bounds = param_bounds)
        
            A, sigma, C, Lambda, phi = popt
            relative_phases[i] = phi
        
        relative_phases = np.unwrap(relative_phases)
        return relative_phases