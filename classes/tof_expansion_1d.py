#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 23 Jan 16:25:15 2024

@author: taufiqmurtadho
"""
import numpy as np
from numpy.fft import fft, ifft, fftfreq

class tof_expansion_1d:
    def __init__(self, init_fields_1d, input_z_grid, input_x_grid, expansion_time, 
                 separation_distance = 2e-6, omega_perp = 1400*2*np.pi):
        #Initialization
        #Input grids
        self.input_z_grid = input_z_grid
        self.input_x_grid = input_x_grid
        self.condensate_length = abs(self.input_z_grid[-1] - self.input_z_grid[0])
        self.dz = abs(self.input_z_grid[1] - self.input_z_grid[0])
        self.dx = abs(self.input_x_grid[1] - self.input_x_grid[0])
        self.dim_z = len(self.input_z_grid)
        self.dim_x = len(self.input_x_grid)
        #Input fields
        self.init_fields = init_fields_1d
        self.condensate_num = len(init_fields_1d)
        if self.condensate_num == 1:
            self.init_field_1 = init_fields_1d[0]
            self.init_field_2 = None
            self.total_atom_number = (np.abs(self.init_field_1)**2)*self.condensate_length
        elif self.condensate_num == 2:
            self.init_field_1 = init_fields_1d[0]
            self.init_field_2 = init_fields_1d[1]
            self.total_atom_number = sum(np.abs(self.init_field_1)**2 + np.abs(self.init_field_2)**2)*self.dz 
        
        #TOF parameters & physical constants in SI unit
        self.expansion_time = expansion_time
        self.atomic_mass_m = 86.9091835*1.66054*(10**(-27))  #mass of Rb-87
        self.hbar = 1.054571817*(10**(-34)) #Reduced Planck constant
        self.expansion_length_scale = np.sqrt(self.hbar*self.expansion_time/(self.condensate_num*self.atomic_mass_m))
        self.separation_distance = separation_distance
        self.omega_perp = omega_perp
        self.init_width = np.sqrt(self.hbar/(self.atomic_mass_m*self.omega_perp))
        self.fringe_spacing = (2*np.pi*self.hbar*self.expansion_time)/(self.atomic_mass_m*self.separation_distance)
        
    #%% Methods 
    #Evolution of transverse field - assume free-particle
    def evolve_transverse_field(self, peak_point):
        global_phases = (self.atomic_mass_m/(2*self.hbar*self.expansion_time))*(self.input_x_grid - peak_point)**2
        width_squared = (self.init_width**2)*(1+1j*self.omega_perp*self.expansion_time)
        evolve_transverse_field = np.exp(1j*global_phases)*np.exp(-((self.input_x_grid - peak_point)**2)/(2*width_squared))
        return evolve_transverse_field
    
    #Evolution of longitudinal field - assume free-particle 
    def evolve_longitudinal_field(self, init_field):
        fourier_field = fft(init_field)
        kz = 2 * np.pi * fftfreq(self.dim_z, d=self.dz)
        for i in range(self.dim_z):
            Ek = (self.hbar**2)*(kz[i]**2)/(2*self.atomic_mass_m)
            fourier_field[i] = fourier_field[i]*np.exp(-1j*Ek*self.expansion_time/self.hbar)
        expanded_field = ifft(fourier_field)
        return expanded_field
    
    #Combining evolution of transverse and longitudial fields
    def evolve_field_2d(self, init_field_z, peak_point_x):
        evolved_field_z = self.evolve_longitudinal_field(init_field_z)
        evolved_field_x = self.evolve_transverse_field(peak_point_x)
        evolved_2d_field = evolved_field_x*np.transpose(np.array([evolved_field_z]))
        return evolved_2d_field
        
    #Calculating density
    def calc_evolved_density(self, normalize = True):
        if self.condensate_num == 1:
            field_tot = self.evolve_field_2d(self.init_field_1, 0)
        elif self.condensate_num == 2:
            field_tot = self.evolve_field_2d(self.init_field_1, self.separation_distance/2)\
                +self.evolve_field_2d(self.init_field_2, -self.separation_distance/2)
        
        density = np.abs(field_tot)**2
        total_atom_number = sum(sum(density))*self.dz*self.dx
        if normalize == True:
            density = (self.total_atom_number/total_atom_number)*density
        return density