#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:20:28 2024

@author: taufiqmurtadho
"""

import numpy as np
from numpy.random import normal

class thermal_fluctuations_sampling:
    def __init__(self, mean_density_1d, temperature, scaling_factor = 1,omega_perp = 2*np.pi*1400):
        #initialize class properties
        self.mean_density = mean_density_1d #assumed uniform over space
        self.temperature = temperature
        self.omega_perp = omega_perp
        self.scaling_factor = scaling_factor
        #physical constant in SI unit
        self.atomic_mass_m = 86.9091835*1.66054*(10**(-27)) #mass of Rb-87 (kg)
        self.hbar = 1.054571817*(10**(-34))     #Reduced Planck constant (SI)
        self.kb = 1.380649*(10**(-23))          #Boltzmann constant
        self.scattering_length = 5.2e-9         #5.2 nm scattering length        
        self.interaction_strength_g = 2*self.hbar*self.omega_perp*self.scattering_length
        
    def generate_fluct_samples(self, fourier_cutoff, z_grid, num_samples = 1):
        
        Lz = z_grid[-1] - z_grid[0]
        k1 = 2*np.pi/Lz
        density_fluct_all_samples = []
        phase_fluct_all_samples = []
        for i in range(num_samples):
            density_fluct_profile = np.zeros(len(z_grid))
            phase_fluct_profile = np.zeros(len(z_grid))
            for n in range(-fourier_cutoff,fourier_cutoff+1):
                if n!=0:
                    kn = k1*n
                    Ekn = ((self.hbar*kn)**2)/(2*self.atomic_mass_m)
                    epsilon_kn = np.sqrt(Ekn*(Ekn+2*self.interaction_strength_g*self.mean_density))
                    
                    mean_occupation_kn = (np.exp(epsilon_kn/(self.kb*self.temperature))-1)**(-1)
                    thermal_sampling = np.sqrt(mean_occupation_kn/2)*(normal()+1j*normal())
                
                    coeffs_density = self.scaling_factor*1j*np.sqrt(self.mean_density/Lz)*np.sqrt(Ekn/epsilon_kn)*thermal_sampling
                    coeffs_phase = self.scaling_factor*np.sqrt(1/(4*self.mean_density*Lz))*np.sqrt(epsilon_kn/Ekn)*thermal_sampling
                
                    density_fluct_cont = np.array([coeffs_density*np.exp(1j*kn*z)+np.conjugate(coeffs_density)*np.exp(-1j*kn*z)for z in z_grid])
                    density_fluct_cont = np.real(density_fluct_cont)
                
                    phase_fluct_cont = np.array([coeffs_phase*np.exp(1j*kn*z)+np.conjugate(coeffs_phase)*np.exp(-1j*kn*z) for z in z_grid])
                    phase_fluct_cont = np.real(phase_fluct_cont)
                    
                    density_fluct_profile += density_fluct_cont
                    phase_fluct_profile += phase_fluct_cont
            
            density_fluct_all_samples.append(density_fluct_profile)
            phase_fluct_all_samples.append(phase_fluct_profile)
        
        phase_fluct_all_samples = np.array(phase_fluct_all_samples)
        density_fluct_all_samples = np.array(density_fluct_all_samples)
        
        return [density_fluct_all_samples, phase_fluct_all_samples]