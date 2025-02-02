#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 21:08:23 2025

@author: taufiqmurtadho
"""

import numpy as np
from numpy.fft import fft

class phase_correlation_functions:
    def __init__(self, phases_data):
        self.phases_data = phases_data
        self.num_shots = np.shape(phases_data)[0]
        self.dim_z = np.shape(phases_data)[1]
        self.vertex_corr_2d = None
        
    def calculate_vertex_corr_func_2d(self):
        vertex_corr_2d = np.zeros((self.dim_z, self.dim_z))
        for i in range(self.dim_z):
            for j in range(self.dim_z):
                elem = 0
                for k in range(self.num_shots):
                    elem = elem + np.exp(1j*(self.phases_data[k,i] - self.phases_data[k,j]))
                elem = elem/self.num_shots
                vertex_corr_2d[i,j] = np.real(elem)
        self.vertex_corr_2d = vertex_corr_2d
        return vertex_corr_2d
    
    
    def calculate_vertex_corr_func_1d(self):
        if self.vertex_corr_2d==None:
            vertex_corr_2d = self.calculate_vertex_corr_func_2d()
        else:
            vertex_corr_2d = self.vertex_corr_2d

        vertex_corr_1d = np.zeros(self.dim_z)
        # Create index arrays
        rows, cols = np.indices((self.dim_z, self.dim_z))

        # Loop over each distance from the diagonal
        dim_z_distance = round(self.dim_z/2)
        for d in range(dim_z_distance):
            # Mask for elements with distance d from the diagonal
            mask = np.abs(rows - cols) == d
            # Compute the average of the elements for the current distance
            vertex_corr_1d[d] = vertex_corr_2d[mask].mean()
        
        self.vertex_corr_1d = vertex_corr_1d
        return vertex_corr_1d
    
    def calculate_vertex_corr_func_slice_1d(self, fixed_index):
        vertex_corr_1d = np.zeros(self.dim_z)
        for i in range(self.dim_z):
            elem = 0
            for k in range(self.num_shots):
                elem = elem + np.exp(1j*(self.phases_data[k,i] - self.phases_data[k,fixed_index]))
            elem  = elem/self.num_shots
            vertex_corr_1d[i] = np.real(elem)
        return vertex_corr_1d
    
    
    def calculate_two_point_function(self, reference = True):
        if reference == True:
            midpoint = round(self.dim_z/2)
            phases_data = np.array([self.phases_data[i,:] - self.phases_data[i,midpoint] 
                                    for i in range(self.num_shots)])
        else:
            phases_data = self.phases_data
        
        avg_phases = np.mean(phases_data, 0)
        two_point_corr = np.zeros((self.dim_z, self.dim_z))
        for i in range(self.dim_z):
            for j in range(self.dim_z):
                elem = 0
                for k in range(self.num_shots):
                    elem = elem + (phases_data[k,i]*phases_data[k,j] - avg_phases[i]*avg_phases[j])
                elem  = elem/self.num_shots
                two_point_corr[i,j] = np.real(elem)
        
        return two_point_corr
    
    def analyze_mean_fourier_coeffs(self, z_grid):
        pixel_size = abs(z_grid[1] - z_grid[0])
        condensate_length = abs(z_grid[-1] - z_grid[0])
        fourier_coeffs = [np.abs(fft(self.phases_data[i]))**2 for i in range(self.num_shots)]
        mean_fourier_coeffs = np.mean(fourier_coeffs, 0)*pixel_size/condensate_length
        return mean_fourier_coeffs
    
    def calculate_full_distribution_function(self, z_grid, integration_length):
        pixel_size = abs(z_grid[1] - z_grid[0])
        start = -integration_length/2
        end = integration_length/2
        
        idx_start = np.argmin(abs(z_grid-start))
        idx_end = np.argmin(abs(z_grid -end))
        
        fdf = np.zeros(self.num_shots)
        for i in range(self.num_shots):
            integ = sum(np.exp(1j*self.phases_data[i,idx_start:idx_end]))*pixel_size
            fdf[i] = np.abs(integ)**2
        
        fdf= fdf/np.mean(fdf)
        return fdf
    
    def calculate_four_point_function(self, z_grid, z3, z4, reference = True):
        if reference == True:
            midpoint = round(self.dim_z/2)
        
            phases_data = np.array([self.phases_data[i,:] - self.phases_data[i,midpoint] 
                                    for i in range(self.num_shots)])
        else:
            phases_data = self.phases_data
        
        z3_idx = np.argmin(abs(z_grid-z3))
        z4_idx = np.argmin(abs(z_grid-z4))
        
        four_point_func = np.zeros((self.dim_z, self.dim_z))
        for n in range(self.num_shots):
            for i in range(self.dim_z):
                for j in range(self.dim_z):
                    four_point_func[i,j] = four_point_func[i,j]+phases_data[n,i]*phases_data[n,j]*phases_data[n, z3_idx]*phases_data[n,z4_idx]
        
        four_point_func = four_point_func/self.num_shots
        return four_point_func
    
    def calculate_disconnected_four_point_function(self, z_grid, z3, z4):
        
        two_point_func = self.calculate_two_point_function()
        
        z3_idx = np.argmin(abs(z_grid-z3))
        z4_idx = np.argmin(abs(z_grid-z4))
        
        wick_four_point = np.zeros((self.dim_z, self.dim_z))
        for i in range(self.dim_z):
            for j in range(self.dim_z):
                wick_four_point[i,j] = two_point_func[i,j]*two_point_func[z3_idx, z4_idx]+two_point_func[i,z3_idx]*two_point_func[j,z4_idx]+two_point_func[i,z4_idx]*two_point_func[j,z3_idx]
                
        return wick_four_point
                