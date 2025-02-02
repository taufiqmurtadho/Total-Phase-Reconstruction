#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:54:03 2024

@author: taufiqmurtadho
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse

class fdm_poisson_1d_solver:
    def __init__(self, source, input_z_grid, extrapolation = True, pad_factor = 0.1):
        self.source = source
        self.input_z_grid = input_z_grid
        self.dim_z =  len(self.input_z_grid) 
        self.dz = abs(self.input_z_grid[1] - self.input_z_grid[0])
        
        if extrapolation == False:
            self.pad_factor = 0
        
        self.pad_factor = pad_factor
        self.padnum = int(self.pad_factor*self.dim_z)
        self.extended_z_grid = np.arange(self.input_z_grid[0]-self.padnum*self.dz,
                                            self.input_z_grid[-1]+(self.padnum+1)*self.dz, self.dz)
             
        # Construct the Laplacian matrix using finite differences
        n = len(self.extended_z_grid)  # Length of extended z_grid
        main_diag = -2 * np.ones(n)         # Main diagonal
        sub_diag = np.ones(n - 1)           # Subdiagonal
        self.Laplacian_mat = sparse.csr_matrix(np.diag(main_diag) + np.diag(sub_diag, -1) 
                              + np.diag(sub_diag, 1))/(self.dz**2)
       
    def decaying_extrapolation(self, rate = 1):
        extended_source = np.zeros(len(self.extended_z_grid))
        for i in range(len(self.extended_z_grid)):
            if i<self.padnum:
                extended_source[i] = self.source[0]*np.exp(-rate*(self.extended_z_grid[i]-self.input_z_grid[0])**2)
            elif i>self.dim_z+self.padnum:
                extended_source[i] = self.source[-1]*np.exp(-rate*(self.extended_z_grid[i] - self.input_z_grid[-1])**2)
            else:
                extended_source[i] = self.source[i-self.padnum-1]
        self.extended_source = extended_source
        return extended_source
    
    def solve_poisson(self):
        extended_source = self.decaying_extrapolation()
        sol = spsolve(self.Laplacian_mat, extended_source)
        sol = sol[self.padnum:self.dim_z+self.padnum]
        sol = sol - sum(sol)/self.dim_z
        self.solution = sol
        return sol