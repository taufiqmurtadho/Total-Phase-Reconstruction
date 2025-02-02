#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:57:21 2025

@author: taufiqmurtadho
"""

import numpy as np
import scipy.io as sio

def _check_keys( dict):
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
    """
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)
"""
mat= loadmat('recurrence_data/density_ripple_data_scan5100.mat')

density_ripples_all = mat['density_ripple_all']
density_ripples_all = np.array([dr[0:500,:] for dr in density_ripples_all])
z_axis = mat['z_axis']
evol_time = mat['evol_time']

data = {'density_ripples_all': density_ripples_all, 'z_grid':z_axis, 'evol_time':evol_time}

np.save('recurrence_data/scan5100_recurrence.npy', data)
"""

mat = loadmat('recurrence_data/interference_picture_example.mat')
rho_tof = mat['rho_tof']
x_grid = mat['x_axis']
z_grid = mat['z_axis']
density_ripple = mat['n_tof']

data_2 = {'density_ripple':density_ripple, 'rho_tof':rho_tof, 'x_grid': x_grid, 'z_grid':z_grid}

np.save('recurrence_data/interference_example.npy', data_2)
