#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 05:55:32 2025

@author: taufiqmurtadho
"""
import scipy.io as sio
import numpy as np

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


mat = loadmat('shaking_data/scan7436/scan7436_exp_data_for_phase_extraction_all.mat')

density_ripple_all = 2*mat['data_structure_basic']['density_profiles_full_stat']
evol_time = mat['data_structure_basic']['time']

data = {'evol_time':evol_time, 'density_ripples':density_ripple_all}

np.save('shaking_data/scan_7436.npy', data)
