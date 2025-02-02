#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:24:34 2024

@author: taufiqmurtadho
"""
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Load the CSV file
# Assuming columns are named 'R', 'G', 'B'
data = pd.read_csv('../plotting_funcs/fast-table-byte-0032.csv')
rgb_vals = data[['RGB_r', 'RGB_g', 'RGB_b']].values.tolist()
normalized_rgb_vals = [[r/255, g/255, b/255] for r, g, b in rgb_vals]

fast_cmap = LinearSegmentedColormap.from_list('fast', normalized_rgb_vals)
