# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:37:26 2022

@author: adamloic
"""

import numpy as np

def points_to_generate(nb, dim, range_min = 0, range_max = 2):

    x = np.random.uniform(range_min,range_max, size=(nb,dim))
    return x

def generate_points_linear(a, b, x, error):
      
    b_new = np.repeat(b, x.shape[0])    
    
    if x.ndim == 1:
        x = x[:,np.newaxis]
        
    y = np.sum(np.multiply(a, x), axis = 1) + b_new + error
    
    return y

def generate_hyperrectangles(x, y, x_err_bounds = (0,0.02), y_err_bounds = (0,0.04)):
        
    l_x = np.random.uniform(x_err_bounds[0], x_err_bounds[1], size = x.shape + (2,))
    l_y = np.random.uniform(y_err_bounds[0], y_err_bounds[1], size = (y.size,2))
    
    if x.ndim == 1:
        l_x[:,0] = -l_x[:,0]
        l_x = l_x + np.repeat(x[:,np.newaxis], 2, axis = 1)
    else:
        l_x[:,0] = -l_x[:,0]
        l_x = l_x + np.repeat(x[:,:,np.newaxis], 2, axis = 2)
        
    l_y[:,0] = -l_y[:,0]
    l_y = l_y + np.repeat(y[:,np.newaxis], 2, axis = 1)
    
    return np.concatenate((l_x[np.newaxis, ...], l_y[np.newaxis, ...]), axis = 0)
