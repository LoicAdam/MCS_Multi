# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import linprog

def mabh(A,b):
    """Determine the minimal axis aligned bounding box of a polytope defined by a half-space.

    Parameters
    ----------
    A : 2-D array
        The inequality constraint matrix.
    b : 1-D array
        The inequality constraint vector.
        
    Returns
    -------
    2-D array
        The bounds of the MABH. Each row corresponds to a dimension.
    """
    
    n, p = A.shape
    I = np.eye(p)
    bounds = []
    
    for i in range(0, p):
        
        bmin = linprog(I[i,:], A, b, method = 'revised simplex').x[i]
        bmax = linprog(-I[i,:], A, b, method = 'revised simplex').x[i]
        
        bounds.append([bmin, bmax])
        
    return np.asarray(bounds)