# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cp #Very nice library!

def mvair(A,b):
    """Determine the maximal volume axis aligned inscribed box of a polytope defined by a half-space.
    Algorithm: Behroozi, 2020.

    Parameters
    ----------
    A : 2-D array
        The inequality constraint matrix.
    b : 1-D array
        The inequality constraint vector.
        
    Returns
    -------
    2-D array
        The bounds of the MVAIR. Each row corresponds to a dimension.
    """
    
    n, p = A.shape
    Aplus = np.maximum(A,0)
    Amin = np.maximum(-A,0)
    
    xl = cp.Variable(p) #Lower bound
    xu = cp.Variable(p) #Upper bound
    
    objective = cp.Minimize(-cp.sum(cp.log(xu-xl)))
    
    constraints = []
    for i in range(0, n):
        constraints.append( cp.sum(cp.multiply(Aplus[i,:],xu) - cp.multiply(Amin[i,:],xl)) <= b[i])
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    bounds = []
    for i in range(0, p):
        bounds.append( [xl.value[i], xu.value[i]] )
        
    return np.asarray(bounds)