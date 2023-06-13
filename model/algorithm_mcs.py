# -*- coding: utf-8 -*-

from copy import copy
import numpy as np

def __transform_interval__(inter):
    """Transform the intervals so they can be used by mcs_from_intervals.

    Parameters
    ----------
    c : 2-D array
        Array of intervals. Each line contains an interval in the form [a, b].
        
    Returns
    -------
    2-D array
        A sorted array containing all the bounds. If we have n intervals, 2n rows:
            - Col 0: to which interval the bound belongs.
            - Col 1: a bound value.
            - Col 2: 0 if a, 1 if b.
    """
    
    n = inter.shape[0]
    
    a = inter[:,0]
    b = inter[:,1]
    
    a_new = np.c_[a, np.arange(0,n), np.zeros(n)]
    b_new = np.c_[b, np.arange(0,n), np.ones(n)]
    
    inter_new = np.vstack((a_new,b_new))
    inter_new = inter_new[np.argsort(inter_new[:, 0])]
    return inter_new

def mcs_from_intervals(inter):
    """Gets MCS from intervals.
    Algorithm: Dubois, Fragier, Prade, 2000.

    Parameters
    ----------
    c : 2-D array
        Array of intervals. Each line contains an interval in the form [a, b].
        
    Returns
    -------
    list
        A list of lists, each element corresponding to a MCS.
    """

    c = __transform_interval__(inter)
    
    List = []
    K = []
    for i in range(0, c.shape[0]):
        if c[i,2] == 0:
            K.append(int(c[i,1]))
            if c[i+1,2] == 1:
                List.append(copy(K))
        else:
            K.remove(c[i,1])
            
    return List
