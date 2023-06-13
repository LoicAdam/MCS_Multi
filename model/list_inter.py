# -*- coding: utf-8 -*-

def __elementwise_intersections__(set1, set2):
    """From two sets of sets, intersect each element of the first set with each element of the second set.

    Parameters
    ----------
    set1 : set
        Set of set.
    set2 : set
        Set of set
        
    Returns
    -------
    set
        All sets obtained from the intersections, excluding empty sets.
    """
    
    set_inter = set()

    for s1 in set1:
        for s2 in set2:
            s_inter = s1 & s2
            if (s_inter != set()):
                set_inter.add(s_inter)
                
    return set_inter

def intersection_sources(sources):
    """From different sources of MCS, obtain the common MCS.

    Parameters
    ----------
    sources : list
        A list of lists. Each list contains the MCS for a source.
        
    Returns
    -------
    set
        The MCS.
    """
    
    #Contain some non Masximal Coherent Subsets.
    set1 = set(frozenset(x) for x in sources[0])
    for i in range(1, len(sources)):
        
        set2 = set(frozenset(x) for x in sources[i])
        setnew = __elementwise_intersections__(set1, set2)
        set1 = setnew
        
    #Remove non Maximaml Coherent Subsets.
    setinter = set()
    for s in set1:
        if any(s < e for e in set1):
            continue
        else:
            setinter.add(s)

    return setinter
   