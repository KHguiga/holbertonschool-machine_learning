#!/usr/bin/env python3

import numpy as np
associated_clustering = __import__('1-clustering').associated_clustering

def new_generation(points,old_generation) :
    K=len(old_generation)
    C=associated_clustering(points,old_generation)
    answer=[]
    for k in range(K) :
        F=np.equal(C,k)
        m=np.sum(F)
        if m!=0 :
            answer.append(np.sum(points*F[:,None],axis=0)/m)
        else :
            answer.append(old_generation[k])
    return np.array(answer)