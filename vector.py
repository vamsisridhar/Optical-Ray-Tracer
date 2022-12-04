# -*- coding: utf-8 -*-

import numpy as np

def magnitude(p):
    
    return np.sqrt(np.dot(np.array(p), np.array(p)))


def normalise(p):
    return p/magnitude(p)


    
__all__ = ['magnitude', 'normalise']