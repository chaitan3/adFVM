# custom norm for numpy 1.7
import numpy as np

from . import cfuncs
from .cfuncs import intersectPlane, getCells, decompose, reduceAbsMin, selectMultipleRange, reduceSum

def norm(a, axis, **kwuser):
    try:
        #return np.linalg.norm(a, axis=axis, keepdims=True)
        return np.linalg.norm(a, axis=axis).reshape((-1,1))
    except:
        #axis=1
        return np.sqrt(np.einsum('ij,ij->i', a, a)).reshape((-1,1))


def add_at(a, indices, b):
    np.add.at(a, indices, b)
    
