# custom norm for numpy 1.7
import numpy as np
import cfuncs

def norm(a, axis, **kwuser):
    try:
        #return np.linalg.norm(a, axis=axis, keepdims=True)
        return np.linalg.norm(a, axis=axis).reshape((-1,1))
    except:
        #axis=1
        return np.sqrt(np.einsum('ij,ij->i', a, a)).reshape((-1,1))


def add_at(a, indices, b):
    try:
        1/0
        np.add.at(a, indices, b)
    except:
        print('sdadsdsad')
        assert indices.dtype == np.int32
        assert a.dtype == np.float64
        assert b.dtype == np.float64
        cfuncs(a, indices, b)
