#from numba import jit,double
import numpy as np
import theano.tensor as T
import theano
import numexpr as ne
import math

#@jit('f8[:](f8[:])')
def test_numba(x):
    res = np.zeros_like(x)
    n = x.shape[0]
    for i in range(n):
        res[i] = math.exp(math.sqrt(math.exp(x[i]**2))**4)
    return res

def test_numpy(x):
    return np.exp(np.sqrt(np.exp(x**2))**4)

x = T.dvector('x')
z = T.exp(T.sqrt(T.exp(x**2))**4)
test_theano = theano.function([x], z)

def test_numexpr(x):
    return ne.evaluate("exp(sqrt(exp(x**2))**4)")


x = np.random.rand(1000)
