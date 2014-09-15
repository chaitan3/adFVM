import op
import numpy as np
import scipy.sparse as sp

class Matrix(object):
    def __init__(A, b=None):
        self.A = A
        m, n = A.shape
        assert m == n
        if b is None:
            self.b = np.zeros(m)

    def __add__(b):
        if isinstance(b, Matrix):
            return self(self.A + b.A, self.b - b)
        else:
            return self(self.A, self.b - b)

    def __sub__(b):
        return self.__add__(-b)

    def __neg__(self):
        return self(-A, -b)
    
    def __rsub__(b):
        pass

    def __radd__(b):
        return self.__add__(b)

    def __mul__(b):
        return self(self.A * b, self.b * b)

    def __rmul__(b):
        return self.__mul__(b)

    def solve(self):
        return sp.linalg.solve(self.A, self.b)  

def laplacian(phi):
    return op.laplacian(phi)

def ddt(phi, phi0, dt):
    shape = phi.getInternalField().shape
    n = np.prod(shape)
    A = sp.eye(n)*(1./dt)
    b = phi0.getInternalField()/dt
    return Matrix(A, b)

