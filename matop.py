import op
from field import Field
import numpy as np
import scipy.sparse as sp
import time
from utils import ad, pprint

# TAKING ADJOINT??????
class Matrix(object):
    def __init__(self, A, b=None):
        self.A = A
        m, n = A.shape
        assert m == n
        self.b = b
        if b is None:
            self.b = ad.zeros(m)

    def __add__(self, b):
        if isinstance(b, Matrix):
            return self.__class__(self.A + b.A, self.b - b.b)
        elif isinstance(b, Field):
            return self.__class__(self.A, self.b - b.field.reshape(np.prod(b.field.shape)))
        else:
            raise Exception("WTF")

    def __sub__(self, b):
        return self.__add__(-b)

    def __neg__(self):
        return self.__class__(-self.A, -self.b)
    
    def __rsub__(self, b):
        pass

    def __radd__(self, b):
        return self.__add__(self, b)

    def __mul__(self, b):
        if isinstance(b, Field):
            return self.__class__(self.A * b.field, self.b * b.field)
        else:
            raise Exception("WTF")

    def __rmul__(self, b):
        return self.__mul__(self, b)

    def solve(self):
        return sp.linalg.spsolve(self.A, ad.value(self.b))

def laplacian(phi, DT):
    if not hasattr(laplacian, 'A'):
        internalField = phi.getInternalField()
        phi.setInternalField(internalField)
        res = op.laplacian(phi, DT)
        laplacian.A = res.field.diff(internalField)
    return Matrix(laplacian.A)

def ddt(phi, dt):
    shape = phi.getInternalField().shape
    n = np.prod(shape)
    A = sp.eye(n)*(1./dt)
    b = phi.old.getInternalField().reshape(n)/dt
    return Matrix(A, b)

def hybrid(equation, boundary, fields, solver):
    start = time.time()

    names = [phi.name for phi in fields]
    shapes = [phi.getInternalField().shape for phi in fields]
    pprint('Time marching for', ' '.join(names))
    for index in range(0, len(fields)):
        fields[index].old = fields[index]
        fields[index].info()

    LHS = equation(*fields)
    internalFields = [LHS[index].solve().reshape(shapes[index]) for index in range(0, len(fields))]
    newFields = boundary(*internalFields)
    for index in range(0, len(fields)):
        newFields[index].name = fields[index].name

    end = time.time()
    pprint('Time for iteration:', end-start)
    return newFields


