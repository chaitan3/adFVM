import numpy as np
from numbers import Number

import ops

def _as_tensor_object(a):
    if isinstance(a, tensor):
        return a
    elif isinstance(a, np.ndarray) or isinstance(a, Number):
        ConstantOp = type('ConstantOp', (ops.ConstantOp,), {'constant':a})
        return ConstantOp().outputs[0]
    else:
        raise Exception('object not recognized', a)

class tensor(object):
    def __init__(self, parent=None):
        self.parent = parent
        self._value = None

    @property
    def value(self):
        #print 'retrieving value', self, self._value
        if self._value is None:
            if self.parent is None:
                raise Exception('input value not provided')
            self.parent.py_compute()
        return self._value

    @value.setter
    def value(self, value):
        #print 'setting value of', self, value
        self._value = value

    def __add__(self, a):
        print ops.AddOp([self, a]).c_code()
        return ops.AddOp([self, a]).outputs[0]

    def __radd__(self, a):
        return self.__add__(a)

    def __sub__(self, a):
        return ops.SubOp([self, a]).outputs[0]

    def __mul__(self, a):
        return ops.MulOp([self, a]).outputs[0]

    def __rmul__(self, a):
        return self.__mul__(a)

    def __div__(self, a):
        return ops.DivOp([self, a]).outputs[0]

    def __pow__(self, a):
        return ops.PowOp([self, a]).outputs[0]

    def __neg__(self, a):
        return ops.NegOp([self]).outputs[0]

scalar = type('vector', (tensor,), {'dims':0})
vector = type('vector', (tensor,), {'dims':1})
matrix = type('vector', (tensor,), {'dims':2})

tensor_dims = {0:scalar, 1:vector, 2:matrix}
