import numpy as np
from numbers import Number

import ops

def _as_tensor_object(obj):
    if isinstance(obj, tensor):
        return obj
    elif isinstance(obj, np.ndarray) or isinstance(obj, Number):
        return ops.ConstantOp(obj).outputs[0]
    else:
        raise Exception('object not recognized', obj)

_tensor_id = 0

class tensor(object):
    def __init__(self, parent=None):
        assert hasattr(self, 'dims')
        if not hasattr(self, 'broadcastable'):
            self.broadcastable = tuple([False]*self.dims)
        global _tensor_id
        self.parent = parent
        self._value = None
        self.id = _tensor_id
        _tensor_id += 1

    @property
    def value(self):
        #print 'retrieving value', self, self._value
        op = self.parent
        if self._value is None:
            if not op:
                raise Exception('input value not provided')
            op.py_compute()
        return self._value

    @value.setter
    def value(self, value):
        #print 'setting value of', self, value
        self._value = value

    def __add__(self, a):
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

    def __getitem__(self, a):
        return ops.SubtensorOp([self, a]).outputs[0]

scalar = type('scalar', (tensor,), {'dims':0})
vector = type('vector', (tensor,), {'dims':1})
matrix = type('matrix', (tensor,), {'dims':2})
col = type('col', (tensor,), {'dims':2, 'broadcastable': (False, True)})

tensor_dims = {0:scalar, 1:vector, 2:matrix}
