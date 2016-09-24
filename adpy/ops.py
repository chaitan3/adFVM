import numpy as np
import operator

import tensor
from opbase import Op

class ConstantOp(Op):
    def make_node(self, inputs):
        x, = inputs
        self.constant = x
        dims = 0
        if isintance(x, np.ndarray):
            dims = len(x.shape) 
        return [tensor.tensor_dims[dims]()]

    def perform(self, inputs):
        return [self.constant]

class BinaryOp(Op):
    def make_node(self, inputs):
        x, y = inputs
        assert isinstance(x, tensor.tensor)
        y = tensor._as_tensor_object(y)
        obj = x.__class__()
        return [obj]

    def perform(self, inputs):
        op = getattr(operator, self.bin_op)
        return [op(inputs[0], inputs[1])]

class NegOp(Op):
    def make_node(self, inputs):
        x, = inputs
        assert isinstance(x, tensor.tensor)
        obj = x.__class__()
        return [obj]

    def perform(self, inputs):
        return [-inputs[0]]

class ReduceOp(Op):
    def make_node(self, inputs):
        x, = inputs
        assert isinstance(x, tensor.tensor)
        if self.axis is None:
            self.axis = tuple(range(x.dims))
        tensor_dims = len(x.dims)-len(self.axis)
        y = tensor.tensor_dims[tensor_dims]()
        return [y]

    def perform(self, inputs):
        return [inputs[0].sum(axis=self.axis)]

AddOp = type('AddOp', (BinaryOp,), {'bin_op':'add'})
SubOp = type('SubOp', (BinaryOp,), {'bin_op':'sub'})
MulOp = type('MulOp', (BinaryOp,), {'bin_op':'mul'})
DivOp = type('DivOp', (BinaryOp,), {'bin_op':'div'})
PowOp = type('PowOp', (BinaryOp,), {'bin_op':'pow'})
