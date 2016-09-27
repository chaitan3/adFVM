import numpy as np
from numbers import Number
import operator

import tensor
from opbase import Op

class ConstantOp(Op):
    def make_node(self, inputs):
        assert len(inputs) == 0
        x = self.constant
        dims = 0
        assert isinstance(x, Number)
        return [tensor.scalar()]
        #if isinstance(x, np.ndarray):
        #    dims = len(x.shape) 
        #return [tensor.tensor_dims[dims]()]

    def perform(self, inputs):
        return [self.constant]

    def c_code(self):
        return ''
#        constant = np.array(self.constant).astype(np.float64)
#        shape = constant.shape
#        if len(shape) == 0:
#            op_cose +int64_t shape[] = {{%(shape)s}};
#        op_code = '''
#    scalar *data = {{%(constant)s}};
#    int dims = %(dims)d;
#    ndarray* {output_0} = ndarray_build(data, shape, dims);
#        ''' % {'constant': constant, 'shape': constant.shape, 'dims': len(constant.shape)}
#        return op_code

class BinaryOp(Op):
    def make_node(self, inputs):
        x, y = inputs
        assert isinstance(x, tensor.tensor)
        y = tensor._as_tensor_object(y)
        inputs[1] = y
        return [x.__class__()]

    def perform(self, inputs):
        op = getattr(operator, self.bin_op)
        return [op(inputs[0], inputs[1])]

    def c_code(self):
        # no broadcasting support yet
        bin_op = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
        bin_op = bin_op[self.bin_op]
        y = self.inputs[1]
        if isinstance(y.parent, ConstantOp):
            return '''
    ndarray* {output_0} = ndarray_alloc({input_0});
    {{
        int size = {input_0}->size;
        scalar *data0 = {input_0}->data;
        scalar *data2 = {output_0}->data;
        for (i=0; i < size; i++) {{
            data2[i] = data0[i] %(bin_op)s %(constant)f;
        }}
    }}
            ''' % {'bin_op': bin_op, 'constant':y.parent.constant}
        else:
            return '''
    ndarray* {output_0} = ndarray_alloc({input_0});
    {{
        int size = {input_0}->size;
        scalar *data0 = {input_0}->data;
        scalar *data1 = {input_1}->data;
        scalar *data2 = {output_0}->data;
        for (i=0; i < size; i++) {{
            data2[i] = data0[i] %(bin_op)s data1[i];
        }}
    }}
            ''' % {'bin_op': bin_op}

class NegOp(Op):
    def make_node(self, inputs):
        x, = inputs
        assert isinstance(x, tensor.tensor)
        return [x.__class__()]

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
