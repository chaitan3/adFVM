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
        x, y = self.inputs
        if isinstance(y.parent, ConstantOp):
            constant = y.parent.constant
            return '''
                ndarray* {output_0} = ndarray_alloc_from({input_0});
                {{
                    index_t size = {input_0}->size;
                    char *data0 = {input_0}->data;
                    char *data2 = {output_0}->data;
                    for (i0 = 0; i0 < size; i0++) {{
                        *((scalar_t *)data2 + i0) = *((scalar_t *)data0 + i0) %(bin_op)s %(constant)f;
                    }}
                }}
            ''' % locals()
        elif (x.broadcastable == y.broadcastable):
            return '''
                ndarray* {output_0} = ndarray_alloc_from({input_0});
                {{
                    int64_t size = {input_0}->size;
                    char *data0 = {input_0}->data;
                    char *data1 = {input_1}->data;
                    char *data2 = {output_0}->data;
                    for (i0 = 0; i0 < size; i0++) {{
                        *((scalar_t *)data2 + i0) = *((scalar_t *)data0 + i0) %(bin_op)s *((scalar_t *)data1 + i0);
                    }}
                }}
            ''' % locals()
        else:
            index = 0
            code = '''
                index_t *shape0 = {input_0}->shape;
                index_t *shape1 = {input_1}->shape;
                index_t *shape2 = malloc(sizeof(int64_t)*{input_0}->dims);
            '''
            assert len(x.broadcastable) == len(y.broadcastable)
            loop_code = ''
            x_stride = ''
            y_stride = ''
            z_stride = ''
            for xb, yb in zip(x.broadcastable, y.broadcastable):
                loop_index = 'i{}'.format(index)
                stride = '%(strider)s->strides[%(index)s]*%(loop_index)s +'
                if not xb:
                    shape = 'shape0'
                    strider = '{input_0}'
                    x_stride += stride % locals()
                    strider = '{output_0}'
                    z_stride += stride % locals()
                else:
                    shape = 'shape1'
                if not yb:
                    strider = '{input_1}'
                    y_stride += stride % locals()
                    if xb:
                        strider = '{output_0}'
                        z_stride += stride % locals()
                code += '\tshape2[%(index)s] = %(shape)s[%(index)s];\n' % locals()
                #loop_code += '\tFOR_LOOP (%(loop_index)s, %(shape)s) {{\n'
                loop_code += '\tfor (index_t %(loop_index)s = 0; %(loop_index)s < %(shape)s[%(index)s]; %(loop_index)s++) {{\n' % locals()
                index += 1
            x_stride = x_stride[:-1]
            y_stride = y_stride[:-1]
            z_stride = z_stride[:-1]

            code += '\tndarray* {output_0} = ndarray_alloc_new(shape2, {input_0}->dims, {input_0}->type);\n'
            code += '''
                char *data0 = {input_0}->data;
                char *data1 = {input_1}->data;
                char *data2 = {output_0}->data;
                '''
            code += loop_code
            code += '\t*(scalar_t*)(data2 + %(z_stride)s) = *(scalar_t*)(data0 + %(x_stride)s) %(bin_op)s *(scalar_t*)(data1 + %(y_stride)s);\n' % locals()
            code += '}'*len(2*x.broadcastable)
            print code 
            return code

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
