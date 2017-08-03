import numpy as np
#class Variable:
#    pass
import operator
import numbers

from . import config, parallel
from .scalar import *
from .variable import *


class Tensor(ArithBase):
    _index = 0
    def __init__(self, shape, scalars=None):
        assert isinstance(shape, tuple)
        index = Tensor._index
        Tensor._index += 1
        self.name = 'Tensor_{}'.format(index)
        self.shape = shape
        self.size = np.prod(shape)
        self.strides = [x/8 for x in np.zeros(shape, np.float64).strides]
        #print('tensor', shape, scalars)
        if scalars is None:
            self.scalars = []
            for i in range(0, self.size):
                self.scalars.append(Scalar())
        else:
            assert isinstance(scalars, list)
            assert len(scalars) == self.size
            if isinstance(scalars[0], Tensor):
                self.scalars = []
                for s in scalars:
                    assert s.size == 1
                    self.scalars.append(s.scalars[0])
            else:
                self.scalars = scalars
        self.dtype = dtype
        self.cellTensor = False
        
        if isinstance(self.scalars[0], IntegerScalar):
            self.dtype = 'integer'

    def _binaryOp(self, b, op, comparison=False):
        if isinstance(b, numbers.Number):
            b = Tensor(self.shape, [b for i in range(0, self.size)])
        if self.shape != b.shape:
            # broadcasting
            if b.shape > self.shape:
                self, b = b, self
                res = [op(b.scalars[0], x) for x in self.scalars]
            else:
                res = [op(x, b.scalars[0]) for x in self.scalars]
            assert b.shape == (1,)
        else:
            res = [op(x, y) for x, y in zip(self.scalars, b.scalars)]
        return Tensor(self.shape, res)

    def _unaryOp(self, op):
        res = [op(x) for x in self.scalars]
        return Tensor(self.shape, res)

    def sqrt(self):
        return Tensor(self.shape, [x.sqrt() for x in self.scalars])

    def stabilise(self, eps):
        res = [ConditionalOp(x < 0, x - eps, x + eps) for x in self.scalars] 
        return Tensor(self.shape, res)

    def __getitem__(self, b):
        if isinstance(b, int):
            b = (b,)
        assert all([isinstance(x, int) for x in b])
        size = self.strides[len(b)-1]
        start = sum([self.strides[i]*b[i] for i in range(0, len(b))])
        shape = self.shape[len(b):]
        if len(shape) == 0:
            shape = (1,)
        res = self.scalars[start:start+size]
        return Tensor(shape, res)

    def __setitem__(self, b, c):
        if isinstance(b, int):
            b = (b,)
        assert isinstance(c, Tensor)
        assert len(b) == len(self.shape)
        assert c.shape == (1,)
        loc = sum([self.strides[i]*b[i] for i in range(0, len(b))])
        self.scalars[loc] = c.scalars[0]

    def extract(self, b):
        assert b.shape == (1,)
        self.cellTensor = True
        res = [Extract(x, b.scalars[0]) for x in self.scalars]
        return Tensor(self.shape, res)

    def dot(self, b):
        assert self.shape == b.shape
        res = sum([self.scalars[i]*b.scalars[i] for i in range(0, self.size)])
        return Tensor((1,), [res])

    
    def magSqr(self):
        return self.dot(self)

    def outer(self, b):
        assert self.shape == (3,)
        assert b.shape == (3,)
        shape = (3, 3)
        res = []
        for i in range(0, 3):
            for j in range(0, 3):
                res.append(self.scalars[i]*b.scalars[j])
        return Tensor(shape, res)

    def _checkMatrix(self):
        assert len(self.shape) == 2
        assert self.shape[0] == self.shape[1]
        return self.shape[0]

    def trace(self):
        n = self._checkMatrix()
        res = 0.
        for i in range(0, n):
            res += self.scalars[i*n + i]
        return Tensor((1,), [res])

    def transpose(self):
        n = self._checkMatrix()
        res = []
        for i in range(0, n):
            for j in range(0, n):
                res.append(self.scalars[j*n + i])
        return Tensor(self.shape, res)

    def tensordot(self, b):
        n = self._checkMatrix()
        assert b.shape == (n,)
        res = []
        for i in range(0, n):
            res.append(sum([self.scalars[i*n + j]*b.scalars[j] for j in range(0, n)]))
        return Tensor((n,), res)

    def matmul(self, b):
        n = self._checkMatrix()
        assert self.shape == b.shape
        res = []
        if isinstance(b, Tensor):
            b = b.scalars
        else:
            b = b.flatten()
        for i in range(0, n):
            for j in range(0, n):
                res.append(sum(self.scalars[i*n + k]*b[k*n + j] for k in range(0, n)))
        return Tensor(self.shape, res)

    # reduction operations
    def sum(self):
        assert self.shape == (1,)
        return Tensor.collate(self, Tensor((1,), [ConstantOp(0)]))

    @classmethod
    def collate(cls, *args):
        n = len(args)/2
        m = args[0].size
        shape = args[0].shape
        res = []
        for j in range(0, m):
            res.append([])
        for i in range(0, n):
            a, b = args[2*i], args[2*i+1]
            assert shape == a.shape
            assert b.shape == (1,)
            for j in range(0, m):
                res[j].extend([a.scalars[j], b.scalars[0]])
        for j in range(0, m):
            res[j] = Collate(*res[j])
        #print len(res), len(res[0].args)
        res = Tensor(shape, res)
        res.cellTensor = True
        return res

    @classmethod
    def switch(cls, cond, ret1, ret2):
        assert ret1.shape == (1,)
        assert ret2.shape == (1,)
        res = [ConditionalOp(cond.scalars[0], ret1.scalars[0], ret2.scalars[0])]
        return cls(ret1.shape, res)

    @classmethod
    def max(cls, x1, x2):
        return Tensor.switch(x1 > x2, x1, x2)


class TensorFunction(object):
    _index = 0
    _module = None
    extraCode = ""

    def __init__(self, name, inputs, outputs, grad=True):
        index = TensorFunction._index
        TensorFunction._index += 1
        #self.name = 'Function_{}'.format(index)
        self.name = 'Function_{}'.format(name)
        #print self.name, len(inputs), len(outputs)
        #print(self.name)
        self._inputTensorIndices = {}
        self._inputTensors = inputs
        self._inputs = []
        for inp in inputs:
            self._inputs.extend(inp.scalars)
            for index, i in enumerate(inp.scalars):
                self._inputTensorIndices[i] = (inp.name, len(inp.scalars), index, inp.cellTensor)
        self._outputTensorIndices = {}
        self._outputTensors = outputs
        self._outputs = []
        for out in outputs:
            self._outputs.extend(out.scalars)
            for index, i in enumerate(out.scalars):
                self._outputTensorIndices[i] = (out.name, len(out.scalars), index, out.cellTensor)
        #self.func = lambdify(self._inputs, self._outputs)

        _outputs = [x for x in self._outputs if x is not None]
        self._children = graphGetChildren(_outputs)
        if config.compile:
            self._genCode(self._inputs, _outputs, self._children.copy())
        OpBase.clear_cache()
        if grad:
            self.grad = self._getAdjoint()

    def _getAdjoint(self):
        gradOutputs = []
        for out in self._outputTensors:
            gradOutputs.append(Tensor(out.shape))
            gradOutputs[-1].cellTensor = out.cellTensor

        #scalarOutput = sum([x.dot(y) for x, y in zip(self._outputTensors, gradInputs)])
        gradients = {}
        for out, grad in zip(self._outputTensors, gradOutputs):
            grad.dtype = out.dtype
            for i, j in zip(out.scalars, grad.scalars):
                gradients[i] = j
        outputScalars = self._diff(self._outputs, self._inputs, gradients)
        #print [out == None for out in outputScalars]
        outputs = []
        i = 0
        #print(self.name)
        for inp in self._inputTensors:
            n = inp.size
            #print(inp.__class__, [(x.func, hash(x), len(x.args)) for x in outputScalars[i:i+n] if x is not None])
            outputs.append(Tensor(inp.shape, outputScalars[i:i+n]))
            outputs[-1].cellTensor = inp.cellTensor
            outputs[-1].dtype = inp.dtype
            i += n
        inputs = self._inputTensors + gradOutputs
        loc = self.name.find('_')
        name = self.name[loc+1:] + '_grad'
        return TensorFunction(name, inputs, outputs, grad=False)

    
    def _diff(self, outputs, inputs, gradients=None):
        if gradients is None:
            gradients = {}
            for out in outputs:
                gradients[out] = 1.
        children = self._children.copy()
        def _diffFunc(out):
            assert children[out] == 0
            grads = []
            if gradients[out] == None:
                grads = [None]*len(out.args)
            elif isinstance(out, OpBase):
                grads = out.grad(gradients[out])
            assert len(grads) == len(out.args)
            for grad, inp in zip(grads, out.args):
                if inp not in gradients or gradients[inp] is None:
                    gradients[inp] = grad
                elif grad is not None:
                    # combining collates
                    if isinstance(gradients[inp], Collate):
                        args = gradients[inp].args + grad.args
                        gradients[inp] = Collate(*args)
                    else:
                        gradients[inp] += grad
                children[inp] -= 1
                if children[inp] == 0:
                    _diffFunc(inp)
        for out in outputs:
            if children[out] == 0:
                _diffFunc(out)
        return [gradients.get(inp, None) for inp in inputs]

    def _genCode(self, inputs, outputs, children):
        sortedOps = graphTopologicalSort(outputs, children)
        codeFile = open(Function.codeDir + 'code.cpp', 'a')

        memString = '' 
        for inp in self._inputTensors:
            memString += 'const {}* {}, '.format(inp.dtype, inp.name)
        for out in self._outputTensors:
            memString += '{}* {}, '.format(out.dtype, out.name)
        codeFile.write('\nvoid {}(int n, {}) {}\n'.format(self.name, memString[:-2], '{'))
        #codeFile.write('\tlong long start = current_timestamp();\n')
        codeFile.write('\tfor (integer i = 0; i < n; i++) {\n')
        names = {}
        for index, op in enumerate(sortedOps):
            if op in names:
                continue
            names[op] = 'Intermediate_{}'.format(index)
            code = ''
            #print names[op], index, op, len(op.args)
            if isinstance(op, ConstScalar) and not isinstance(op, OpBase):
                tensorIndex = self._inputTensorIndices[op]
                code = '{} {} = {}[0];'.format(dtype, names[op], tensorIndex[0])
            elif isinstance(op, Scalar) and not isinstance(op, OpBase):
                tensorIndex = self._inputTensorIndices[op]
                if not tensorIndex[3]:
                    code = '{} {} = {}[i*{} + {}];'.format(dtype, names[op], tensorIndex[0], tensorIndex[1], tensorIndex[2])
            elif isinstance(op, IntegerScalar) and not isinstance(op, OpBase):
                tensorIndex = self._inputTensorIndices[op]
                code = '{} {} = {}[i*{} + {}];'.format('integer', names[op], tensorIndex[0], tensorIndex[1], tensorIndex[2])
            elif isinstance(op, Extract):
                a, b = op.args
                tensorIndex = self._inputTensorIndices[a]
                assert tensorIndex[3]
                code = '{} {} = {}[{}*{} + {}];'.format(dtype, names[op], tensorIndex[0], names[b], tensorIndex[1], tensorIndex[2])
            elif isinstance(op, Collate):
                #print len(op.args)
                tensorIndex = self._outputTensorIndices[op]
                assert tensorIndex[3]
                n = len(op.args)/2
                for i in range(0, n):
                    a, b = op.args[2*i], op.args[2*i+1]
                    code += '{}[{}*{} + {}] += {};\n\t\t'.format(tensorIndex[0], names[b], tensorIndex[1], tensorIndex[2], names[a])
                #print code
            else:
                code = op.c_code(names)

            if op in self._outputTensorIndices:
                tensorIndex = self._outputTensorIndices[op]
                if not tensorIndex[3]:
                    code += '\n\t\t{}[i*{} + {}] += {};'.format(tensorIndex[0], tensorIndex[1], tensorIndex[2], names[op])
            codeFile.write('\t\t' + code + '\n')
        codeFile.write('\t}\n')
        #codeFile.write('\tlong long end = current_timestamp(); mil += end-start; printf("c module {}: %lld\\n", mil);\n'.format(self.name))
        codeFile.write('}\n')
        codeFile.close()
        return


import random, string
def randomName(N):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


def Tensorize(func):
    def ParamFunc(indices=None, outputs=None):
        def Func(*args, **kwargs):
            if not hasattr(ParamFunc, 'tensorFunc'):
                name = randomName(12)
                #print 'here', name
                tensorArgs = []
                #print len(args), len(kwargs)
                for x in args:
                    if x.shape[0] == 1:
                        tensorArgs.append(Tensor((1,), scalars=[ConstScalar()]))
                    elif x.dtype == 'scalar':
                        tensorArgs.append(Tensor(x.shape[1:]))
                    elif x.dtype == 'integer':
                        tensorArgs.append(Tensor(x.shape[1:], scalars=[IntegerScalar()]))
                    else:
                        raise Exception(x.dtype)
                tensorOutputs = func(*tensorArgs, **kwargs)
                shape = args[0].shape[0]
                if not isinstance(tensorOutputs, tuple):
                    tensorOutputs = (tensorOutputs,)
                ParamFunc.outputShapes = [(shape,) + x.shape for x in tensorOutputs]
                ParamFunc.tensorFunc = TensorFunction(name, tensorArgs, tensorOutputs)

            _indices = indices
            if _indices == None:
                _indices = ParamFunc.outputShapes[0][0]
            if outputs is None:
                _outputs = tuple([Zeros(x) for x in ParamFunc.outputShapes])
            else:
                assert len(outputs) == len(ParamFunc.outputShapes)
                #args = args + outputs
                _outputs = outputs
            return TensorFunctionOp(ParamFunc.tensorFunc, args, _outputs, _indices).outputs
        return Func
    return ParamFunc
