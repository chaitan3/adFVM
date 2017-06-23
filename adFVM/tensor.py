import numpy as np
#class Variable:
#    pass
from sympy import Symbol, sqrt, Piecewise, lambdify, Function
from sympy.utilities.autowrap import ufuncify
from sympy.core import add, mul, power, numbers, relational
from sympy.functions.elementary import piecewise
from sympy.logic import boolalg
import operator

dtype = 'scalar'
import adFVM
import os, sys, subprocess
import ctypes
codeDir = os.path.dirname(adFVM.__file__) + '/gencode/'

class Container(object):
    pass

def prod(factors):
    return reduce(operator.mul, factors, 1)

Extract = Function('extract')
Collate = Function('collate')

class Scalar(Symbol):
    _index = 0
    def __new__(cls):
        index = Scalar._index
        Scalar._index += 1
        return Symbol.__new__(cls, 'Scalar_{}'.format(index))

class IntegerScalar(Symbol):
    _index = 0
    def __new__(cls):
        index = IntegerScalar._index
        IntegerScalar._index += 1
        return Symbol.__new__(cls, 'Integer_{}'.format(index))

#class Constant(Symbol):
#    _index = 0
#    def __new__(cls):
#        index = Constant._index
#        Constant._index += 1
#        return Symbol.__new__(cls, 'Constant_{}'.format(index))

class Tensor(object):
    _index = 0
    def __init__(self, shape, scalars=None):
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
            self.scalars = scalars

    def tolist():
        return self.scalars

    def _binaryOp(self, b, op):
        if isinstance(b, float) or isinstance(b, int):
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

    def __add__(self, b):
        return self._binaryOp(b, operator.add)

    def __radd__(self, b):
        return self.__add__(b)

    def __sub__(self, b):
        return self._binaryOp(b, operator.sub)

    def __mul__(self, b):
        return self._binaryOp(b, operator.mul)

    def __rmul__(self, b):
        return self.__add__(b)

    def __div__(self, b):
        return self._binaryOp(b, operator.div)

    def __neg__(self):
        return Tensor(self.shape, [-x for x in self.scalars])


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
        res = [Extract(x, b.scalars[0]) for x in self.scalars]
        return Tensor(self.shape, res)

    def dot(self, b):
        assert self.shape == (3,)
        assert b.shape == (3,)
        res = sum([self.scalars[i]*b.scalars[i] for i in range(0, 3)])
        return Tensor((1,), [res])

    def abs(self):
        #return Tensor(self.shape, [abs(x) for x in self.scalars])
        return Tensor(self.shape, [Piecewise((-x, x<0), (x, True)) for x in self.scalars])

    def sqrt(self):
        return Tensor(self.shape, [sqrt(x) for x in self.scalars])

    def magSqr(self):
        return self.dot(self)

    def stabilise(self, eps):
        res = [Piecewise((x - eps, x < 0), (x + eps, True)) for x in self.scalars]
        return Tensor(self.shape, res)

    def outer(self, b):
        assert self.shape == (3,)
        assert b.shape == (3,)
        shape = (3, 3)
        res = []
        for i in range(0, 3):
            for j in range(0, 3):
                res.append(self.scalars[i]*b.scalars[j])
        return Tensor(shape, res)

    @classmethod
    def collate(cls, *args):
        n = len(args)/2
        m = args[0].size
        shape = args[0].shape
        res = [[]]*m
        for i in range(0, n):
            a, b = args[2*i], args[2*i+1]
            assert shape == a.shape
            assert b.shape == (1,)
            for j in range(0, m):
                res[j].extend([a.scalars[j], b.scalars[0]])
        for j in range(0, m):
            res[j] = Collate(*res[j])
        return CellTensor(shape, res)

    @classmethod
    def switch(cls, cond, ret1, ret2):
        assert ret1.shape == (1,)
        assert ret2.shape == (1,)
        res = [Piecewise((ret1.scalars[0], cond), (ret2.scalars[0], True))]
        return cls(ret1.shape, res)

class CellTensor(Tensor):
    pass

def ZeroTensor(shape):
    return Tensor(shape, [numbers.Zero() for i in range(0, np.prod(shape))])

class Function(object):
    _index = 0
    _module = None
    def __init__(self, name, inputs, outputs):
        index = Function._index
        Function._index += 1
        #self.name = 'Function_{}'.format(index)
        self.name = 'Function_{}'.format(name)
        self._inputTensorIndices = {}
        self._inputTensors = inputs
        self._inputs = []
        for inp in inputs:
            self._inputs.extend(inp.scalars)
            for index, i in enumerate(inp.scalars):
                self._inputTensorIndices[i] = (inp.name, len(inp.scalars), index, isinstance(inp, CellTensor))
        self._outputTensorIndices = {}
        self._outputTensors = outputs
        self._outputs = []
        for out in outputs:
            self._outputs.extend(out.scalars)
            for index, i in enumerate(out.scalars):
                self._outputTensorIndices[i] = (out.name, len(out.scalars), index, isinstance(out, CellTensor))
        #self.func = lambdify(self._inputs, self._outputs)

        self._children = self._getChildren(self._outputs)
        self._genCode(self._inputs, self._outputs, self._children.copy())

    def _getAdjoint(self):
        gradInputs = [Scalar() for x in self._outputs]
        scalarOutput = sum([x * y for x, y in zip(self._outputs, gradInputs)])
        gradient = self._diff(scalarOutput, self._inputs)

    def _getChildren(self, outputs):
        children = {}
        funcs = set()
        def _childrenFunc(out):
            if out.func not in funcs:
                funcs.add(out.func)
            for inp in out.args:
                if inp in children:
                    children[inp] += 1
                else:
                    children[inp] = 1
                    _childrenFunc(inp)

        for out in outputs:
            children[out] = 0
        for out in outputs:
            _childrenFunc(out)
        #print funcs
        return children

    def _genCode(self, inputs, outputs, children):
        sortedOps = self._topologicalSort(outputs, children)
        codeFile = open(codeDir + 'code.c', 'a')

        memString = '' 
        for inp in self._inputTensors:
            memString += '{}* {}, '.format(dtype, inp.name)
        for out in self._outputTensors:
            memString += '{}* {}, '.format(dtype, out.name)
        codeFile.write('\nvoid {}(int n, {}) {}\n'.format(self.name, memString[:-2], '{'))
        codeFile.write('\tlong long start = current_timestamp();\n')
        codeFile.write('\tfor (int i = 0; i < n; i++) {\n')
        names = {}
        for index, op in enumerate(sortedOps):
            names[op] = 'Intermediate_{}'.format(index)
            argNames = [names[inp] for inp in op.args]
            code = ''
            if op.func == Scalar:
                tensorIndex = self._inputTensorIndices[op]
                if not tensorIndex[3]:
                    code = '{} {} = *({} + i*{} + {});'.format(dtype, names[op], tensorIndex[0], tensorIndex[1], tensorIndex[2])
            elif op.func == IntegerScalar:
                tensorIndex = self._inputTensorIndices[op]
                code = '{} {} = *({} + i*{} + {});'.format('integer', names[op], tensorIndex[0], tensorIndex[1], tensorIndex[2])
            elif op.func == mul.Mul:
                code = '{} {} = {};'.format(dtype, names[op], '*'.join(argNames))
            elif op.func == add.Add:
                code = '{} {} = {};'.format(dtype, names[op], '+'.join(argNames))
            elif op.func == power.Pow:
                code = '{} {} = pow({},{});'.format(dtype, names[op], argNames[0], argNames[1])
            elif op.func == numbers.NegativeOne:
                code = 'const {} {} = {};'.format(dtype, names[op], -1)
            elif op.func == numbers.Float:
                code = 'const {} {} = {};'.format(dtype, names[op], op.num)
            elif op.func == numbers.Integer:
                code = 'const {} {} = {};'.format(dtype, names[op], op.p)
            elif op.func == numbers.Half:
                code = 'const {} {} = {};'.format(dtype, names[op], 0.5)
            elif op.func == numbers.Zero:
                code = 'const {} {} = {};'.format(dtype, names[op], 0)
            elif op.func == numbers.One:
                code = 'const {} {} = {};'.format(dtype, names[op], 1)
            elif op.func == numbers.Rational:
                code = '{} {} = {};'.format(dtype, names[op], float(op.p)/op.q)
            elif op.func == relational.StrictLessThan:
                code = 'int {} = {} < {};'.format(names[op], names[op.args[0]], names[op.args[1]])
            elif op.func == Piecewise:
                code = """
                {4} {0};
                if ({1}) 
                    {0} = {2};
                else 
                    {0} = {3};
                """.format(names[op], names[op.args[0].args[1]], names[op.args[0].args[0]], names[op.args[1].args[0]], dtype)
                #code = '{} {} = {};'.format(dtype, names[op], 0.5)
            elif op.func == Extract:
                a, b = op.args
                tensorIndex = self._inputTensorIndices[a]
                code = '{} {} = *({} + {}*{} + {});'.format(dtype, names[op], tensorIndex[0], names[b], tensorIndex[1], tensorIndex[2])
            elif op.func == Collate:
                tensorIndex = self._outputTensorIndices[op]
                n = len(op.args)/2
                for i in range(0, n):
                    a, b = op.args[2*i], op.args[2*i+1]
                    code += '*({} + {}*{} + {}) = {};\n\t\t'.format(tensorIndex[0], names[b], tensorIndex[1], tensorIndex[2], names[a])
            else:
                if op.func not in [boolalg.BooleanTrue, piecewise.ExprCondPair]:
                    raise Exception("ss", op.func)
            if op in self._outputTensorIndices:
                tensorIndex = self._outputTensorIndices[op]
                if not tensorIndex[3]:
                    code += ' *({} + i*{} + {}) = {};'.format(tensorIndex[0], tensorIndex[1], tensorIndex[2], names[op])
            codeFile.write('\t\t' + code + '\n')
            #print op.func, len(op.args)
        codeFile.write('\t}\n')
        codeFile.write('\tlong long end = current_timestamp(); mil += end-start; printf("c module: %lld\\n", mil);\n')
        codeFile.write('}\n')
        codeFile.close()

        return

    def _topologicalSort(self, outputs, children):
        output = Container()
        output.args = tuple(outputs)
        for out in outputs:
            children[out] += 1
        children[output] = 1
        #print children.values()
        sortedOps = []
        def _sort(outputs):
            for out in outputs:
                #children[out] -= 1
                children[out] = max(children[out]-1, 0)
            for out in outputs:
                if children[out] == 0:
                    sortedOps.append(out)
                    _sort(out.args)
        _sort([output])
        #print children.values()
        return sortedOps[1:][::-1]

    def _diff(self, output, inputs):
        # optimizations: clubbing, common subexpression elimination
        # rely on compiler to figure it out?
        gradients = {}
        gradients[output] = 1.
        children = self._children.copy()
        for out in self._outputs:
            children[out] += 1
        def _diffFunc(out):
            grads = []
            if out.func == add.Add:
                for inp in out.args:
                    grads.append(gradients[out])
            elif out.func == mul.Mul:
                for index, inp in enumerate(out.args):
                    factors = [gradients[out]] + list(out.args[:index]) + list(out.args[index+1:])
                    grads.append(gradients[out]*prod(out.args))
            elif out.func == power.Pow:
                x, n = out.args
                grads.append(n*x**(n-1))
                grads.append(None)
            elif out.func == Piecewise:
                cond = out.args[0].args[1]
                x1, x2 = out.args[0].args[0], out.args[1].args[0]
                grads.append(Piecewise((gradients[out], cond), (0, True)))
                grads.append(Piecewise((gradients[out], ~cond), (0, True)))
            elif out.func == piecewise.ExprCondPair:
                grads.append(gradients[out])
                grads.append(None)
            else:
                if (out.func not in [Scalar]) and \
                   (len(out.args) > 0):
                    raise Exception(out.func, len(out.args))
            for grad, inp in zip(grads, out.args):
                if grad is None:
                    continue
                if inp not in gradients:
                    gradients[inp] = grad
                else:
                    gradients[inp] += grad
                children[inp] -= 1
                if children[inp] == 0:
                    _diffFunc(inp)
        _diffFunc(output)
        return [gradients.get(inp, None) for inp in inputs]

    @classmethod
    def clean(self):
        try:
            os.remove(codeDir + 'code.c')
        except:
            pass

    @classmethod
    def compile(self):
        subprocess.check_call(['make'], cwd=codeDir)
        from .gencode import interface as mod
        Function._module = mod
        #Function._module = ctypes.cdll.LoadLibrary(codeDir + 'interface.so')

    #def __call__(self, inputs, outputs):
    #    func = Function._module[self.name] 
    #    args = [ctypes.c_int(inputs[0].shape[0])] + \
    #            [np.ctypeslib.as_ctypes(x) for x in inputs] + \
    #            [np.ctypeslib.as_ctypes(x) for x in outputs]
    #    func(*args)

class Variable(object):
    pass


