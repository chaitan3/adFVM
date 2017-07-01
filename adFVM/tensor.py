import numpy as np
#class Variable:
#    pass
from sympy import Function
from sympy import Symbol, sqrt, Piecewise, lambdify
from sympy.utilities.autowrap import ufuncify
from sympy.core import add, mul, power, numbers, relational, cache
from sympy.functions.elementary import piecewise
from sympy.logic import boolalg
import operator

dtype = 'scalar'
from . import config

import os, sys, subprocess, shutil
import ctypes

scriptDir = os.path.dirname(os.path.realpath(__file__))

class Container(object):
    pass

def prod(factors):
    return reduce(operator.mul, factors, 1)


class Extract(Function):
    @classmethod
    def _should_evalf(cls, arg):
        return -1
class Collate(Function):
    @classmethod
    def _should_evalf(cls, arg):
        return -1

#Extract = Function('extract')
#Collate = Function('collate')

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
        self.dtype = dtype
        
        if isinstance(self.scalars[0], IntegerScalar):
            self.dtype = 'integer'

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
        return self.__mul__(b)

    def __div__(self, b):
        return self._binaryOp(b, operator.div)

    def __neg__(self):
        return Tensor(self.shape, [-x for x in self.scalars])

    def __pow__(self, b):
        return Tensor(self.shape, [x**b for x in self.scalars])

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
        assert self.shape == b.shape
        res = sum([self.scalars[i]*b.scalars[i] for i in range(0, self.size)])
        return Tensor((1,), [res])

    def tensordot(self, b):
        assert self.shape == (3,3)
        assert b.shape == (3,)
        res = []
        for i in range(0, 3):
            res.append(sum([self.scalars[i*3 + j]*b.scalars[j] for j in range(0, 3)]))
        return Tensor((3,), res)

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

    def trace(self):
        assert self.shape == (3, 3)
        res = [sum(self.scalars)]
        return Tensor((1,), res)

    def transpose(self):
        assert self.shape == (3, 3)
        res = []
        for i in range(0, 3):
            for j in range(0, 3):
                res.append(j*3 + i)
        return Tensor(self.shape, res)

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
        return CellTensor(shape, res)

    @classmethod
    def switch(cls, cond, ret1, ret2):
        assert ret1.shape == (1,)
        assert ret2.shape == (1,)
        res = [Piecewise((ret1.scalars[0], cond), (ret2.scalars[0], True))]
        return cls(ret1.shape, res)

class CellTensor(Tensor):
    pass

class TensorFunction(object):
    _index = 0
    _module = None
    codeDir = os.path.dirname(__file__) + '/gencode/'

    def __init__(self, name, inputs, outputs, grad=True):
        index = TensorFunction._index
        TensorFunction._index += 1
        #self.name = 'Function_{}'.format(index)
        self.name = 'Function_{}'.format(name)
        print(self.name)
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

        _outputs = [x for x in self._outputs if x is not None]
        self._children = self._getChildren(_outputs)
        self._genCode(self._inputs, _outputs, self._children.copy())
        #grad = False
        if grad:
            self.grad = self._getAdjoint()

    def _getAdjoint(self):
        gradOutputs = [x.__class__(x.shape) for x in self._outputTensors]

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
            outputs.append(inp.__class__(inp.shape, outputScalars[i:i+n]))
            outputs[-1].dtype = inp.dtype
            i += n
        inputs = self._inputTensors + gradOutputs
        return TensorFunction(self.name.split('_')[1] + '_grad', inputs, outputs, grad=False)

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

    def _topologicalSort(self, outputs, children):
        output = Container()
        output.args = tuple(outputs)
        for out in outputs:
            children[out] += 1
        children[output] = 0
        #print children.values()
        sortedOps = []
        def _sort(out):
            sortedOps.append(out)
            for inp in out.args:
                #print hash(inp), children[inp]
                children[inp] -= 1
                if children[inp] == 0:
                    _sort(inp)
        _sort(output)
        #print children.values()
        return sortedOps[1:][::-1]

    def _diff(self, outputs, inputs, gradients=None):
        if gradients is None:
            gradients = {}
            for out in outputs:
                gradients[out] = 1.
        #children = self._getChildren(outputs)
        children = self._children.copy()
        #print children.values()
        def _diffFunc(out):
            #print out.func
            assert children[out] == 0
            grads = []
            if gradients[out] == None:
                grads = [None]*len(out.args)
            elif out.func == add.Add:
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
            elif out.func == Extract:
                x, b = out.args
                #print 'here', gradients[out].func, b.func
                grads.append(Collate(gradients[out], b))
                grads.append(None)
            elif out.func == Collate:
                n = len(out.args)/2
                for i in range(0, n):
                    a, b = out.args[2*i], out.args[2*i+1]
                    grads.append(Extract(gradients[out], b))
                    grads.append(None)
            else:
                if (len(out.args) > 0):
                    raise Exception(out.func, len(out.args))
            assert len(grads) == len(out.args)
            for grad, inp in zip(grads, out.args):
                if inp not in gradients or gradients[inp] is None:
                    gradients[inp] = grad
                elif grad is not None:
                    # combining collates
                    if gradients[inp].func == Collate:
                        #print gradients[inp].func, grad.func
                        args = gradients[inp].args + grad.args
                        gradients[inp] = Collate(*args)
                    else:
                        gradients[inp] += grad
                children[inp] -= 1
                if children[inp] == 0:
                    _diffFunc(inp)
        for out in outputs:
            _diffFunc(out)
        #print children.values()
        return [gradients.get(inp, None) for inp in inputs]

    def _genCode(self, inputs, outputs, children):
        sortedOps = self._topologicalSort(outputs, children)
        codeFile = open(self.codeDir + 'code.cpp', 'a')

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
                _power = op.args[1]
                if _power.func == numbers.Float:
                    expr = 'pow({},{})'.format(argNames[0], argNames[1])
                else:
                    r = (_power.p, _power.q)
                    if r == (2, 1):
                        expr = '{0}*{0}'.format(argNames[0])
                    elif r == (-2, 1):
                        expr = '1./({0}*{0})'.format(argNames[0])
                    elif r == (-1, 1):
                        expr = '1./{}'.format(argNames[0])
                    elif r == (1, 2):
                        expr = 'sqrt({})'.format(argNames[0])
                    elif r == (-1, 2):
                        expr = '1./sqrt({})'.format(argNames[0])
                    else:
                        expr = 'pow({},{})'.format(argNames[0], argNames[1])
                code = '{} {} = {};'.format(dtype, names[op], expr)

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
            elif op.func == numbers.NegativeOne:
                code = 'const {} {} = {};'.format(dtype, names[op], -1)
            elif op.func == numbers.Rational:
                code = 'const {} {} = {};'.format(dtype, names[op], float(op.p)/op.q)
            elif op.func == relational.StrictLessThan:
                code = 'int {} = {} < {};'.format(names[op], names[op.args[0]], names[op.args[1]])
            elif op.func == relational.GreaterThan:
                code = 'int {} = {} >= {};'.format(names[op], names[op.args[0]], names[op.args[1]])
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
                #print(op.func, hash(op), len(op.args))
                tensorIndex = self._outputTensorIndices[op]
                n = len(op.args)/2
                #code += '// hash {}: {}\n'.format(n, hash(op))
                for i in range(0, n):
                    a, b = op.args[2*i], op.args[2*i+1]
                    code += '*({} + {}*{} + {}) += {};\n\t\t'.format(tensorIndex[0], names[b], tensorIndex[1], tensorIndex[2], names[a])
            else:
                if op.func not in [boolalg.BooleanTrue, piecewise.ExprCondPair]:
                    raise Exception("ss", op.func)
            #if op.func not in [Collate, Scalar, boolalg.BooleanTrue, piecewise.ExprCondPair]:
            #    code += '//{}\n'.format(op.func)
            #    code += 'if (i == 0) cout << "{}" << " " << {} << endl;\n'.format(names[op], names[op])

            if op in self._outputTensorIndices:
                tensorIndex = self._outputTensorIndices[op]
                if not tensorIndex[3]:
                    code += '\n\t\t*({} + i*{} + {}) += {};'.format(tensorIndex[0], tensorIndex[1], tensorIndex[2], names[op])
            codeFile.write('\t\t' + code + '\n')
            #print op.func, len(op.args)
        codeFile.write('\t}\n')
        #codeFile.write('\tlong long end = current_timestamp(); mil += end-start; printf("c module {}: %lld\\n", mil);\n'.format(self.name))
        codeFile.write('}\n')
        codeFile.close()

        return

    @classmethod
    def createCodeDir(self, case):
        self.codeDir = case + 'gencode/'
        if config.user.compile:
            assert not os.path.exists(self.codeDir)
            shutil.copytree(scriptDir + '/gencode', self.codeDir)

    @classmethod
    def clean(self):
        #try:
        #    os.remove(self.codeDir + 'code.cpp')
        #except:
        #    pass
        with open(self.codeDir + 'code.cpp', 'a') as f:
            f.write('#include "code.hpp"\n')

    @classmethod
    def compile(self):
        if config.user.compile:
            subprocess.check_call(['make'], cwd=self.codeDir)
        sys.path.append(self.codeDir)
        import interface as mod
        TensorFunction._module = mod


    #def __call__(self, inputs, outputs):
    #    func = Function._module[self.name] 
    #    args = [ctypes.c_int(inputs[0].shape[0])] + \
    #            [np.ctypeslib.as_ctypes(x) for x in inputs] + \
    #            [np.ctypeslib.as_ctypes(x) for x in outputs]
    #    func(*args)

class Variable(object):
    pass


