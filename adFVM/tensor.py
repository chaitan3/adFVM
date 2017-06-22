import numpy as np
#class Variable:
#    pass
from sympy import Symbol, sqrt, Piecewise, lambdify
from sympy.utilities.autowrap import ufuncify

import operator

import operator
def prod(factors):
    return reduce(operator.mul, factors, 1)

class Scalar(Symbol):
    _index = 0
    def __new__(cls):
        index = Scalar._index
        Scalar._index += 1
        return Symbol.__new__(cls, 'Variable_{}'.format(index))

class Tensor(object):
    def __init__(self, shape, scalars=None):
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
        assert len(self.shape) == len(b.shape)
        if self.shape != b.shape:
            # broadcasting
            assert len(self.shape) == 1
            if b.shape[0] > self.shape[0]:
                self, b = b, self
            b = Tensor(self.shape, [b.scalars[0]]*self.shape[0])
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

    @classmethod
    def switch(cls, cond, ret1, ret2):
        assert ret1.shape == (1,)
        assert ret2.shape == (1,)
        res = [Piecewise((ret1.scalars[0], cond), (ret2.scalars[0], True))]
        return cls(ret1.shape, res)

def ZeroTensor(shape):
    return Tensor(shape, [0. for i in range(0, np.prod(shape))])

class Function(object):
    def __init__(self, inputs, outputs, mesh):
        inputs = inputs + [getattr(mesh, attr) for attr in mesh.gradFields]
        self._inputs = []
        for inp in inputs:
            self._inputs.extend(inp.scalars)
        self._outputs = []
        for out in outputs:
            self._outputs.extend(out.scalars)
        #self.func = lambdify(self._inputs, self._outputs)

        gradInputs = [Scalar() for x in self._outputs]
        scalarOutput = sum([x * y for x, y in zip(self._outputs, gradInputs)])
        self._children = self._getChildren(scalarOutput)
        gradient = self._diff(scalarOutput, self._inputs)

    def _getChildren(self, output):
        children = {}
        funcs = set()
        def _children_func(out):
            if out.func not in funcs:
                funcs.add(out.func)
            for inp in out.args:
                if inp in children:
                    children[inp] += 1
                else:
                    children[inp] = 1
                    _children_func(inp)
        _children_func(output)
        #print funcs
        return children

    def _diff(self, output, inputs):
        # optimizations: clubbing, common subexpression elimination
        # rely on compiler to figure it out?
        from sympy.core import add, mul, power
        from sympy.functions.elementary import piecewise
        gradients = {}
        gradients[output] = 1.
        children = self._children.copy()
        def _diff_func(out):
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
                    _diff_func(inp)
        _diff_func(output)
        return [gradients.get(inp, None) for inp in inputs]

    def __call__(self, *args):
        print(args)
        return self.func(*args)

class Variable(object):
    pass


