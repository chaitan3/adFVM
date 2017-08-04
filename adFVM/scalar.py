import operator
import numbers
dtype = 'scalar'

import functools
def prod(factors):
    return functools.reduce(operator.mul, factors, 1)

class Container(object):
    pass

SQRT = 123412341234

class ArithBase(object):
    def _binaryOp(self, b, op):
        raise NotImplementedError(self, b, op)

    def _unaryOp(self, op):
        raise NotImplementedError(self, b, op)

    def __add__(self, b):
        return self._binaryOp(b, operator.add)

    def __radd__(self, b):
        return self.__add__(b)

    def __sub__(self, b):
        return self._binaryOp(b, operator.sub)

    def __rsub__(self, b):
        return (-self)._binaryOp(b, operator.add)

    def __mul__(self, b):
        return self._binaryOp(b, operator.mul)

    def __rmul__(self, b):
        return self.__mul__(b)

    def __div__(self, b):
        return self.__truediv__(b)

    def __truediv__(self, b):
        return self._binaryOp(b, operator.truediv)

    def __neg__(self):
        return self._unaryOp(operator.neg)

    def __pow__(self, b):
        return self._binaryOp(b, operator.pow)

    def __lt__(self, b):
        return self._binaryOp(b, operator.lt, True)

    def __abs__(self):
        return self._unaryOp(operator.abs)

    def __invert__(self):
        return self._unaryOp(operator.invert)

    def sqrt(self):
        return self._unaryOp(SQRT)

class IntegerScalar(ArithBase):
    _index = 0
    def __init__(self):
        index = IntegerScalar._index
        IntegerScalar._index += 1
        self.name = 'Integer_{}'.format(index)
        self.args = tuple()
        self.dtype = 'integer'

    #def _binaryOp(self, b, op, comparison=False):
    #    assert isinstance(b, IntegerScalar)
    #    return binaryOpClass[op](op, self, b, comparison)

class Scalar(ArithBase):
    _index = 0
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self):
        index = Scalar._index
        Scalar._index += 1
        self.name = 'Scalar_{}'.format(index)
        self.args = tuple()
        self.dtype = dtype

    def _binaryOp(self, b, op, comparison=False):
        if isinstance(b, numbers.Number):
            b = ConstantOp(b)
        return binaryOpClass[op](op, self, b, comparison)

    def _unaryOp(self, op):
        return unaryOpClass[op](op, self)

class ConstScalar(Scalar):
    pass

class OpBase(Scalar):
    _cache = {}
    def __new__(cls, *args, **kwargs):
        assert len(kwargs) == 0
        key = (cls,) + args
        if key in OpBase._cache:
            obj = OpBase._cache[key]
        else:
            obj = Scalar.__new__(cls, *args, **kwargs)
            OpBase._cache[key] = obj
        return obj

    @staticmethod
    def clear_cache():
        OpBase._cache = {}

    def c_code(self, *args):
        raise NotImplementedError(self)

    def grad(self, gradient):
        raise NotImplementedError(self)

class ConstantOp(OpBase):
    def __init__(self, constant):
        self.constant = constant
        self.args = tuple()
        self.dtype = dtype
        if isinstance(constant, int):
            self.dtype = 'integer'

    def c_code(self, names):
        return '{} {} = {};'.format(self.dtype, names[self], self.constant)

    def grad(self, gradient):
        return []
        

class BinaryOp(OpBase):
    def __init__(self, op, a, b, comparison=False):
        self.op = op
        self.args = (a, b)
        self.comparison = comparison

    def c_code(self, names):
        argNames = [names[inp] for inp in self.args]
        op = ' ' + binaryOpString.get(self.op) + ' ' 
        typeString = 'int' if self.comparison else dtype
        return '{} {} = {};'.format(typeString, names[self], op.join(argNames))

class AddOp(BinaryOp):
    def grad(self, gradient):
        grads = []
        for inp in self.args:
            grads.append(gradient)
        return grads

class SubOp(BinaryOp):
    def grad(self, gradient):
        return [gradient, -gradient]

class MulOp(BinaryOp):
    def grad(self, gradient):
        grads = []
        for index, inp in enumerate(self.args):
            factors = [gradient] + list(self.args[:index]) + list(self.args[index+1:])
            grads.append(prod(factors))
        return grads

class DivOp(BinaryOp):
    def grad(self, gradient):
        x, y = self.args
        z = gradient
        return [z/y, -(z*x)/(y*y)]

class LessThanOp(BinaryOp):
    pass

class PowerOp(BinaryOp):
    def c_code(self, names):
        argNames = [names[inp] for inp in self.args]
        return '{} {} = pow({}, {});'.format(dtype, names[self], argNames[0], argNames[1]);

    def grad(self, gradient):
        x, n = self.args
        return [gradient*n*x**(n-1), None]

class UnaryOp(OpBase):
    def __init__(self, op, a):
        self.op = op
        self.args = (a,)

    def c_code(self, names):
        argNames = [names[inp] for inp in self.args]
        op = unaryOpString[self.op]
        return '{} {} = {}({});'.format(dtype, names[self], op, argNames[0]);

class NegOp(UnaryOp):
    def grad(self, gradient):
        return [-gradient]

class AbsOp(UnaryOp):
    def grad(self, gradient):
        x = self.args[0]
        return [ConditionalOp(x < 0, -gradient, gradient)]

class SqrtOp(UnaryOp):
    def grad(self, gradient):
        x = self.args[0]
        return [gradient/(2*self)]

class InvertOp(UnaryOp):
    pass

unaryOpClass = {operator.abs: AbsOp,
                operator.neg: NegOp,
                operator.invert: InvertOp,
                SQRT: SqrtOp}

unaryOpString = {operator.abs: 'abs',
                operator.neg: '-',
                operator.invert: '!',
                SQRT: 'sqrt'}

binaryOpString = {operator.truediv: '/',
                  operator.sub: '-',
                  operator.mul: '*',
                  operator.add: '+',
                  operator.lt: '<'}

binaryOpClass = {operator.truediv: DivOp,
                  operator.sub: SubOp,
                  operator.mul: MulOp,
                  operator.add: AddOp,
                  operator.lt: LessThanOp,
                  operator.pow: PowerOp}

class ConditionalOp(OpBase):
    def __init__(self, cond, a, b):
        self.args = (cond, a, b)

    def c_code(self, names):
        cond, x1, x2 = self.args
        return """
                {4} {0};
                if ({1}) 
                    {0} = {2};
                else 
                    {0} = {3};
                """.format(names[self], names[cond], names[x1], names[x2], dtype)


    def grad(self, gradient):
        cond, x1, x2 = self.args
        grads = [None]
        zero = ConstantOp(0.)
        grads.append(ConditionalOp(cond, gradient, zero))
        grads.append(ConditionalOp(~cond, gradient, zero))
        return grads

class Extract(OpBase):
    def __init__(self, *args):
        self.args = tuple(args)
                                
    def grad(self, gradient):
        x, b = self.args        
        return [Collate(gradient, b), None]
        
class Collate(OpBase):
    def __init__(self, *args):
        self.args = tuple(args)
    
    def grad(self, gradient):
        n = len(self.args)//2
        grads = []
        for i in range(0, n):
            a, b = self.args[2*i], self.args[2*i+1]
            grads.append(Extract(gradient, b))
            grads.append(None)
        return grads

