import operator
import numbers
dtype = 'scalar'

def prod(factors):
    return reduce(operator.mul, factors, 1)

SQRT = 123412341234

class ArithBase(object):
    def _binaryOp(self, b, op):
        raise NotImplementedError(self)

    def _unaryOp(self, op):
        raise NotImplementedError(self)

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

class Scalar(ArithBase):
    _index = 0
    def __init__(self):
        index = Scalar._index
        Scalar._index += 1
        self.name = 'Scalar_{}'.format(index)
        self.args = tuple()

    def _binaryOp(self, b, op, comparison=False):
        if isinstance(b, numbers.Number):
            b = ConstantOp(b)
        return binaryOpClass[op](op, self, b, comparison)

    def _unaryOp(self, op):
        return unaryOpClass[op](op, self)

    # define unique hash function

class OpBase(Scalar):
    def c_code(self, *args):
        raise NotImplementedError(self)

    def grad(self, gradients):
        raise NotImplementedError(self)

class ConstantOp(OpBase):
    def __init__(self, const):
        self.constant = const
        self.args = tuple()

    def c_code(self, names):
        return '{} {} = {};'.format(dtype, names[self], self.constant)

    def grad(self, gradients):
        return []
        

class BinaryOp(OpBase):
    def __init__(self, op, a, b, comparison=False):
        self.op = op
        self.args = (a, b)
        self.comparison = comparison

    def c_code(self, names):
        argNames = [names[inp] for inp in self.args]
        op = binaryOpString.get(self.op, None)
        typeString = 'int' if self.comparison else dtype
        return '{} {} = {};'.format(typeString, names[self], op.join(argNames))

class AddOp(BinaryOp):
    def grad(self, gradients):
        grads = []
        for inp in self.args:
            grads.append(gradients[self])
        return grads

class SubOp(BinaryOp):
    def grad(self, gradients):
        return [gradients[self], -gradients[self]]

class MulOp(BinaryOp):
    def grad(self, gradients):
        grads = []
        for index, inp in enumerate(self.args):
            factors = [gradients[self]] + list(self.args[:index]) + list(self.args[index+1:])
            grads.append(gradients[self]*prod(self.args))
        return grads

class DivOp(BinaryOp):
    def grad(self, gradients):
        x, y = self.args
        z = gradients[self]
        return [z/y, -(z*x)/(y*y)]

class LessThanOp(BinaryOp):
    pass

class PowerOp(BinaryOp):
    def c_code(self, names):
        argNames = [names[inp] for inp in self.args]
        return '{} {} = pow({}, {});'.format(dtype, names[self], argNames[0], argNames[1]);

    def grad(self, gradients):
        x, n = self.args
        return [n*x**(n-1), None]

class UnaryOp(OpBase):
    def __init__(self, op, a):
        self.op = op
        self.args = (a,)

    def c_code(self, names):
        argNames = [names[inp] for inp in self.args]
        op = unaryOpString[self.op]
        return '{} {} = {}({});'.format(dtype, names[self], op, argNames[0]);

class NegOp(UnaryOp):
    def grad(self, gradients):
        return [-gradients[self]]

class AbsOp(UnaryOp):
    def grad(self, gradients):
        x = self.args[0]
        return [ConditionalOp(x < 0, -gradients[self], gradients[self])]

class SqrtOp(UnaryOp):
    def grad(self, gradients):
        x = self.args[0]
        return [gradients[self]/(2*self)]

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

binaryOpString = {operator.div: '/',
                  operator.sub: '-',
                  operator.mul: '*',
                  operator.add: '+',
                  operator.lt: '<'}

binaryOpClass = {operator.div: DivOp,
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


    def grad(self, gradients):
        cond, x1, x2 = self.args
        grads = [None]
        zero = ConstantOp(0)
        grads.append(ConditionalOp(cond, gradients[self], zero))
        grads.append(ConditionalOp(~cond, gradients[self], zero))
        return grads

class Extract(OpBase):
    def __init__(self, *args):
        self.args = tuple(args)
                                
    def grad(self, gradients):
        x, b = self.args        
        return [Collate(gradients[self], b), None]
        
class Collate(OpBase):
    def __init__(self, *args):
        self.args = tuple(args)
    
    def grad(self, gradients):
        n = len(self.args)/2
        grads = []
        for i in range(0, n):
            a, b = self.args[2*i], self.args[2*i+1]
            grads.append(Extract(gradients[self], b))
            grads.append(None)
        return grads

