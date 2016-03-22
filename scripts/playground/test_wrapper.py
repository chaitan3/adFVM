import theano as T
import numpy as np

class FunctionWrapper(T.Op):
    def __init__(self, function):
        self.function = function

    def make_node(self, *inputs):
        #assert hasattr(self, '_props')
        # HACK
        #print T.Apply(self, inputs, [x.type() for x in inputs[:3]])
        return T.Apply(self, inputs, [inputs[0].type()])

    def perform(self, node, inputs, output_storage):
        out =  self.function(*inputs)
        #for i in range(0, len(out)):
        #    output_storage[i][0] = out[i]
        output_storage[0][0] = out

a = T.tensor.scalar()
b = T.tensor.scalar()
c = T.tensor.scalar()

d = a+b+c
x = T.function([a,b,c],d)
print x(1,2,3)

y = FunctionWrapper(x)
g = y(a,b,c) + y(c,b,a) + c
z = T.function([a,b,c],g)
print z(1,2,3)

a = T.tensor.scalar()
bn = T.tensor.scalar()
b = T.gradient.disconnected_grad(bn)
#b = T.tensor.scalar()

c = a + b
print T.function([a,b], c)(1,2)
#d = T.tensor.grad(c, [a,b])
d = T.tensor.grad(c, [a,b])
print d
print T.function([a,b], d)(1,2)

print T.function([a,bn], c)(1,2)
d = T.tensor.grad(c, [a,bn], disconnected_inputs='ignore')
print d
print T.function([a,bn], d)(1,2)
