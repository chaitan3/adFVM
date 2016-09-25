import numpy as np

import tensor
from func import Function
#from . import tensor as ad

import function
a = np.ones(4)
print function.interface(a)

a = tensor.vector()
b = tensor.vector()
c = a+b
d = tensor.vector()

f = Function([a, b, d], [2*c])

print f(np.random.rand(2), np.random.rand(2), np.random.rand(2))
