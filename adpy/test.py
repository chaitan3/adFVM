import numpy as np

import tensor
from func import Function
#from . import tensor as ad

a = tensor.vector()
b = tensor.vector()
c = a+b
d = tensor.vector()

f = Function([a, b, d], [2*c])

print f(100., 2, 4)

