import numpy as np
import theano as T
ad = T.tensor

a = np.random.rand(10, 3)
b = np.array([2,3,4,1], np.int32)

def red(a,ac,b):
    return b[ac-a:ac].min(axis=0)

x = ad.matrix()
y = ad.ivector()
res = T.scan(fn=red, sequences=[y, y.cumsum()], non_sequences=x, n_steps=y.shape[0])
c = res[0]
d = T.function([x,y], c)
print a, b
print d(a,b)
