from sympy import Array, Matrix, lambdify
from sympy.utilities.autowrap import ufuncify
from sympy import sqrt, log, symbols, refine, Q, exp
from sympy import simplify as _simplify
from sympy import derive_by_array, tensorproduct, tensorcontraction, permutedims
import numpy as np

#def simplify(a):
#    return _simplify(refine(a, Q.positive(rho) & \
#                     Q.positive(p) & \
#                     Q.positive(pref) & \
#                     Q.positive(g-1)))
def simplify(a):
    return _simplify(a)
    #return a


def diff(a, b):
    D = derive_by_array(a, b)
    dims = tuple([x + len(b.shape) for x in range(0, len(a.shape))]) + \
           tuple(range(0, len(b.shape)))
    return permutedims(D, dims)

g = symbols('g')
zero = symbols('zero')
one = symbols('one')

#rho = symbols('rho')
#u1 = symbols('u1')
#u2 = symbols('u2')
#u3 = symbols('u3')
#p = symbols('p')

#F1 = [rho*u1, rho*u1*u1 + p, rho*u1*u2, rho*u1*u3, u1*(rho*)]
#F2 =
#F3 = 

r = symbols('r')
ru1 = symbols('ru1')
ru2 = symbols('ru2')
ru3 = symbols('ru3')
rE = symbols('rE')

U = Array([r, ru1, ru2, ru3, rE])

u1 = ru1/r
u2 = ru2/r
u3 = ru3/r
q2 = u1*u1 + u2*u2 + u3*u3
g1 = g-1
p = g1*(rE - r*q2/2)
a2 = g*p/r
H = a2/g1 + q2/2

F = Array([[ru1, ru2, ru3],
          [ru1*u1 + p, ru2*u1, ru3*u1],
          [ru1*u2, ru2*u2 + p, ru3*u2],
          [ru1*u3, ru2*u3, ru3*u3 + p],
          [u1*(rE + p), u2*(rE + p), u3*(rE+p)]])

A = simplify(diff(F, U))

A0 = Array([[r, ru1, ru2, ru3, rE],
      [ru1, ru1*u1 + p, ru1*u2, ru1*u3, ru1*H],
      [ru2, ru2*u1, ru2*u2 + p, ru2*u3, ru2*H],
      [ru3, ru3*u1, ru3*u2, ru3*u3 + p, ru3*H],
      [rE, ru1*H, ru2*H, ru3*H, r*H*H - a2*p/g1]])
A0 = simplify(A0)

A0U = simplify(diff(A0, U))
AU = simplify(diff(A, U))

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def sanitize(arr):
    arr2 = []
    for x in arr:
        if isinstance(x, float) and x != 0.0:
            arr2.append(one*x)
        elif x == 0:
            arr2.append(zero)
        elif x == 1:
            arr2.append(one)
        else:
            arr2.append(x)
    return arr2

print 'A:', A.shape
print 'AU:', AU.shape
print 'A0:', A0.shape
print 'A0U:', A0U.shape

A = sanitize(flatten(A))
AU = sanitize(flatten(AU))
A0 = sanitize(flatten(A0))
A0U = sanitize(flatten(A0U))

with open('entropy_barth_numerical.py', 'w') as f:
    f.write("""
import numpy as np
def expression(r, ru1, ru2, ru3, rE, g, one, zero):
    return {}, \
            {}, \
            {}, \
            {}
    """.format(A0U, AU, A0, A))

#with open('entropy_hughes_gen_code.py', 'w') as f:
#    f.write("""
#flatten = lambda l: [item for sublist in l for item in sublist]
#def expression(g, rho, rhoU, rhoE, gradrho, gradU, gradp):
#    u1 = U[0]
#    u2 = U[0]
#    u3 = U[0]
#    rhox1 = gradrho[0]
#    rhox2 = gradrho[1]
#    rhox3 = gradrho[2]
#    px1 = gradp[0]
#    px2 = gradp[1]
#    px3 = gradp[2]
#    u1x1 = gradU[0,0]
#    u1x2 = gradU[0,1]
#    u1x3 = gradU[0,2]
#    u2x1 = gradU[1,0]
#    u2x2 = gradU[1,1]
#    u2x3 = gradU[1,2]
#    u3x1 = gradU[2,0]
#    u3x2 = gradU[2,1]
#    u3x3 = gradU[2,2]
#    M1 = flatten({})
#    M2 = flatten({})
#    return M1, M2
#    """.format(M1s, M2s))
