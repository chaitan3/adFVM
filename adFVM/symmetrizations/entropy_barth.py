from sympy import Array, Matrix
from sympy import sqrt, log, symbols, refine, Q, exp
from sympy import simplify as _simplify
from sympy import derive_by_array, tensorproduct, tensorcontraction, permutedims

def diff(a, b):
    D = derive_by_array(a, b)
    dims = tuple([x + len(b.shape) for x in range(0, len(a.shape))]) + \
           tuple(range(0, len(b.shape)))
    return permutedims(D, dims)

g = symbols('g')

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

U = [r, ru1, ru2, ru3, rE]

u1 = ru1/r
u2 = ru2/r
u3 = ru3/r
q2 = u1*u1 + u2*u2 + u3*u3
g1 = g-1
p = g1*(rE - r*q2/2)
a2 = g*p/r
H = a2/g1 + q2/2

F1 = [ru1, ru1*u1 + p, ru1*u2, ru1*u3, u1*(rE + p)]
F2 = [ru2, ru2*u1 + p, ru2*u2, ru2*u3, u2*(rE + p)]
F3 = [ru3, ru3*u1 + p, ru3*u2, ru3*u3, u3*(rE + p)]

A1 = diff(F1, U)
A2 = diff(F2, U)
A3 = diff(F3, U)

A0 = [[r, ru1, ru2, ru3, rE],
      [ru1, ru1*u1, ru1*u2, ru1*u3, ru1*H],
      [ru2, ru2*u1, ru2*u2, ru2*u3, ru2*H],
      [ru3, ru3*u1, ru3*u2, ru3*u3, ru3*H],
      [rE, ru1*H, ru2*H, ru3*H, r*H*H - a2*p/g1]]

#def simplify(a):
#    return _simplify(refine(a, Q.positive(rho) & \
#                     Q.positive(p) & \
#                     Q.positive(pref) & \
#                     Q.positive(g-1)))
def simplify(a):
    return a
    #return _simplify(a)

B = diff(A0, U)
A1U = diff(A1, U)
A2U = diff(A2, U)
A3U = diff(A3, U)

with open('entropy_barth_numerical.py', 'w') as f:
    f.write("""
import numpy as np
from numpy import exp, log
def expression(rho, rhoU, rhoE, g):
    B = np.array({})
    A1U = np.array({})
    A2U = np.array({})
    A3U = np.array({})
    return B, A1U, A2U, A3U
    """.format(B, A1U, A2U, A3U))

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
