from sympy import Array, Matrix
from sympy import sqrt, log, symbols, refine, Q, exp
from sympy import simplify as _simplify
from sympy import derive_by_array, tensorproduct, tensorcontraction, permutedims

def tensordot(a, b):
    s1 = a.shape
    s2 = b.shape
    assert s1[-1] == s2[0]
    s3 = s1[:-1] + s2[1:]
    k = len(s1)-1
    return tensorcontraction(tensorproduct(a, b), (k,k+1))

def diff(a, b):
    D = derive_by_array(a, b)
    dims = tuple([x + len(b.shape) for x in range(0, len(a.shape))]) + \
           tuple(range(0, len(b.shape)))
    return permutedims(D, dims)

g = symbols('g')
V1 = symbols('V1')
V2 = symbols('V2')
V3 = symbols('V3')
V4 = symbols('V4')
V5 = symbols('V5')
pref = symbols('pref')
#re = symbols('re')

#def simplify(a):
#    return _simplify(refine(a, Q.positive(rho) & \
#                     Q.positive(p) & \
#                     Q.positive(pref) & \
#                     Q.positive(g-1)))
def simplify(a):
    return a
    #return _simplify(a)

g1 = g-1
k1 = (V2*V2+V3*V3+V4*V4)/(2*V5)
k2 = k1-g
k3 = k1*k1 -2*g*k1 + g
k4 = k2-g1
k5 = k2*k2 - g1*(k1+k2)
c1 = g1*V5-V2*V2
c2 = g1*V5-V3*V3
c3 = g1*V5-V4*V4
d1 = -V2*V3
d2 = -V2*V4
d3 = -V3*V4
e1 = V2*V5
e2 = V3*V5
e3 = V4*V5
s=g-V1+k1
#re = pref*exp(-s/g1)*(g1/(-V5*pref)**g)**(1/g1)
re = (1/pref)**(1/g1)*exp(-s/g1)*(g1/(-V5*pref)**g)**(1/g1)
print 'here'

A = Array([[[e1*V5, c1*V5,d1*V5, d2*V5, k2*e1],
            [c1*V5, -(c1+2*g1*V5)*V2, -c1*V3, -c1*V4, c1*k2+g1*V2*V2],
            [d1*V5,-c1*V3, -c2*V2, -d1*V4,k4*d1],
            [d2*V5,-c1*V4,-d1*V4,-c3*V2,k4*d2],
            [k2*e1,c1*k2+g1*V2*V2,k4*d1,k4*d2,k5*V2]],
           [[e2*V5,d1*V5, c2*V5,d3*V5,k2*e2],
            [d1*V5,-c1*V3,-c2*V2,-d1*V4,k4*d1],
            [c2*V5,-c2*V2,-(c2+2*g1*V5)*V3, -c2*V4,c2*k2+g1*V3*V3],
            [d3*V5,-d1*V4,-c2*V4,-c3*V3,k4*d3],
            [k2*e2,k4*d1,c2*k2+g1*V3*V3,k4*d3,k5*V3]],
           [[e3*V5,d2*V5,d3*V5,c3*V5,k2*e3],
            [d2*V5,-c1*V4,-d2*V3,-c3*V2,k4*d2],
            [d3*V5,-d2*V3,-c2*V4,-c3*V3,k4*d3],
            [c3*V5,-c3*V2,-c3*V3,-(c3+2*g1*V5)*V4,c3*k2+g1*V4*V4],
            [k2*e3,k4*d2,k4*d3,c3*k2+g1*V4*V4,k5*V4]]])
A = re/(g1*V5*V5)*A
A = simplify(A)
A0 = Array([[-V5*V5,e1,e2,e3,V5*(1-k1)],
            [e1,c1,d1,d2,V2*k2],
            [e2,d1,c2,d3,V3*k2],
            [e3,d2,d3,c3,V4*k2],
            [V5*(1-k1),V2*k2,V3*k2,V4*k2,-k3]])
A0 = re/(g1*V5)*A0
A0 = simplify(A0)
A0I = Array([[k1**2+g,k1*V2,k1*V3,k1*V4,(k1+1)*V5],
             [k1*V2,V2**2-V5,-d1,-d2,e1],
             [k1*V3,-d1,V3**2-V5,-d3,e2],
             [k1*V4,-d2,-d3,V4**2-V5,e3],
             [(k1+1)*V5,e1,e2,e3,V5**2]])
A0I = -1/(re*V5)*A0I
A0I = simplify(A0I)
print 'here2'

#rho = symbols('rho')
#u1 = symbols('u1')
#u2 = symbols('u2')
#u3 = symbols('u3')
#p = symbols('p')
#q2 = u1*u1 + u2*u2 + u3*u3
#TwqI = Array([[1,0,0,0,0],[-u1/rho,1/rho,0,0,0], [-u2/rho, 0, 1/rho, 0,0],[-u3/rho,0,0,1/rho,0],[g1*q2/2, -g1*u1,-g1*u2,-g1*u3,g1]])
#Twv = A0
#Tqv = tensordot(TwqI, Twv)
#Tqv = simplify(Tqv)
print 'here3'

v = Array([V1,V2,V3,V4,V5])
B = diff(A, v)
B = simplify(B)
Y = diff(A0, v)
Y = simplify(Y)
print B
print Y

with open('entropy_hughes_numerical.py', 'w') as f:
    f.write("""
import numpy as np
from numpy import exp, log
def expression(rho, u1, u2, u3, p, V1, V2, V3, V4, V5, pref, g):
    B = np.array({}).transpose((4, 1, 0, 2, 3))
    Y = np.array({}).transpose((3, 0, 1, 2))
    A0I = np.array({}).transpose((2, 0, 1))
    A = np.array({}).transpose((3, 1, 0, 2))
    return B, Y, A0I, A
    """.format(B, Y, A0I, A))

#with open('entropy_hughes_gen_code.py', 'w') as f:
#    f.write("""
#flatten = lambda l: [item for sublist in l for item in sublist]
#def expression(g, rho, U, p, gradrho, gradU, gradp):
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
