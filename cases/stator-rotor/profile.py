
from scipy import interpolate, integrate
from numpy import *

c = 0.067647
pitch = 0.0575
def get_length(pat, coords):
    t = interpolate.UnivariateSpline(pat[0], pat[1], s=0)
    d = t.derivative()
    s = []
    for i in coords:
        s.append(integrate.quad(lambda x: sqrt(1 + d(x)**2), 0, i)[0])
    return array(s)

f = open('/home/talnikar/adFVM/cases/vane/mesh/coords.txt', 'r')
v1 = []
v2 = []
n = 18
for l in f.readlines():
    t = l.split()
    v1.append(t[0])
    v2.append(t[1])
f.close()
v1 = array(v1, dtype=float)/1000
v2 = array(v2, dtype=float)/1000
suction = [v1[:n], v2[:n]]
pressure = [v1[n:], v2[n:]]
