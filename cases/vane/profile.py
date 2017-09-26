
from scipy import interpolate, integrate, spatial
from numpy import *

c = 0.067647
pitch = 0.0575
def get_length(pat, coords):
    t, u = interpolate.splprep(pat, s=0)
    tev = linspace(0, 1, 10000)
    x = interpolate.splev(tev, t)
    coords = array(coords).T
    #print array(x).T.shape, coords.shape
    dist = spatial.distance.cdist(array(x).T, coords)
    indices = dist.argmin(axis=0)
    tn = tev[indices]
    def integrand(tx):
        d = interpolate.splev(tx, t, der=1)
        return sqrt(d[0]**2 + d[1]**2)
    s = []
    for i in tn:
        s.append(integrate.quad(integrand, 0, i)[0])
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

if __name__ == '__main__':
    get_length(pressure, pressure)
