from scipy import interpolate, integrate, spatial
from numpy import *

def get_length(pat, coords):
    t, u = interpolate.splprep(pat, s=0)
    tev = linspace(0, 1, 1000)
    x = interpolate.splev(tev, t)
    d = interpolate.splev(tev, t, der=1)
    s = (1+(d[1]/d[0])**2)**0.5*abs(d[0])
    dist = spatial.distance.cdist(array(x).T, coords)
    indices = dist.argmin(axis=0)
    l = integrate.cumtrapz(s, tev, initial=0)
    return l[indices]

def import_nozzle_profile():
    chord = 0.067647
    pitch = 0.0575

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
    return pressure, suction, chord, pitch

def import_blade_profile():
    delta = -0.04
    #delta = -0.04 - 0.0288
    chord = 0.03973
    pitch = 0.0575
    f = open('/home/talnikar/adFVM/cases/stator-rotor/mesh/blade_coords.txt', 'r')
    v1 = []
    v2 = []
    n = 36
    for index, l in enumerate(f.readlines()):
        if index == 0:
            continue
        t = l.split()
        v1.append(t[0])
        v2.append(t[1])
    f.close()
    v1 = array(v1, dtype=float)/1000
    v2 = array(v2, dtype=float)/1000  + delta
    pressure = [v1[:n][::-1], v2[:n][::-1]]
    suction = [v1[n:], v2[n:]]
    return pressure, suction, chord, pitch
