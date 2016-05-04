import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.labelsize':'large'})
plt.rcParams.update({'xtick.labelsize':'large'})
plt.rcParams.update({'ytick.labelsize':'large'})
plt.rcParams.update({'legend.fontsize':'large'})

# cylinder
#viscosity = np.array([5e-4,1e-4,1e-3,6e-4,4e-4,5e-3,1e-2, 3e-3, 7e-3])
#sens = np.array([-23.545240152,-757.17808868,99.2337272681,38.1457597971,-119.314335486,-37.6285100821,-31.1810471935,-24.6231952264,-35.0582046915])
#perturb = -37.5296614557
# 2d vane
#viscosity = np.array([0,1e-5,1e-4,3.5e-4,3.8e-4,5e-4,6e-4,1e-3,1e-2])
#sens = np.array([-63958,-33637,39.72,0.0040,-0.00397,-0.00789,-0.00777,-0.00717,-0.0041])
#perturb = -0.01097
# read from objective.txt
viscosity = []
sens = []
perturb = 0.

f = open('vane/objective.txt')
lines =  f.readlines()
for line in lines:
    words = line.split(' ')
    if words[0] == 'perturb':
        perturb = float(words[1])
    elif words[0] == 'adjoint':
        sens.append(float(words[1]))
    else:
        if words[0] != 'orig':
            viscosity.append(float(words[0]))
index = np.argsort(viscosity)
viscosity = np.array(viscosity)[index]
sens = np.array(sens)[index]
Ah = {}
for i in range(0, len(sens)):
    Ah[viscosity[i]] = sens[i]


def polynomial(h, t, Ah, n=None, k=None):
    if n is None:
        n = len(Ah)-1
    if k is None:
        k = np.arange(1, n+1)
    def Ar(x, i):
        if i == 0:
            return Ah[x]
        return (t**k[i-1]*Ar(x/t, i-1)-Ar(x, i-1))/(t**k[i-1]-1)
    Ac = []
    for i in range(0, n):
        A = Ar(h, i+1)
        Ac.append(A)
    return Ac

from scipy.optimize import brentq
def unknown(h, t, s, Ah):
    def f(x):
        def A(h, t):
            return (t**x*Ah[h/t]-Ah[h])/(t**x-1)
        v = A(h, t)-A(h, s)
        return v
    #x = np.linspace(-10, 10, 1000)
    #y = []
    #for i in x:
    #    y.append(f(i))
    #plt.plot(x, y)
    #plt.show()
    return brentq(f, -0.1, 0.1)

h = 3.84
t = 2.
print polynomial(h, t, Ah, 5)
h = 3.84
t = 2.**1
s = 2.**2
print unknown(h, t, s, Ah)

viscosity = viscosity[3:]
sens = sens[3:]
error = 100*(sens-perturb)/abs(perturb)


from numpy import *
from matplotlib.pyplot import *
#coeff = polyfit(viscosity, error, 1)
#poly = poly1d(coeff)
#plt.semilogx(viscosity, poly(viscosity))
plt.xlabel('viscosity scaling factor')

plt.semilogx(viscosity, perturb*np.ones_like(viscosity))
plt.semilogx(viscosity, sens, 'b.',markersize=20)
plt.ylabel('sensitivity')
#plt.semilogx(viscosity, np.zeros_like(viscosity))
#plt.semilogx(viscosity, error, 'b.',markersize=20)
#plt.ylabel('percent error in sensitivity')

plt.savefig('vane/error.png')
