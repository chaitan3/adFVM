import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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

f = open('naca0012/objective.txt')
lines =  f.readlines()
#for line in lines:
#    words = line.split(' ')
#    if words[0] == 'perturb':
#        perturb = float(words[1])
#    elif words[0] == 'adjoint':
#        sens.append(float(words[1]))
#    else:
#        if words[0] != 'orig':
#            viscosity.append(float(words[0]))
for line in lines:
    words = line.split(' ')
    if words[0] == 'perturb':
        perturb = float(words[1])
    elif words[0] == 'adjoint':
        viscosity.append(float(words[2]))
        sens.append(float(words[3]))

#index = np.argsort(viscosity)[2:]
index = np.argsort(viscosity)
viscosity = np.array(viscosity)[index]
sens = np.array(sens)[index]

#viscosity = viscosity[-8:]
#sens = sens[-8:]
#error = 100*(sens-perturb)/abs(perturb)

from numpy import *
from matplotlib.pyplot import *
#coeff = polyfit(viscosity, error, 1)
#poly = poly1d(coeff)
#plt.semilogx(viscosity, poly(viscosity))

#import cPickle
#cmcd = cPickle.load(open('colors.pkl'))
#colors = [cmcd.get(i,(0.,0.,0.,1.)) for i in viscosity]

sens = np.abs(sens)
from scipy.interpolate import UnivariateSpline
z = UnivariateSpline(viscosity, sens, k=4, s=len(sens))
plt.xlabel('viscosity scaling factor')
plt.ylabel('sensitivity')
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
#plt.semilogx(viscosity, perturb*np.ones_like(viscosity), 'k',label='finite difference')
#plt.semilogx(viscosity, perturb*np.ones_like(viscosity), 'k.', markerfacecolor='white', markersize=20)
#for i in range(0, len(viscosity)):
#    plt.loglog(viscosity[i], sens[i], '.', color=colors[i],markersize=20)
plt.loglog(viscosity, sens, 'k.',markersize=15, label='adjoint sensitivity')
plt.loglog(viscosity, sens, 'k')
x = np.linspace(viscosity[0], viscosity[-1], 100)
print 'x', x
print 'f', z(x)
#plt.plot(x, z(x), 'b-',label='fit')
#xr = [3e-4, 0.011]
#x = np.log(xr)
#x = np.linspace(x[0], x[1], 100)
#y = f(x)
#print x
#f = np.poly1d(z)
#plt.semilogx(np.exp(x), y, 'k--')
#plt.xlim(xr)
#plt.axis('tight')
#sens = np.abs(sens-perturb)/np.abs(perturb)
#plt.ylabel('relative error in sensitivity')
#plt.loglog(viscosity, sens, '.', markersize=20, label='samples')
print viscosity
print sens
#plt.semilogx(viscosity, np.zeros_like(viscosity))
#plt.semilogx(viscosity, error, 'b.',markersize=20)
#plt.ylabel('percent error in sensitivity')

plt.legend()
plt.savefig('naca0012/adj_sens.png')
#plt.show()
