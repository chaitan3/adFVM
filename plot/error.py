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

viscosity = viscosity[3:]
sens = sens[3:]

print viscosity
print sens
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
