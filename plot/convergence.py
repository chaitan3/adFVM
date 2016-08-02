import numpy as np
import matplotlib.pyplot as plt

folder = 'vane2/'

M2 = np.loadtxt(folder + 'M2')
energy = np.loadtxt(folder + 'energy')
convergence = open(folder + 'convergence').readlines()
convergence = convergence[::5]
convergence = [np.array(x.split(' ')).astype(float) for x in convergence]
convergence = np.array([x[-1]/x[0] for x in convergence])
print convergence.max(), convergence.min()

plt.semilogy(M2/M2.max(), label='M2')
plt.semilogy(energy/energy.max(), label='energy')
plt.semilogy(convergence, label='convergence')
#plt.semilogy(convergence/convergence.max(), label='convergence')
plt.legend(loc=0)
plt.show()

