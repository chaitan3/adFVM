import numpy as np
import matplotlib.pyplot as plt

folder = '2d_vane/'

M2 = np.loadtxt(folder + 'M2')
energy = np.loadtxt(folder + 'energy')
convergence = open(folder + 'convergence').readlines()
convergence = convergence[::5]
convergence = [np.array(x.split(' ')).astype(float) for x in convergence]
convergence = np.array([x[0] for x in convergence])

plt.plot(M2/M2.max(), label='M2')
plt.plot(energy/energy.max(), label='energy')
plt.plot(convergence/convergence.max(), label='convergence')
plt.legend()
plt.show()

