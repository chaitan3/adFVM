import numpy as np
a=np.genfromtxt('output', delimiter='  ')
b=np.genfromtxt('rho.csv', delimiter=',')

from matplotlib import pyplot as plt
plt.rcParams.update({'axes.labelsize':'large'})
plt.rcParams.update({'xtick.labelsize':'large'})
plt.rcParams.update({'ytick.labelsize':'large'})
plt.rcParams.update({'legend.fontsize':'large'})

plt.plot(a[:, 1], a[:, 2], label='analytical')
plt.plot(b[:, -3], b[:, 5], label='simulation')
plt.xlabel('x (m)')
plt.ylabel('density')
plt.legend()
plt.ylim([0.1, 1.1])
plt.xlim([-5,5])

plt.show()

