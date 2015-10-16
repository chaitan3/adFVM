import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.labelsize':'large'})
plt.rcParams.update({'xtick.labelsize':'large'})
plt.rcParams.update({'ytick.labelsize':'large'})
plt.rcParams.update({'legend.fontsize':'large'})

for name,label in zip(['adj_5e-4.txt'], ['adjoint']):
#for name,label in zip(['data.txt', 'data2.txt', 'data3.txt'], ['2D adjoint', '3D adjoint', '3D long term adjoint']):
    f = open(name)
    lines = [line for line in f.readlines() if line.startswith('L2 norm adjoint')]
    x = []
    y = []
    for line in lines:
        terms = line.split(' ')
        x.append(float(terms[-2])-3)
        y.append(float(terms[-1]))
    x = x[-1] - np.array(x)
    plt.semilogy(x, y, label=label)
plt.xlabel('Time (s)')
plt.ylabel('Adjoint fields L2 norm')
plt.legend(loc=0)
plt.show()


