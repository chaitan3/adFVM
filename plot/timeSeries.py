import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'legend.fontsize': 18,
                     'xtick.labelsize': 14,
                     'ytick.labelsize': 14,
                     'axes.labelsize': 16
                    })
y = np.loadtxt('timeSeries.txt')
y *= 4
x = np.arange(0, len(y))*1./30000
plt.plot(x, y, label='instantaneous stagnation pressure')
plt.xlabel('time (T)')
plt.ylabel('stagnation pressure loss coefficient')
s = np.cumsum(y)/np.arange(1,len(y)+1)
plt.plot(x, s, label='cumulative averaged ' + 'stagnation pressure')
#plt.show()
plt.legend(loc='lower right')
plt.show()


