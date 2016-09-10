import matplotlib.pyplot as plt
import numpy as np
import sys

plt.rcParams.update({'legend.fontsize': 18,
                     'xtick.labelsize': 14,
                     'ytick.labelsize': 14,
                     'axes.labelsize': 16
                    })

try:
    n, start = [int(x) for x in sys.argv[2:4]]
except:
    n, start = 1, 0
print 'norm interval', n
print 'norm start', start
y = np.loadtxt(sys.argv[1])
y = y[start:]
x = np.arange(0, len(y))
#plt.xlabel('time (T)')
plt.xlabel('iteration')
plt.ylabel('objective')
plt.semilogy(x, y, label='instantaneous objective')
print 'time steps for e times growth:', np.abs(n/np.polyfit(x, np.log(y), 1)[0])
plt.show()


