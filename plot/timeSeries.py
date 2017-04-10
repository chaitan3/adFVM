import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

plt.rcParams.update({'legend.fontsize': 18,
                     'xtick.labelsize': 14,
                     'ytick.labelsize': 14,
                     'axes.labelsize': 16
                    })

n = len(sys.argv) - 1
cmap = plt.get_cmap('nipy_spectral')
colors = [cmap(i) for i in np.linspace(0, 1, n)]
c = 0
for f in sys.argv[1:]:
    y = np.loadtxt(f)
    #y = y[-200000:]
    #index = np.abs(y) > 3e-10
    #y[index] = np.sign(y[index])*3e-10
    #print y.sum()

    x = np.arange(0, len(y))
    s = y
    s = np.cumsum(y)/np.arange(1,len(y)+1)
    plt.xlabel('time (T)')
    plt.semilogy(x, y, label='instantaneous objective')
    #plt.plot(x, y, label='instantaneous objective')
    #plt.plot(x, s, label='cumulative averaged objective')
    #token = os.path.basename(f).split('_')[0]
    #xy = (x, s)
    #plt.plot(xy[0], xy[1], c=colors[c], label=f)
    #plt.annotate(str(c), xy=(xy[0][-1], xy[1][-1]))
    #plt.legend(loc='lower right')
    c += 1
plt.xlabel('iteration')
plt.ylabel('objective')
#plt.xlabel('Time unit')
#plt.ylabel('Drag over cylinder (N)')
#plt.savefig('test.png')
plt.show()


