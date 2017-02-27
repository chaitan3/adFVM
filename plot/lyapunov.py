import numpy as np
import matplotlib.pyplot as plt
import sys
plt.rc('text', usetex=True)

fs = sys.argv[1:]
labels = [r'\lambda = 0.0', r'\lambda = 10^{-2}']
for f, l in zip(fs, labels):
    x = np.loadtxt(f)
    plt.plot(1+np.arange(0, len(x)), x, 'o', markersize=8, label=l)
    plt.plot(np.arange(0, len(x)), np.zeros_like(x))
plt.legend()
plt.xlabel('Exponent number')
plt.ylabel('Lyapunov exponent')
plt.show()
