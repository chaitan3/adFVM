import matplotlib.pyplot as plt
import numpy as np
import pickle

n = 100
n = 1000
#a = np.loadtxt('output_grad.log')
#c = a[:,0]
#cs = a[:,1]
c = 0.
cs = 0.
b = 0.
bs = 0.
t = []
for i in range(0, n):
    print i
    a = np.loadtxt('output_{}.log'.format(i), delimiter=',')
    b += a[:,-2]
    bs += a[:,-1]
    a = np.loadtxt('output_grad_{}.log'.format(i), delimiter=',')
    c += a[:,-2]
    cs += a[:,-1]
    t.append(a[-1,-2])
b /= n
bs /= n
c /= n
cs /= n
#print np.std(t)/np.mean(t)
#exit(0)

plt.errorbar(np.arange(0, len(b)), b, yerr=bs**0.5, fmt='o-', label='no gradient')
plt.errorbar(np.arange(0, len(c)), c, yerr=cs**0.5, fmt='o-', label='with gradient')
plt.legend()
plt.xlabel('Optimization step')
plt.ylabel('Objective surrogate minimum')
plt.show()
