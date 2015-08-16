import numpy as np
import matplotlib.pyplot as plt

f = open('data.txt')
lines = f.readlines()
x = []
y = []
for line in lines:
    terms = line.split(' ')
    x.append(float(terms[-2]))
    y.append(float(terms[-1]))
plt.plot(x, y)
plt.xlabel('Time (s)')
plt.ylabel('Adjoint fields L2 norm')
plt.show()


