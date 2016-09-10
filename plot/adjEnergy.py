import matplotlib as mpl
mpl.use( "agg" )

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'legend.fontsize': 14,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'axes.labelsize': 18
                    })

import os

loc = '2d_vane/'
names = filter(lambda name: name.startswith('adj') and name.endswith('txt'), os.listdir(loc))
print names
labels = [float(name.split('_')[1].split('.txt')[0]) for name in names]
print labels
labels, names = zip(*sorted(zip(labels, names)))
#labels = labels[:-2]
#names = names[:-2]
cm = plt.get_cmap('gist_rainbow')
n = len(names)
cmc =[cm(1.*i/n) for i in range(n)]
lines = ["-","--","-.",":", '']
#plt.gca().set_color_cycle(cmc)
cmcd = {}
for i,j in enumerate(labels):
    cmcd[j] = cmc[i]
for name, label, l in zip(names, labels, lines):
#for name,label in zip(['data.txt', 'data2.txt', 'data3.txt'], ['2D adjoint', '3D adjoint', '3D long term adjoint']):
    f = open(loc + name)
    #f = open(name)
    #lines = [line for line in f.readlines() if line.startswith('L2 norm adjoint')]
    #x = []
    #y = []
    #for line in lines:
    #    terms = line.split(' ')
    #    x.append(float(terms[-2])-3)
    #    y.append(float(terms[-1]))
    #x = x[-1] - np.array(x)
    #x /= 0.001
    y = np.loadtxt(f)
    x = np.linspace(0,1,len(y))
    plt.semilogy(x, y, 'k'+l, label=label)#, color=np.random.rand(3,1))

    #s = len(y)/5
    #visc = float(label)
    #ilambda = -1./np.polyfit(x[s:]*20000, np.log(y[s:]), 1)[0]
    #dt = 2.22e-8
    #ilambda0 = 279.
    #print visc, ilambda, np.sqrt(visc*ilambda0*dt)
plt.xlabel('Time (T)')
plt.ylabel('Adjoint energy')
plt.legend(loc='upper right')
plt.savefig(loc + 'vane_energy.png')
plt.clf()

import cPickle
print cmcd
with open('colors.pkl', 'w') as f:
    cPickle.dump(cmcd, f)


