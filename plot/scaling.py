from re import findall
from numpy import *
from matplotlib import pyplot as plt
plt.rcParams.update({'axes.labelsize':'large'})
plt.rcParams.update({'xtick.labelsize':'large'})
plt.rcParams.update({'ytick.labelsize':'large'})
plt.rcParams.update({'legend.fontsize':'large'})

string = {'primal':'Time for iteration',
        'adjoint':'Time for adjoint iteration'}

def parse_log(n, sim):
    f = open(n)
    times = array(findall(r'' + string[sim] + ': [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', f.read()))[:, 0]
    times = times.astype(float)
    return times

def get_average(times):
    #times = times[1:] - times[:-1]
    #plt.hist(times, 50)
    #plt.show()
    return average(times)

#import sys
#times = parse_log(sys.argv[1])
#print get_average(times)

x = [2, 16, 128, 1024, 8192]
p = []
a = []
for n in x:
    f = 'primal/' + str(n) + '.out'
    p.append(get_average(parse_log(f, 'primal')))
    f = 'adjoint/' + str(n) + '.out'
    a.append(get_average(parse_log(f, 'adjoint')))

plt.semilogx(x, p, 'bo-', label='Primal solver')
plt.semilogx(x, a, 'ro-', label='Adjoint solver')
plt.xlabel('Number of processors')
plt.ylabel('Time per iteration (s)')
plt.legend(loc='upper left')
plt.show()
