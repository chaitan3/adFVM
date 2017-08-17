from re import findall
from numpy import *
from matplotlib import pyplot as plt
plt.rcParams.update({'legend.fontsize':'18'})
plt.rcParams.update({'axes.labelsize':'16'})
plt.rcParams.update({'xtick.labelsize':'16'})
plt.rcParams.update({'ytick.labelsize':'14'})
plt.rcParams.update({'figure.figsize':(10, 6)})

string = {'primal':'Time for iteration',
        'adjoint':'Time for adjoint iteration'}

def parse_log(n, sim):
    f = open(n)
    content = f.read()
    times = array(findall(r'' + string[sim] + ': [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', content))
    times = times[:, 0]
    times = times.astype(float)
    return times

def get_average(times):
    #times = times[1:] - times[:-1]
    #plt.hist(times, 50)
    #plt.show()
    #return average(times)
    return min(times)

case = 'vane/scaling/3d_10/'
adjoint = False
#case = 'vane/scaling/laminar/'
#adjoint = True
x = [1, 2, 4, 8, 16]
p = []
a = []
for n in x:
    f = case + 'par-{}/problem.py_output.log'.format(n)
    p.append(get_average(parse_log(f, 'primal')))
    if adjoint:
        f = case + 'par-{}/adjoint.py_output.log'.format(n)
        a.append(get_average(parse_log(f, 'adjoint')))
pi = [p[0]]
for n in x[1:]:
    pi.append(p[0]/(n/(x[0])))
plt.loglog(x, p, 'bo-', label='Primal solver')
plt.loglog(x, pi, 'b--', label='Primal ideal')
if adjoint:
    ai = [a[0]]
    for n in x[1:]:
        ai.append(a[0]/(n/(x[0])))
    plt.loglog(x, a, 'ro-', label='Adjoint solver')
    plt.loglog(x, ai, 'r--', label='Adjoint ideal')
plt.xlabel('Number of processors')
plt.ylabel('Time per iteration (s)')
plt.xlim([0.5, 64])
plt.legend(loc='upper right')
plt.show()
