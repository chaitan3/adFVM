from re import findall
from numpy import *
from matplotlib import pyplot as plt

string = 'Time since beginning'
#string = 'Time for iteration'

def parse_log(n):
    f = open(n)
    times = array(findall(r'' + string + ': [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', f.read()))[:, 0]
    times = times.astype(float)
    return times

def get_average(times):
    itert = times[1:] - times[:-1]
    #plt.hist(itert, 50)
    #plt.show()
    return average(itert)

times = parse_log('jobAD.sh.o2381620')
print get_average(times)

exit(1)

x = []
nt = []
for i in range (0, 7):
    n = 24*2**i
    x.append(n)
    avg = 1
    nt.append(avg)
    print n, avg

plt.plot(x,nt)
plt.xlabel('nprocs')
plt.ylabel('seconds/iteration')
plt.show()
plt.loglog(x, nt, 'bo-')
plt.xlabel('nprocs')
plt.ylabel('seconds/iteration')
plt.show()
