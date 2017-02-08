import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def auto_corr(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x) 
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

def auto_corr(x) :
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal.
    """
    xp = x-np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:x.size/2]/np.sum(xp**2)

plt.rcParams.update({'legend.fontsize': 18,
                     'xtick.labelsize': 14,
                     'ytick.labelsize': 14,
                     'axes.labelsize': 16
                    })
import ar
e = [[], []]
cmap = plt.get_cmap('nipy_spectral')
colors = [cmap(i) for i in np.linspace(0, 1, len(sys.argv)-1)]
cs = []
c = 1

start = 50000
for f in sys.argv[1:]:
    y = np.loadtxt(f)[start:]
    n = len(y)
    x = np.arange(0, n)
    m = y.mean()
    y = y - m
    e[0].append(m)
    e[1].append(ar.arsel(y).mu_sigma[0])
    print e[0][-1], e[1][-1]
    token = os.path.basename(f).split('_')[0]
    cs.append(c)
    #plt.plot(x[:n/2], auto_corr(y)[:n/2])
    #plt.xlabel('Time unit')
    #plt.ylabel('Autocorrelation')
    #plt.savefig(f.replace('timeSeries.txt', 'autocorr.png'))
    #plt.clf()
    #plt.show()
    c += 1

x = np.arange(0, len(cs)+2,1)
for i, c in enumerate(cs):
    c = (33 + 0.01*(c - 3))/360.
    plt.errorbar(c, e[0][i], e[1][i], ecolor='b', fmt='o')
    #plt.xticks(x)
    plt.xlabel('Mach number')
    plt.ylabel('Drag over cylinder')
plt.show()
