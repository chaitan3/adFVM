import numpy as np
import matplotlib.pyplot as plt

def corr2d(Z):
    m, n = Z.shape
    res = np.zeros_like(Z)
    for i in range(0, m):
        x = m-i
        for j in range(0, n):
            y = n-j
            res[i, j] = (Z[:x,:y]*Z[i:,j:]).sum()/(x*y)
    return res

from scipy.fftpack import fft2, fftshift
def fft(Z, img):
    g = Z.shape[0]
    Z = corr2d(Z)
    f = fftshift(fft2(Z))
    #f = f[:g/2,:g/2]
    f = np.absolute(f)**2
    x = np.linspace(-1, 1, g)
    X, Y = np.meshgrid(x,x)
    #plt.imshow(np.log(f))
    plt.contourf(X, Y, np.log(f), 50)
    plt.xlabel('Lx')
    plt.ylabel('Ly')
    plt.colorbar()
    plt.savefig(img)
