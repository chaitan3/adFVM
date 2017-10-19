# Copyright Qiqi Wang (qiqi@mit.edu) 2013

import sys
from pylab import *
from numpy import *

sys.path.append('..')
from lssode import *

set_fd_step(1E-30j)
sigma, beta = 10, 8./3

def lorenz(u, rho):
    shp = u.shape
    x, y, z = u.reshape([-1, 3]).T
    dxdt, dydt, dzdt = sigma*(y-x), x*(rho-z)-y, x*y - beta*z
    return transpose([dxdt, dydt, dzdt]).reshape(shp)

def dfdu_lorenz(u, rho, per=0.0):
    shp = u.shape
    x, y, z = u.reshape([-1, 3]).T
    one = np.ones_like(x)
    f1 = np.stack((-sigma*one - per, sigma*one, 0*one), axis=1)
    f2 = np.stack((rho-z, -one - per, -x), axis=1)
    f3 = np.stack((y, x, -beta*one - per), axis=1)
    dfdu = np.stack((f1, f2, f3), axis=1)
    return dfdu

def obj(u, r):
    return u[:,2]**2

rhos = linspace(28, 34, 10)
dt = 0.01

tangent = []
adjoint = []
for rho in rhos:
    print(rho)
    for i in range(11):
        T = 20
        if i == 10: T = 200

        t = 30 + dt * arange(int(T / dt))

        x0 = random.rand(3)
        tan = Tangent(lorenz, x0, rho, t)
        v1 = tan.v
        J = tan.evaluate(obj)
        dJds1 = tan.dJds(obj)

        pers = [10**x for x in range(-10, 4)]
        errs = []
        for per in pers:
            dfdu = lambda x, y: dfdu_lorenz(x, y, per=per)
            tan = Tangent(lorenz, x0, rho, t, dfdu=dfdu)
            v2 = tan.v
            J = tan.evaluate(obj)
            dJds2 = tan.dJds(obj)
            #plt.plot(v1-v2)
            #plt.show()
            err = np.abs(dJds2-dJds1)/dJds1
            errs.append(err)
            print per, err
        r = errs[0]/pers[0]
        errs_line = [r*x for x in pers]
        plt.loglog(pers, errs, 'o', label='error in sensitivity')
        plt.loglog(pers, errs_line, label='error bound')
        plt.legend(loc='upper left')
        plt.xlabel('perturbation size')
        print([x-y for x,y in zip(errs_line, errs)])
        plt.savefig('tangent_error.png')
        exit(1)

        #J = tan.evaluate(obj)
        #dJds = tan.dJds(obj)
        #tangent.append(dJds)

        #x0 = random.rand(3)
        #adj = Adjoint(lorenz, x0, rho, t, obj)

        #J = adj.evaluate()
        #dJds = adj.dJds()
        #adjoint.append(dJds)

tangent = array(tangent).reshape([rhos.size, -1])
adjoint = array(adjoint).reshape([rhos.size, -1])

figure(figsize=(5,4))
plot(rhos, tangent[:,:-1], 'xr')
plot(rhos, tangent[:,-1], '-r')
plot(rhos, adjoint[:,:-1], '+b')
plot(rhos, adjoint[:,-1], '--b')
ylim([0, 1.5])
xlabel(r'$\rho$')
ylabel(r'$d\overline{J}/d\rho$')

show()

