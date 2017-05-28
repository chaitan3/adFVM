from numpy import *
import numpy as np
from scipy.integrate import odeint
from lssode import *
orig_bounds = np.array([[25.,35.], [2.5, 3.5]])

def lorenz(rho, sigma, beta, dt, T):
    init_state = array([[1, 0, 1]]) + random.rand(1,3)

    def ddt(u, t):
        x, y, z = u
        dxdt, dydt, dzdt = sigma * (y - x), \
                           x * (rho - z) - y, \
                           x * y - beta * z
        return array([dxdt, dydt, dzdt])

    t = (arange(0, T / dt) + 1) * dt
    state = odeint(ddt, init_state[-1], t)
    return vstack([init_state, state])

def lss_lorenz(rho, sigma, beta, dt, T, init_state=None):
    if init_state is None:
        init_state = array([1, 0, 1]) + random.rand(3)
    else:
        init_state = array([1, 0, 1]) + init_state
    t = 30 + dt*np.arange(int(T/dt))

    def ddt(u, (rhos, betas)):
        shp = u.shape
        x, y, z = u.reshape([-1, 3]).T
        dxdt, dydt, dzdt = sigma*(y-x), x*(rhos-z)-y, x*y - betas*z
        return np.transpose([dxdt, dydt, dzdt]).reshape(shp)

    def objective(u, (rhos, betas)):
        #N0 = int(round(burnin / dt))
        #N = int(round(.2 / dt))
        #samples = data[N0::N]
        zr = 50
        xr = 12
        obj = (u[:,2]-zr)**2 + (u[:,0]**2 - xr**2)**2/2
        import matplotlib.pyplot as plt
        plt.plot(u[:,2])
        plt.plot(u[:,0])
        plt.show()
        return (obj-5e3)/5e3#*(1+np.sin(2*np.pi*(rhos-orig_bounds[0,0])/(orig_bounds[0,1]-orig_bounds[0,0]))*\
                            #   np.sin(2*np.pi*(betas-orig_bounds[1,0])/(orig_bounds[1,1]-orig_bounds[1,0])))

    t = (arange(0, T / dt) + 1) * dt
    adj = Adjoint(ddt, init_state, [rho, beta], t, objective)

    return adj.evaluate(), adj.dJds()
