import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def lorenz(rho, sigma, beta, dt, T, init_state=None):
    if init_state is None:
        init_state = np.array([[1, 0, 1]]) + np.random.randn(1,3)

    def ddt(u, t):
        x, y, z = u
        dxdt, dydt, dzdt = sigma * (y - x), \
                           x * (rho - z) - y, \
                           x * y - beta * z
        return np.array([dxdt, dydt, dzdt])

    t = (np.arange(0, T / dt) + 1) * dt
    state = odeint(ddt, init_state[-1], t)
    return np.vstack([init_state, state])

if __name__ == '__main__':
    y = lorenz(28., 10., 8./3, 0.01, 100)
    x = np.linspace(0, 100, 100/0.01+1)
    x, y = x[1000:], y[1000:]
    for ys in y.T:
        plt.plot(x, ys)
    plt.show()
