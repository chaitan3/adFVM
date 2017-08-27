import numpy as np
import matplotlib.pyplot as plt

def M(A, B):
    return np.dot(A, B)

def func((rho, sigma, beta), state, dt):
    x, y, z = state[:,0], state[:,1], state[:,2]
    dxdt, dydt, dzdt = sigma * (y - x), \
                       x * (rho - z) - y, \
                       x * y - beta * z
    dfdt = np.array([dxdt, dydt, dzdt]).T
    return state + dfdt*dt

rho, sigma, beta = 28., 10., 8./3
p = (rho, sigma, beta)
dt, T = 0.01, 100
Xd = [np.array([1, 0, 1]) + np.random.randn(3)]
Li, L = 1000, 10000
for i in range(0, Li):
    Xd[-1] = func(p, np.array([Xd[-1]]), dt)[0]
for i in range(0, L):
    Xd.append(func(p, np.array([Xd[-1]]), dt)[0])
Xd = np.array(Xd)

H = np.array([[0., 0., 1.]])
Y = Xd[:,2] #+ np.random.randn(n)*0.01

N = 2000
X =  [np.array([[0, 0, 30]]) + 5*np.random.randn(N,3)]

R = 1e-2
ps = p
ps = (p[0] + 0.001, p[1], p[2])
for i in range(0, L):
    Xp = X[-1]
    Xf = func(ps, Xp, dt)
    d = Y[i]
    D = d + np.sqrt(R)*np.random.randn(N)

    C = np.cov(Xf.T) + np.eye(3)*1e-12
    K = M(M(C, H.T), np.linalg.inv(M(M(H, C), H.T) + R))
    Xn = Xf + np.dot(K, D-np.dot(H,Xp.T)).T
    X.append(Xn)

X = np.array(X)
Xm = X.mean(axis=1)
c = ['r', 'g', 'b']
print(np.abs(Xd[L/2:]-Xm[L/2:])).max()
for i in range(0, 3):
    plt.plot(Xd[:L, i], color=c[i])
    plt.plot(Xm[:L, i], '--', color=c[i])
plt.show()
