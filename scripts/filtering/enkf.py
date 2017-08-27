import numpy as np
import matplotlib.pyplot as plt

def M(A, B):
    return np.dot(A, B)


def enkf(dynamic, prior, (N, L), (H, Y, R)):
    X = [prior(N)]
    for i in range(0, L):
        Xp = X[-1]
        Xf = dynamic(Xp)
        d = Y[i]
        D = d + np.sqrt(R)*np.random.randn(N)

        C = np.cov(Xf.T) + np.eye(3)*1e-12
        K = M(M(C, H.T), np.linalg.inv(M(M(H, C), H.T) + R))
        Xn = Xf + np.dot(K, D-np.dot(H,Xf.T)).T
        X.append(Xn)
    R = 0.1

    X = np.array(X)
    Xm = X.mean(axis=1)
    XC = [np.cov(Xs.T) for Xs in X] 
    return Xm, XC

def func((rho, sigma, beta), state, dt):
    x, y, z = state[:,0], state[:,1], state[:,2]
    dxdt, dydt, dzdt = sigma * (y - x), \
                       x * (rho - z) - y, \
                       x * y - beta * z
    dfdt = np.array([dxdt, dydt, dzdt]).T
    return state + dfdt*dt

def prior(N):
    return np.array([[1, 0, 30]]) + 5*np.random.randn(N,3)

def reference(p, (Li, L, dt)):
    # reference trajectory
    rho, sigma, beta = p
    p = (rho, sigma, beta)
    Xd = [np.array([1, 0, 1]) + np.random.randn(3)]
    for i in range(0, Li):
        Xd[-1] = func(p, np.array([Xd[-1]]), dt)[0]
    for i in range(0, L):
        Xd.append(func(p, np.array([Xd[-1]]), dt)[0])
    Xd = np.array(Xd)
    # observations
    H = np.array([[0., 0., 1.]])
    return Xd, H

if __name__ == '__main__':
    Li, L, dt = 1000, 10000, 0.01
    p = (28., 10., 8./3)
    Xd, H = reference(p, (Li, L, dt))
    Y = np.dot(Xd, H[0])
    N = 200
    Xm, XC = enkf(lambda x: func(p, x, dt), prior, (N, L), (H, Y, 0.1))
    c = ['r', 'g', 'b']
    print(np.sqrt(np.linalg.norm((Xd[L/2:]-Xm[L/2:])**2/(L/2))))
    for i in range(0, 3):
        plt.plot(Xd[:L, i], color=c[i])
        plt.plot(Xm[:L, i], '--', color=c[i])
    plt.show()
    #for i in range(0, 9):
    #    plt.plot([np.abs(x.flatten()[i]) for x in XC])
    #plt.show()
