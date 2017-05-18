import numpy as np
import scipy as sp
import sys
import os
import pyflux as pf

#import matplotlib.pyplot as plt
#plt.rcParams.update({'legend.fontsize': 18,
#                     'xtick.labelsize': 14,
#                     'ytick.labelsize': 14,
#                     'axes.labelsize': 16
#                    })

n = len(sys.argv) - 1
#cmap = plt.get_cmap('nipy_spectral')
#colors = [cmap(i) for i in np.linspace(0, 1, n)]
c = 0
s = []
visc = lambda x: float(os.path.basename(x)[:-4])
fs = sorted(sys.argv[1:], key=visc)
visc = [visc(x) for x in fs]

submean = True
ms = []

for f in fs:
    print f
    y = np.loadtxt(f)
    #reverse
    #differencing
    #y = y[1:]-y[:-1]
    y = y[-200000:]
    #index = np.abs(y) > 3e-10
    #y[index] = np.sign(y[index])*3e-10
    #print y.sum()
    #y = np.abs(y*200000)

    x = np.arange(0, len(y))
    ys = y[::100][200:]
    #ys = ys.reshape((len(ys)/10, 10)).sum(axis=1)
    ms.append(ys.mean())
    if submean:
        ys -= ms[-1]
    s.append(ys)

    #import statsmodels.tsa.arima_model
    #m = statsmodels.tsa.arima_model.ARMA(ys, (4, 2))
    ##m = statsmodels.tsa.ar_model.AR(ys)
    #res = m.fit(solver='nm')
    #print res.params
    #yp = res.predict(end=300)

    #plt.semilogy(x, y, label=os.path.basename(f))
    #plt.plot(x, y, label=os.path.basename(f))
    #plt.plot(x, y, label='instantaneous objective')
    #plt.plot(x, s, label='cumulative averaged objective')
    #token = os.path.basename(f).split('_')[0]
    #xy = (x, s)
    #plt.plot(xy[0], xy[1], c=colors[c], label=f)
    #plt.annotate(str(c), xy=(xy[0][-1], xy[1][-1]))
    c += 1

p = 3
#p = 6
n = len(s) + 1
Np = (2*n-1)*p + n
s = np.array(s)
# HACK
#s = s[:,:100]
N = len(s[0])

t = np.zeros(Np)
import ar
for i in range(0, n-1):
    A = ar.arsel(s[i], 0, 1, "CIC", p, p)
    c = -np.array(A.AR[0][1:])
    t[i:n*p:n] = c
    t[n-1:n*p:n] = c
    t[(2*n-1)*p + i] = A.sigma2eps[0]

    from scipy.signal import lfilter, lfiltic
    zp = lfiltic([1], A.AR[0], s[i][:p][::-1])
    yp = A.mu[0] + lfilter([1], A.AR[0], np.sqrt(A.sigma2eps[0])*np.random.randn(3000), zi=zp)[0]
    #yp = -np.convolve(s[i], A.AR[0][1:], 'valid')
    #plt.plot(np.arange(0, len(yp)), yp, label='model')
    #plt.plot(np.arange(0, len(s[i][p:])), s[i][p:], label='data')
    #plt.legend()
    #plt.show()
    #print A.AR[0], A.sigma2eps[0]
    #plt.plot(m.AR[0], 'o', label=os.path.basename(f))
    #print m.sigma2eps[0]
#plt.savefig('series_sim.pdf')
#plt.clf()

t[n*p:(n+1)*p] = 1.
t[-1] = t[(2*n-1)*p:].max()

def kalman_filter(params, s, p):
    n2 = n*n

    a = params[:n*p].reshape(p, n)
    b = params[n*p:(2*n-1)*p].reshape(p, n-1)
    dt = params[(2*n-1)*p:]*t[(2*n-1)*p:]
    if not submean:
        d = dt[:n]
        mu = tf.reshape(dt[n:], (n,1))[:-1]
    else:
        d = dt
        mu = np.zeros((n-1,1))

    A = np.zeros((p, n, n))
    B = np.eye((p+1)*n*n)
    ii = np.arange(0, n)
    A[:,ii,ii] = a
    A[:,:n-1,-1] = b

    F = np.hstack([A[i] for i in range(0, p)])
    F = np.vstack((F, np.hstack((np.eye((p-1)*n), np.zeros(((p-1)*n, n))))))
    Q = np.zeros((p*n, p*n))
    Q[:n, :n] = np.diag(d)
    H = np.zeros(((n-1), n*p))
    ii = np.arange(0, n-1)
    H[ii, ii] = 1

    Y = np.array(s).T
    I = np.eye(n*p)
    M = lambda A, B: np.matmul(A, B)
    T = lambda A: np.transpose(A)
    Ets = []
    Ps = []
    Pts = []
    Ss = []
    Cs = []
    Ys = []
    L = 0
    print 'param:', params
    print 'constraint:', constraint(params)
    if (constraint(params)[1:] < 0).any():
        raise Exception("constraint failure")

    Ei = np.concatenate((Y[:p], np.zeros((p, 1))), axis=1)
    E = sum([np.matmul(A[i], np.reshape(Ei[p-i-1], (n,1))) for i in range(0, p)])
    E = np.reshape(np.concatenate((np.reshape(E, (1,n)), Ei[::-1][:-1]), axis=0), (n*p, 1))
    Et = E
    Pt = Q

    #Pr = Q[:]**0.5
    #for i in range(p, N):
    #    E = M(F, Et)
    #    P = M(T(Pr), Pr)
    #    S = M(H, M(P, T(H)))
    #    Ss.append(S)

    #    y = np.matrix(Y[i].reshape((n-1, 1)))
    #    yt = y-M(H, E)-mu
    #    Ys.append(yt)
    #    L = L + 0.5*(np.log(np.linalg.det(S)) + M(T(yt), np.linalg.solve(S, yt))[0,0])
    #    C = I - M(P, M(T(H), np.linalg.solve(S, H)))

    #    Ets.append(Et)
    #    Pts.append(Pt)
    #    Ps.append(P)
    #    Cs.append(C)
    #    
    #    Pt = M(C, P)
    #    T2 = np.concatenate((np.zeros((n-1,(n-1)+p*n), np.float64), 
    #              np.concatenate((M(Pr, T(H)), Pr), axis=1)), axis=0)
    #    T2R = np.linalg.qr(T2)[1]
    #    Pr = T2R[n-1:,n-1:]
    #    K1 = np.transpose(T2R[:n-1,n-1:])
    #    K2 = T2R[:n-1,:n-1]
    #    Et = E + M(K1, np.linalg.solve(T(K2), yt))
    #    T1 = np.concatenate((M(Pr, T(F)), Q**0.5), axis=0)
    #    Pr = np.linalg.qr(T1)[1][:n*p]

    ## forward recursions
    for i in range(p, N):

        E = M(F, Et)
        P = M(F, M(Pt, T(F))) + Q
        S = M(H, M(P, T(H)))
        Ss.append(S)

        y = Y[i].reshape((n-1, 1))
        yt = y-M(H, E)-mu
        Ys.append(yt)
        L = L + 0.5*(np.log(np.linalg.det(S)) + M(T(yt), np.linalg.solve(S, yt))[0,0])
        C = I - M(P, M(T(H), np.linalg.solve(S, H)))

        Ets.append(Et)
        Pts.append(Pt)
        Ps.append(P)
        Cs.append(C)

        Et = E + M(P, M(T(H), np.linalg.solve(S, yt)))
        Pt = M(C, P)
    print 'forward pass'

    Ebs = [Et]
    Pbs = [Pt + np.outer(Et, Et)]
    Pb2s = []
    Lh = 0
    lh = 0

    for i in range(N-p-1,-1,-1):
        P, Pt, Et = Ps[i], Pts[i], Ets[i]
        C, y, S = Cs[i], Ys[i], Ss[i]

        FC = M(F, C)

        if isinstance(Lh, int):
            Lh *= M(T(H), np.linalg.solve(S, H))
            lh *= M(T(H), np.linalg.solve(S, y))

        lh = M(T(FC), lh) - M(T(H), np.linalg.solve(S, y))
        Eb = Et - M(Pt, M(T(F), lh))
        Lh = M(T(FC), M(Lh, FC)) + M(T(H), np.linalg.solve(S, H))
        Pb = Pt - M(Pt, M(M(T(F), M(Lh, F)), Pt))
        Pb2 = M(I-M(P, Lh), M(F, Pt))

        Pb2s.append(Pb2 + np.outer(Ebs[-1], Eb))
        Ebs.append(Eb)
        Pbs.append(Pb + np.outer(Eb, Eb))

    A = sum(Pbs[1:])
    B = sum(Pb2s)
    C = sum(Pbs[:-1])

    An = M(B, np.linalg.inv(A))
    Q = (C-M(B, np.linalg.solve(A, T(B))))/(N-p)
    #Q = np.abs(Q)

    new_params = params.copy()
    A = np.array(np.hsplit(An[:n], p))
    new_params[:n*p] = np.concatenate([np.diag(x) for x in A])
    new_params[n*p:(2*n-1)*p] = np.concatenate([x[:-1,-1] for x in A])
    new_params[(2*n-1)*p:] = np.diag(Q[:n,:n])/t[(2*n-1)*p:]
    print 'likelihood:', L
    print
    return new_params

    #Js = []
    #Ebs = []
    #Pbs = []
    ## backward recursions
    #for i in range(N-p-1, -1, -1):
        #P, Pt, Et = Ps[i], Pts[i], Ets[i]
        #J = M(Pt, T(F))
        #Ebs.append(Eb)
        #Pbs.append(Pb)
        #Js.append(J)
        ## HACK
        ##P = P + np.diag(1e-39*np.ones_like(np.diag(P)))

        #Eb = Et + M(J, np.linalg.solve(P, (Eb-M(F, Et))))
        #Pb = Pt + M(J, M(np.linalg.solve(P, T(np.linalg.solve(P, T(Pb-P)))), T(J)))
    #V = []
    #Vs = []
    #for i in range(N-p-1, 0, -1):
        #P, Pt, Et = Ps[i], Pts[i], Ets[i]
        #Pp = Ps[i-1]
        #J = Js[N-p-1-i]
        #Jp = Js[N-p-2-i]
        ## HACK
        ##P = P + np.diag(1e-39*np.ones_like(np.diag(P)))
        ##Pp = Pp + np.diag(1e-39*np.ones_like(np.diag(P)))

        #Pb2 = M(T(np.linalg.solve(Pp, T(Pt))), T(Jp)) + M(J, M(np.linalg.solve(P, T(np.linalg.solve(Pp, T(Pb2-M(F, Pt))))), T(Jp)))
        #V.append(Pbs[N-p-1-i] + np.outer(Ebs[N-p-1-i], Ebs[N-p-1-i]))
        #Vs.append(Pb2 + np.outer(Ebs[N-p-1-i], Ebs[N-p-2-i]))
    #V.append(Pbs[-1] + np.outer(Ebs[-1], Ebs[-1]))

    #An = M(sum(Vs), np.linalg.inv(sum(V[1:])))
    #Q = (sum(V[:-1])-M(An, sum([T(x) for x in Vs])))/(N-p-1)
    #print An, Q

    #Ei = np.concatenate((Y[:p], np.zeros((p, 1))), axis=1)
    #E = sum([np.matmul(A[i], np.reshape(Ei[p-i-1], (n,1))) for i in range(0, p)])
    #E = np.reshape(np.concatenate((np.reshape(E, (1,n)), Ei[::-1][:-1]), axis=0), (n*p, 1))
    #P = Q[:]**0.5

    #for i in range(0, N):
    #    y = np.matrix(Y[i].reshape((n-1, 1)))
    #    yt = y-M(H, E)-mu
    #    S = M(H, M(T(P), M(P, T(H))))
    #    L = L + 0.5*(np.log(np.linalg.det(S)) + M(T(yt), np.linalg.solve(S, yt))[0,0])

    #    T2 = np.concatenate((np.zeros((n-1,(n-1)+p*n), np.float64), 
    #              np.concatenate((M(P, T(H)), P), axis=1)), axis=0)
    #    T2R = np.linalg.qr(T2)[1]
    #    P = T2R[n-1:,n-1:]
    #    K1 = np.transpose(T2R[:n-1,n-1:])
    #    K2 = T2R[:n-1,:n-1]
    #    E = M(F, E) + M(F, M(K1, np.linalg.solve(T(K2), yt)))
    #    T1 = np.concatenate((M(P, T(F)), Q**0.5), axis=0)
    #    P = np.linalg.qr(T1)[1][:n*p]

    #    Es.append(E.flatten())
    #    Ps.append(P)

    return L, np.array(Es)[:,0,:n], np.array(Ps)

def constraint(params):
    #params = get_params(params)
    a = params[:n*p].reshape(p, n)
    b = params[n*p:(2*n-1)*p].reshape(p, n-1)
    d = params[(2*n-1)*p:]*t[(2*n-1)*p:]

    A = np.zeros((p, n, n))
    B = np.zeros(((p+1)*n*n, (p+1)*n*n))
    ii = np.arange(0, n)
    A[:,ii,ii] = a
    A[:,:n-1,-1] = b

    I = np.hstack((np.eye(n*(p-1)), np.zeros((n*(p-1), n))))
    As = np.hstack([x[0] for x in np.split(A, p, axis=0)])
    F = np.vstack((As, I))
    return np.concatenate([[1-np.absolute(np.linalg.eig(F)[0]).max()], params[(2*n-1)*p:]])

init_params = t.copy()
init_params[(2*n-1)*p:] = 1.

#import tensorflow as tf
#def ols_fit(X, params):
#    n2 = n*n
#
#    a = params[:n*p]
#    b = params[n*p:(2*n-1)*p]
#    #d = params[(2*n-1)*p:(2*n-1)*p + n]
#
#    ii = np.repeat(np.arange(0, p), n).astype(np.int32)
#    jj = np.tile(np.arange(0, n), p).astype(np.int32)
#    Ad = tf.zeros((p, n, n), dtype=np.float64)
#    A = tf.sparse_add(Ad, tf.SparseTensor(np.vstack((ii, jj, jj)).T, a, (p, n, n)))
#    ii = np.repeat(np.arange(0, p), n-1).astype(np.int32)
#    jj = np.tile(np.arange(0, n-1), p).astype(np.int32)
#    kk = (n-1)*np.ones((n-1)*p).astype(np.int32)
#    A = tf.sparse_add(A, tf.SparseTensor(np.vstack((ii, jj, kk)).T, b, (p, n, n)))
#    J = 0
#    Xs = []
#    J = tf.transpose(X[p:N])
#    for i in range(1, p+1):
#        Xs = tf.transpose(X[p-i:N-i])
#        J -= tf.matmul(A[i-1], Xs)
#    Js = J
#    Jt = tf.reduce_sum(Js**2, axis=1)/(N-p)
#    J = tf.reduce_sum(Js**2)
#    print 'here0'
#    grad = tf.gradients(J, params)[0]
#    print 'here1'
#    hess = tf.hessians(J, params)[0]
#    print 'here2'
#    return Js, Jt, grad, hess


#def ols_fit_func(E):
#    b, A = sess.run([grad, hess], feed_dict={tf_params:np.zeros(Np-n), X:E})
#    a = np.linalg.solve(A, -b)
#    d = sess.run(J, feed_dict={tf_params:a, X:E})
#    params = np.zeros(Np)
#    params[:(2*n-1)*p] = a
#    params[(2*n-1)*p:] = d/t[(2*n-1)*p:]
#    return params

#sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#tf_params = tf.Variable(tf.zeros((Np-n), np.float64))
#X = tf.placeholder(np.float64)
#Js, J, grad, hess = ols_fit(X, tf_params)

params = init_params.copy()
history = []
import cPickle as pkl
for i in range(0, 10000):
    history.append(params)
    params = kalman_filter(params, s, p)
    if i % 100 == 0:
        with open("optim.pkl", 'w') as f:
            pkl.dump(history, f)
exit(0)

def plot_params(params):
    L, E, _ = kalman_filter(params, s, p)
    print params, L
    plt.plot(E[:,-1])
    #err = sess.run(Js, feed_dict={tf_params: params[:Np-n], X:E})
    #for i in range(0, n-1):
    #    print (err[i]**2).sum()
    #    plt.plot(err[i])
    #plt.show()

from plot.richardson import extrapolation
def extrapolate(params):
    a = params[:n*p].reshape(p, n)
    b = params[n*p:(2*n-1)*p].reshape(p, n-1)
    d = params[(2*n-1)*p:]*t[(2*n-1)*p:]
    ex_params = []
    for ai in a:
        ex_params.append(extrapolation(visc, ai[:n-1]))
    for bi in b:
        ex_params.append(extrapolation(visc, bi))
    ex_params.append(extrapolation(visc, d[:n-1]))
    return ex_params

params = init_params.copy()
plot_params(params)
print extrapolate(params)
plt.savefig('params1_E.pdf')
plt.clf()
exit(0)
#params2 = np.array([ 2.93999432,  2.94073627 , 2.95828042,  1.88773472, -2.9040424 , -2.8992611,
# -2.93165969, -1.06189888 , 0.96356349,  0.95824646,  0.97320128,  0.10177968,
#  0.42558045,  0.16601026 , 0.10827181, -0.72850299, -0.278598  , -0.19839493,
#  0.3787633 ,  0.14254076 , 0.10047673,  0.63210986,  0.01      ,  0.24412381,
#  1.35018268])
#plot_params(params2)
#print extrapolate(params2)
#plt.savefig('p5_nomean/params.pdf')
#params2 = np.array([ 3.06921995 , 2.84828593 , 3.47139109 , 0.5991141 , -3.2209785,  -2.28585814,
# -4.68346759,  0.13287314,  1.505171  , -0.05635432 , 3.50637432,  0.0371567,
# -1.02971844,  0.26742701, -2.21012476,  0.03058964 , 1.06909036,  0.5266578,
#  1.27204699, -0.07427776, -0.39377573, -0.30072019 ,-0.3564681 , -0.09141167,
#  0.27179061,  0.12193701,  0.06870277,  0.41314612 , 0.15126041,  0.06634894,
# -0.59557672, -0.22190422, -0.21058224, -0.05105516 ,-0.06057996,  0.12203609,
#  0.05925965,  0.09797584, -0.07180606,  0.27900204 , 0.07439238,  0.06803588,
#  0.63530284,  0.03992902,  0.13110956,  0.70080074]
#    )
#plot_params(params2)
#print extrapolate(params2)
#plt.savefig('p5_nomean/params.pdf')
#exit(0)

#for i in range(0, 100):
#    print params
#    print constraint(params)
#    L, E, P = kalman_filter(params, s, p)
#    print L
#    params = ols_fit_func(E)

from scipy.optimize import *
objective = lambda x: kalman_filter(x, s, p)[0]
con = 0.0
bounds = np.vstack((
    np.hstack((-10*np.ones((n*p, 1)), 10*np.ones((n*p, 1)))),
    np.hstack((-5*np.ones(((n-1)*p, 1)), 5*np.ones(((n-1)*p, 1)))),
    np.hstack((0.01*np.ones(((1+0)*n, 1)), 100*np.ones(((1+0)*n, 1)))),
    ))

params = params + con*np.random.rand(Np)*(bounds[:,1]-bounds[:,0])
#params = bounds[:,0] + np.random.rand(Np)*(bounds[:,1]-bounds[:,0])
# nelder mead
print fmin(objective, params, maxfun=100000000, maxiter=100000000)
# slsqp
#print fmin_slsqp(objective, params, bounds=bounds, iter=10000000)
#print fmin_slsqp(objective, params, bounds=bounds)#, ieqcons=[lambda x: constraint(x)[0]])
# newton-cg
#print fmin_ncg(objective, params)#, fprime=objective_grad)

plt.legend(loc='lower left')
plt.xlabel('time')
plt.ylabel('objective')
#plt.ylabel('adjoint energy')
#plt.ylim([-1e-10, 1e-10])
#plt.xlabel('Time unit')
#plt.ylabel('Drag over cylinder (N)')
#plt.savefig('test.png')
#plt.show()
