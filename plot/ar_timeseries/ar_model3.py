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
submean = False
#submean = True
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

print ms

p = 3
n = len(s) + 1
Np = (2*n-1)*p + n
if not submean:
    Np += n
s = np.array(s)
# HACK
#s = s[:,:100]
N = len(s[0])

t = np.zeros(Np)
import ar
import matplotlib.pyplot as plt
for i in range(0, n-1):
    A = ar.arsel(s[i], 0, 1, "CIC", p, p)
    c = -np.array(A.AR[0][1:])
    t[i:n*p:n] = c
    t[n-1:n*p:n] = c
    #t[n*i+n-1] = 1.
    t[(2*n-1)*p + i] = A.sigma2eps[0]
    if not submean:
        t[(2*n-1)*p + n + i] = ms[i]

    #yp = []
    #for j in range(p, N):
    #    yp.append(np.dot(s[i][j-p:j+1][::-1],A.AR[0]))
    #plt.plot(yp)
    ##plt.hist(yp, bins=100)
    #plt.show()

    #from scipy.signal import lfilter, lfiltic
    #zp = lfiltic([1], A.AR[0], s[i][:p][::-1])
    #yp = A.mu[0] + lfilter([1], A.AR[0], np.sqrt(A.sigma2eps[0])*np.random.randn(3000), zi=zp)[0]
    #yp = A.mu[0] + lfilter([1], A.AR[0], np.sqrt(A.sigma2eps[0])*np.zeros(3000), zi=zp)[0]
    #yp = -np.convolve(s[i], A.AR[0][1:], 'valid')
    #plt.plot(np.arange(0, len(yp)), yp, label='model')
    #plt.plot(np.arange(0, len(s[i][p:])), s[i][p:], label='data')
    #plt.legend()
    #plt.show()
    #print A.AR[0], A.sigma2eps[0]
    #plt.plot(m.AR[0], 'o', label=os.path.basename(f))
    #print m.sigma2eps[0]

t[n*p:(n+1)*p] = 1.
t[(2*n-1)*p+n-1] = t[(2*n-1)*p:(2*n-1)*p+n].max()
if not submean:
    t[-1] = t[(2*n-1)*p+n:].max()

import tensorflow as tf
sess = tf.Session()

def kalman_filter(params, s, p):
    n2 = n*n
    M = lambda A, B: tf.matmul(A, B)
    T = lambda A: tf.transpose(A)

    a = params[:n*p]
    b = params[n*p:(2*n-1)*p]
    d = params[(2*n-1)*p:(2*n-1)*p+n]*t[(2*n-1)*p:(2*n-1)*p+n]
    if not submean:
        mu = tf.reshape(params[(2*n-1)*p+n:]*t[(2*n-1)*p+n:], (n,1))[:-1]
    else:
        mu = np.zeros((n-1,1))

    ii = np.repeat(np.arange(0, p), n).astype(np.int32)
    jj = np.tile(np.arange(0, n), p).astype(np.int32)
    A = tf.zeros((p, n, n), dtype=np.float64)
    A = tf.sparse_add(A, tf.SparseTensor(np.vstack((ii, jj, jj)).T, a, (p, n, n)))
    ii = np.repeat(np.arange(0, p), n-1).astype(np.int32)
    jj = np.tile(np.arange(0, n-1), p).astype(np.int32)
    kk = (n-1)*np.ones((n-1)*p).astype(np.int32)
    A = tf.sparse_add(A, tf.SparseTensor(np.vstack((ii, jj, kk)).T, b, (p, n, n)))
    E = tf.diag(d)

    F = tf.concat([A[i] for i in range(0, p)], axis=1)
    Fd = tf.concat((tf.eye((p-1)*n, dtype=np.float64), tf.zeros(((p-1)*n, n), np.float64)), axis=1)
    F = tf.concat((F, Fd), axis=0)
    Q = tf.diag(tf.concat((d, tf.zeros((p-1)*n, np.float64)), axis=0))

    H = np.zeros(((n-1), n*p))
    ii = np.arange(0, n-1)
    H[ii, ii] = 1

    Fk = []
    for i in range(0, n*p):
        Fi = []
        for j in range(0, n*p):
            Fi.append(F[i,j]*F)
        Fk.append(tf.concat(Fi, axis=1))
    Fk = tf.concat(Fk, axis=0)

    I = tf.eye(n*p, dtype=np.float64)
    Y = tf.Variable(np.array(s).T)
    L = tf.Variable(np.float64(0.))
    Es = []

    # conditional mle vs mle

    # loglikelihood prior
    #P = tf.matmul(tf.matrix_inverse(tf.eye(n**2*p**2, dtype=np.float64)-Fk), tf.reshape(Q, (p**2*n**2, 1)))
    #P = tf.reshape(P, (n*p, n*p))
    #P = (P + T(P))/2
    #P = T(tf.cholesky(P))
    #E = tf.zeros((p*n, 1), np.float64)
    #i = tf.constant(0)

    P = Q[:]**0.5
    Ei = tf.concat((Y[:p]-tf.reshape(mu, (1, n-1)), np.zeros((p, 1))), axis=1)
    E = sum([tf.matmul(A[i], tf.reshape(Ei[p-i-1], (n,1))) for i in range(0, p)])
    E = tf.reshape(tf.concat((tf.reshape(E, (1,n)), Ei[::-1][:-1]), axis=0), (n*p, 1))
    i = tf.constant(p)

    def qr(A):
        Q, R = tf.qr(A)
        #m, n = A.shape.as_list()
        #norm = lambda v, u: tf.reduce_sum(v*u)
        #qi = tf.reshape(A[:,0], (m,1))
        #ri = norm(qi, qi)
        #q = [qi/ri]
        #R = [tf.sparse_to_dense([0], (n,), [ri])]
        #for i in range(1, n):
        #    qi = tf.reshape(A[:,i], (m, 1))
        #    r = []
        #    for j in range(0, i):
        #        ri = norm(qi, q[j])
        #        qi -= ri*q[j]
        #        r.append(ri)
        #    r.append(norm(qi, qi)**0.5)
        #    q.append(qi/r[-1])
        #    R.append(tf.sparse_to_dense(np.arange(0, i+1), (n,), ri))
        #Q = tf.concat(q, axis=1)
        #R = tf.transpose(tf.concat([tf.reshape(r, (n, 1)) for r in R], axis=1))
        return Q, R

    def condition(i, E, P, L):
        return i < N
    def body(i, E, P, L):
        y = tf.reshape(Y[i], (n-1, 1))
        yt = y-M(H,E)-mu
        S = M(H, M(T(P), M(P, T(H))))
        L = L + 0.5*(tf.log(tf.matrix_determinant(S)) + M(T(yt), tf.matrix_solve(S, yt))[0,0])

        T2 = tf.concat((tf.zeros((n-1,(n-1)+p*n), np.float64), 
                  tf.concat((M(P, T(H)), P), axis=1)), axis=0)
        T2R = qr(T2)[1]
        P = T2R[n-1:,n-1:]
        K1 = tf.transpose(T2R[:n-1,n-1:])
        K2 = T2R[:n-1,:n-1]
        E = M(F, E) + M(F, M(K1, tf.matrix_solve(T(K2), yt)))
        T1 = tf.concat((M(P, T(F)), Q**0.5), axis=0)
        P = qr(T1)[1][:n*p]
        return [i+1, E, P, L]
    _, E, _, L = tf.while_loop(condition, body, [i, E, P, L])
    #Lg = tf.gradients(L, params)[0]
    Lg = tf.constant(2)
    return L, Lg#, Lg, tf.stack(Es, axis=0)[:,:n,0]

def constraint(params):
    #params = get_params(params)
    a = params[:n*p].reshape(p, n)
    b = params[n*p:(2*n-1)*p].reshape(p, n-1)
    d = params[(2*n-1)*p:(2*n-1)*p+n]*t[(2*n-1)*p:(2*n-1)*p+n]

    A = np.zeros((p, n, n))
    B = np.zeros(((p+1)*n*n, (p+1)*n*n))
    ii = np.arange(0, n)
    A[:,ii,ii] = a
    A[:,:n-1,-1] = b

    I = np.hstack((np.eye(n*(p-1)), np.zeros((n*(p-1), n))))
    As = np.hstack([x[0] for x in np.split(A, p, axis=0)])
    F = np.vstack((As, I))
    return np.concatenate([[1-np.absolute(np.linalg.eig(F)[0]).max()], params[(2*n-1)*p:(2*n-1)*p+n]])

init_params = t.copy()
init_params[(2*n-1)*p:] = 1.
init_params[-1] = 0.

tf_params = tf.Variable(np.zeros(Np), np.float64)
L, Lg = kalman_filter(tf_params, s, p)
init = tf.global_variables_initializer()
sess.run(init)

evals = 0
import cPickle as pkl
data = []
def kalman_filter_func(params, grad=True):
    global evals, data
    cons = constraint(params)
    print 'param:', params
    print 'constraint:', cons
    if not grad:
        res = sess.run(L, feed_dict={tf_params:params})
        data.append([params, cons, [res]])
    else:
        res = sess.run([L, Lg], feed_dict={tf_params:params})
        data.append([params, cons, res])
    print 'res:', res
    print
    #evals += 1
    #if evals % 100 == 0:
    #    with open('optim.pkl', 'w') as f:
    #        pkl.dump(data, f)
    return res

params = init_params.copy()
kalman_filter_func(params)[1]

bounds = np.vstack((
    np.hstack((-10*np.ones((n*p, 1)), 10*np.ones((n*p, 1)))),
    np.hstack((-5*np.ones(((n-1)*p, 1)), 5*np.ones(((n-1)*p, 1)))),
    np.hstack((0.0001*np.ones((n, 1)), 100*np.ones((n, 1)))),
    ))
if not submean:
    bounds = np.vstack((
        bounds, 
        np.hstack((0.5*np.ones((n, 1)), 1.5*np.ones((n, 1)))),
        ))

from scipy.optimize import *
objective = lambda x: kalman_filter_func(x, grad=False)
objective_grad = lambda x: kalman_filter_func(x)[1]
con = 0.0
params = params + con*np.random.rand(Np)*(bounds[:,1]-bounds[:,0])
#params = bounds[:,0] + np.random.rand(Np)*(bounds[:,1]-bounds[:,0])
# nelder mead
#print fmin(objective, params, maxfun=100000000, maxiter=100000000)
# slsqp
print fmin_slsqp(objective, params, bounds=bounds, iter=10000000)
#print fmin_slsqp(objective, params, bounds=bounds)#, ieqcons=[lambda x: constraint(x)[0]])
# newton-cg
#print fmin_ncg(objective, params)#, fprime=objective_grad)

#import nlopt 
#def nlopt_objective(params, grads):
#    res = kalman_filter_func(params, False)
#    #res = kalman_filter_func(params, grads.size > 0)
#    #if grads.size > 0:
#    #    grads[:] = res[1]
#    #    return res[0]
#    #return res
#    return res
#
#def nlopt_constraint(params, grad):
#    result = -constraint(params)[0]
#    return result
#
#opt = nlopt.opt(nlopt.LD_MMA, Np)
#opt.set_min_objective(nlopt_objective)
#opt.set_lower_bounds(bounds[:,0])
#opt.set_upper_bounds(bounds[:,1])
##opt.add_inequality_constraint(nlopt_constraint)
#print opt.optimize(params)

plt.legend(loc='lower left')
plt.xlabel('time')
plt.ylabel('objective')
#plt.ylabel('adjoint energy')
#plt.ylim([-1e-10, 1e-10])
#plt.xlabel('Time unit')
#plt.ylabel('Drag over cylinder (N)')
#plt.savefig('test.png')
#plt.show()
