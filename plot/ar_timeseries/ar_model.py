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
for f in sys.argv[1:]:
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
    ys -= ys.mean()
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
n = len(s) + 1
Np = (2*n-1)*p + n
s = np.array(s)
# HACK
#s = s[:,:10]
N = len(s[0])

t = np.zeros(Np)
import ar
import matplotlib.pyplot as plt
for i in range(0, n-1):
    A = ar.arsel(s[i], 0, 1, "CIC", p, p)
    c = -np.array(A.AR[0][1:])
    t[i:n*p:n] = c
    t[n*i+n-1] = 1e-1
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
    


t[n*p:(n+1)*p] = 1e-2
t[-1] = t[(2*n-1)*p:].max()

import tensorflow as tf
def loglikelihood_tf(params, s, p):
    n2 = n*n

    a = params[:n*p]
    b = params[n*p:(2*n-1)*p]
    d = params[(2*n-1)*p:(2*n-1)*p + n]

    ii = np.repeat(np.arange(0, p), n).astype(np.int32)
    jj = np.tile(np.arange(0, n), p).astype(np.int32)
    Ad = tf.zeros((p, n, n), dtype=np.float64)
    A = tf.sparse_add(Ad, tf.SparseTensor(np.vstack((ii, jj, jj)).T, a, (p, n, n)))
    ii = np.repeat(np.arange(0, p), n-1).astype(np.int32)
    jj = np.tile(np.arange(0, n-1), p).astype(np.int32)
    kk = (n-1)*np.ones((n-1)*p).astype(np.int32)
    A = tf.sparse_add(A, tf.SparseTensor(np.vstack((ii, jj, kk)).T, b, (p, n, n)))
    E = tf.diag(d)
    #cdef np.ndarray[dtype, ndim=2] E = np.diag(d)

    a1 = []
    a2 = []
    a22 = []
    a3 = []
    a4 = []
    for m in range(0, n):
        for k in range(0, n):
            for l in range(0, n):
                a1.append(m*n + k)
                a2.append(m*n + l)
                a22.append(l*n + m)
                a3.append(k)
                a4.append(l)
    a1 = np.array(a1).astype(np.int32)
    a2 = np.array(a2).astype(np.int32)
    a22 = np.array(a22).astype(np.int32)
    a3 = np.array(a3).astype(np.int32)
    a4 = np.array(a4).astype(np.int32)
    a34 = np.vstack((a3, a4)).T
    B = tf.eye((p+1)*n2, dtype=np.float64)
    #A = tf.Print(A, [A])
    for i in range(0, p+1):
        index = 0
        for j in range(i-1, -1, -1):
            B = tf.sparse_add(B, tf.SparseTensor(np.vstack((i*n2 + a1, j*n2 + a22)).T, -tf.gather_nd(A[index], a34), ((p+1)*n2,(p+1)*n2)))
            index += 1
        for j in range(1, p-i+1):
            B = tf.sparse_add(B, tf.SparseTensor(np.vstack((i*n2 + a1, j*n2 + a2)).T, -tf.gather_nd(A[index], a34), ((p+1)*n2,(p+1)*n2)))
            index += 1
    r = tf.concat((tf.reshape(E, (n2,)), tf.zeros((p*n2,), dtype=np.float64)), axis=0)
    #r = tf.sparse_to_dense(np.arange(0, n2), ((p+1)*n2,), tf.reshape(E, (n2,)))
    Ej = tf.matrix_solve(B, tf.reshape(r, ((p+1)*n2, 1)))
    Ej = tf.reshape(Ej, (p+1, n, n))
    #Ej = tf.Print(Ej, [Ej])
    Es = []
    print 'here0'
    for i in range(0, p+1):
        Es.append(Ej[i])
    for i in range(p+1, N):
        Es.append(0)
        for j in range(i-1, i-p-1, -1):
            Es[i] = Es[i] + tf.matmul(A[i-j-1], Es[j])
    #Es[-1] = tf.Print(Es[-1], [Es[-1]])
    print 'here1'
    n1 = n-1
    Es = tf.concat([x[:n1, :n1] for x in Es], axis=1)
    S = []
    for i in range(0, N):
        S.append(tf.concat((tf.zeros((n1, i*n1), dtype=np.float64), Es[:,:(N-i)*n1]), axis=1))
    S = tf.concat(S, axis=0)
    print 'here2'
    St = tf.matrix_band_part(S, 0, -1)
    #S = tf.transpose(St) + St - tf.diag(tf.diag_part(S) + 1e-30)
    S = tf.transpose(St) + St - tf.diag(tf.diag_part(S))
    R = tf.cholesky(S)
    Y = (s.T)[::-1].flatten()
    X = tf.cholesky_solve(R, Y.reshape(Y.shape[0], 1))
    z1 = tf.matmul(tf.reshape(Y, (1, Y.shape[0])), X)[0,0]
    z2 = 2*tf.reduce_sum(tf.log(tf.diag_part(R)))
    print 'here3'
    res = 0.5*z1 + 0.5*z2
    return res



def loglikelihood(params, s, p):
    n2 = n*n

    a = params[:n*p].reshape(p, n)
    b = params[n*p:(2*n-1)*p].reshape(p, n-1)
    d = params[(2*n-1)*p:(2*n-1)*p + n]

    A = np.zeros((p, n, n))
    B = np.eye((p+1)*n*n)
    ii = np.arange(0, n)
    A[:,ii,ii] = a
    A[:,:n-1,-1] = b

    E = np.diag(d)
    #cdef np.ndarray[dtype, ndim=2] E = np.diag(d)

    print a, b, d
    a1 = []
    a2 = []
    a22 = []
    a3 = []
    a4 = []
    for m in range(0, n):
        for k in range(0, n):
            for l in range(0, n):
                a1.append(m*n + k)
                a2.append(m*n + l)
                a22.append(l*n + m)
                a3.append(k)
                a4.append(l)
    a1 = np.array(a1)
    a2 = np.array(a2)
    a22 = np.array(a22)
    a3 = np.array(a3)
    a4 = np.array(a4)
    for i in range(0, p+1):
        index = 0
        for j in range(i-1, -1, -1):
            B[i*n2 + a1, j*n2 + a22] += -A[index, a3, a4]
            index += 1
        for j in range(1, p-i+1):
            B[i*n2 + a1, j*n2 + a2] += -A[index, a3, a4]
            index += 1
    r = np.zeros((p+1)*n2)
    r[:n2] = E.flatten()
    Ej = np.linalg.solve(B, r)
    Ej = Ej.reshape(p+1, n2)
    Es = np.zeros((N, n2))
    Es[:p+1] = Ej
    for i in range(p+1, N):
        for j in range(i-1, i-p-1, -1):
            Es[i] += np.dot(A[i-j-1], Es[j].reshape(n, n)).flatten()
    n1 = n-1
    S = np.zeros((N*n1, N*n1))
    Ess = np.hstack([x[0].reshape(n, n)[:-1,:-1] for x in np.split(Es, N, axis=0)])
    for i in range(0, N):
        S[i*n1:i*n1+n1, i*n1:] = Ess[:,:(N-i)*n1]
    print 'here1'
    S = np.triu(S) + np.triu(S, k=1).T
    print S.max()
    R = sp.linalg.cholesky(S)
    Y = (s.T)[::-1].flatten()
    z1 = np.dot(Y, sp.linalg.cho_solve((R, False), Y))
    z2 = 2*np.log(np.diag(R)).sum()
    print 'here2'
    print z1, z2
    res = 0.5*z1 + 0.5*z2
    return res



def constraint(params):
    params = get_params(params)
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
    return 1-np.absolute(np.linalg.eig(F)[0]).max()

def objective(params):
    params = get_params(params)
    params = params.copy()
    params[(2*n-1)*p:] *= t[(2*n-1)*p:]
    #return ARLogLikelihood(params, s, p)
    return loglikelihood(params, s, p)

init_params = t.copy()
init_params[(2*n-1)*p:] = 1.
#print objective(params)

def get_params(params):
    full_params = init_params.copy()
    full_params[np.arange(0, p)*n + n-1] = params[:p]
    full_params[n*p:n*p+n-1] = params[p:-1]
    full_params[-1] = params[-1]
    return full_params

def red_params(full_params):
    params = np.concatenate((
            full_params[np.arange(0, p)*n + n-1],
            full_params[n*p:n*p+n-1],
            full_params[[-1]]
        ))
    return params

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf_params = tf.placeholder(np.float64)
res = loglikelihood_tf(tf_params, s, p)
grad = tf.gradients(res, [tf_params])

def objective(params):
    #print 'obj', params, constraint(params)
    #if constraint(params) <= 1e-9:
    #    return np.inf
    params = get_params(params)
    params = params.copy()
    params[(2*n-1)*p:] *= t[(2*n-1)*p:]
    return sess.run(res, feed_dict={tf_params:params})

def objective_grad(params):
    #print 'grad', params, constraint(params)
    #if constraint(params) <= 1e-9:
    #    return np.inf*np.ones_like(params)
    params = get_params(params)
    params = params.copy()
    params[(2*n-1)*p:] *= t[(2*n-1)*p:]
    grads = sess.run(grad, feed_dict={tf_params:params})[0]
    grads[(2*n-1)*p:] *= t[(2*n-1)*p:]
    return red_params(grads)

from adFVM.compat import ARLogLikelihood

import nlopt 
def nlopt_objective(params, grads):
    #print params
    #if (constraint(params) < 0):
    #    return np.inf
    params = get_params(params)
    params = params.copy()
    params[(2*n-1)*p:] *= t[(2*n-1)*p:]

    if grads.size > 0:
        tgrads[:] = sess.run(grad, feed_dict={tf_params:params})[0]
        tgrads[(2*n-1)*p:] *= t[(2*n-1)*p:]
        grads[:] = red_params(tgrads)
        #print grads
    result = sess.run(res, feed_dict={tf_params:params})
    print params, constraint(params), result
    return result

def nlopt_constraint(params, grad):
    params = get_params(params)
    #print 'constraint', params
    result = -constraint(params)
    return result

#opt = nlopt.opt(nlopt.GN_ISRES, Np)
#opt = nlopt.opt(nlopt.LD_MMA, Np)
#opt.set_min_objective(nlopt_objective)
#opt.add_inequality_constraint(nlopt_constraint, 0)
#opt.verbose = 1
##opt.set_inequality_constraint()
#print opt.optimize(params)
from scipy.optimize import *
#print ARLogLikelihood(t, s, p)

#print fmin_slsqp(objective, red_params(init_params), fprime=objective_grad, f_ieqcons=constraint, iter=1000)
#print minimize(loglikelihood, np.random.rand(Np), jac=lambda x: loglikelihood(x, True), method='BFGS', options={'maxiter':100})

bounds = np.vstack((
    np.hstack((-10*np.ones((p, 1)), 10*np.ones((p, 1)))),
    np.hstack((0*np.ones((n-1, 1)), 1*np.ones((n-1, 1)))),
    np.array([[0.1, 10]])
    ))

npoints = 10
points = np.linspace(0, 1, npoints+2)[1:-1]
import pyDOE
designs = pyDOE.fullfact([npoints]*(n+p)).astype(np.int32)
data = []
import cPickle as pkl
params = [red_params(init_params)]
for i, des in enumerate(designs):
    print i
    param = []
    for j in range(0, n+p):
        param.append(bounds[j][0]+points[des[j]]*(bounds[j][1]-bounds[j][0]))
    if constraint(param) < 0:
        continue
    #result = objective(np.array(param))
    try:
        result = objective(np.array(param))
        data.append([param, result, constraint(param)])
        print data[-1]
    except:
        pass
    if i % 1000 == 0:
        with open("optim.pkl", "w") as f:
            pkl.dump(data, f)


plt.legend(loc='lower left')
plt.xlabel('time')
plt.ylabel('objective')
#plt.ylabel('adjoint energy')
#plt.ylim([-1e-10, 1e-10])
#plt.xlabel('Time unit')
#plt.ylabel('Drag over cylinder (N)')
#plt.savefig('test.png')
#plt.show()
