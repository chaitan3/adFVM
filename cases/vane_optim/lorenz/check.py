import cPickle as pkl

with open('optim_nd2_lorenz.pkl') as f:
    a = pkl.load(f)

evals, gps, values = a
import numpy as np
import matplotlib.pyplot as plt
#import pdb;pdb.set_trace()

print len(evals)
y = np.array([[x[1] for x in z] for z in gps])
#plt.plot(np.mean(y, axis=0))
X = np.array([[x[0] for x in z] for z in gps])
print X.shape
plt.plot(np.mean(X[:,:,1], axis=0))
#plt.scatter(X[:,0], X[:,1])

plt.show()
