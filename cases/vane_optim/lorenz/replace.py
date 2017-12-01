import pickle
import sys
import numpy as np

with open('state.pkl') as f:
    b = pickle.load(f)
print [(x, len(b[x])) for x in b.keys()]
print [x[0] for x in b['evals']]
if len(sys.argv) > 1:
    status = sys.argv[1]
    #b['state'][-1] = 'PRIMADJ'
    #b['points'].append(np.array([  3.83163784e-14,   3.33333333e-01,   2.18844952e-14,  6.66666667e-01]))
    del b['state'][-1]
    del b['points'][-1]
else:
    print b['evals'][3]
    print sum([x[0] for x in b['evals']])/8
    print b; exit(1)

#b['state'] = b['state'][:-1]
#b['points'] = b['points'][:-1]
#b['state'] = ['ADJOINT' for i in range(0, 8)]
#for i in range(0, len(b['evals'])):
#    for j in range(6, 10):
#        b['evals'][i][j] = b['evals'][i][j]*10

with open('state.pkl', 'w') as f:
    pickle.dump(b, f)

