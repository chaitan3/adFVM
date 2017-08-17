import pickle
import sys


with open('state.pkl') as f:
    b = pickle.load(f)
print [len(b[x]) for x in b.keys()]
print [x[0] for x in b['evals']]
if len(sys.argv) > 1:
    status = sys.argv[1]
    b['state'][-1] = status
else:
    print b; exit(1)

#b['state'] = b['state'][:-1]
#b['points'] = b['points'][:-1]
#b['state'] = ['ADJOINT' for i in range(0, 8)]
#for i in range(0, len(b['evals'])):
#    for j in range(6, 10):
#        b['evals'][i][j] = b['evals'][i][j]*10

with open('state.pkl', 'w') as f:
    pickle.dump(b, f)

