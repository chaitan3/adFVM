import theano
import cPickle

class Mul:
    def __init__(self):
        self.m = theano.shared(2)

    def __getstate__():
        print 'here'
        return {}

class Test:
    def __init__(self, mul):
        a = theano.tensor.scalar()
        b = a*mul.m
        self.f = theano.function([a], b)
        pkl = cPickle.dumps(self.f)

    def __call__(self, x):
        return self.f(x)

m = Mul()
o = Test(m)
print o(3)

#from mpi4py import MPI
#import theano

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()

#comm.Barrier()
#print rank, 'overcome barrier'

#if rank == 0:
#    c = theano.shared(rank)
#    a = theano.tensor.scalar()
#    b = a*c
#    f = theano.function([a], b)
#else:
#    f = None
#    c = None
#print rank, 'compiled function, starting bcast'
#c = comm.bcast(c, root=0)
#f = comm.bcast(f, root=0)
#c.set_value(rank)
#print rank, 'finished bcast'
#v = 3
#print rank, f(v)
