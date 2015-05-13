from mpi4py import MPI
import theano

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

c = theano.shared(rank + 2)
if rank == 0:
    a = theano.tensor.scalar()
    b = a*c
    f = theano.function([a], b)
    comm.send(f, dest=1, tag=11)
elif rank == 1:
    f = comm.recv(source=0, tag=11)
print f(rank)
