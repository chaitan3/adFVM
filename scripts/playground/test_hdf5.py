from mpi4py import MPI
import h5py
rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
a = f.create_dataset('a', (4,), dtype='i')
b = f.create_dataset('b', (4,), dtype='i')
with a.collective:
    a[rank] = rank
    b[rank] = rank**2
f.close()
