from mpi4py import MPI
import h5py
rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
f = h5py.File('parallel_test.hdf5', 'a', driver='mpio', comm=MPI.COMM_WORLD)
#dset = f.create_dataset('test', (4,), dtype='i')
#dset[rank] = rank
print rank
f.create_dataset('test' + str(rank), data=2)
f.close()
