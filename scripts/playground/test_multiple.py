from mpi4py import MPI
import subprocess

nProcessors = 2
nCalls = 2
program = './test_mpi4py.py'

for i in range(0, nCalls):
    with open('output.log', 'a') as f:
        returncode = subprocess.call(['mpirun', '-np', str(nProcessors),
                          program], stdout=f, stderr=f)
