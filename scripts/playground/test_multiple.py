#!/usr/bin/python
from mpi4py import MPI
import subprocess

print 
nProcessors = 4
program = './test_mpi4py.py'
nCalls = 4

print 'Host running on', MPI.Get_processor_name()
a = []
for i in range(0, nCalls):
    p = subprocess.Popen(['srun', '-N', '1', '-n', str(nProcessors), '--resv-ports', program, str(i)])
    a.append(p)
for i in range(0, nCalls):
    p.wait()
