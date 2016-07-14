#!/usr/bin/python2
import socket
import subprocess

nProcessors = 2
program = './test_mpi4py.py'
nCalls = 4

print 'Host running on', socket.gethostname()
a = []
for i in range(0, nCalls):
    p = subprocess.Popen(['aprun', '-n', str(nProcessors), 'python', program, str(i)]) 
    #p = subprocess.Popen(['mpirun', '-n', str(nProcessors), program, str(i)])
    #p = subprocess.Popen(['srun', '-N', '1', '-n', str(nProcessors), '--resv-ports', program, str(i)])
    a.append(p)
for i in range(0, nCalls):
    a[i].wait()
