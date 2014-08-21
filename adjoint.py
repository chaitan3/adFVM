#!/usr/bin/python2
from __future__ import print_function
from pyRCF import Solver

primal = Solver('tests/forwardStep/', {'R': 8.314, 'Cp': 1006., 'gamma': 1.4, 'mu': 2.5e-5, 'Pr': 0.7, 'CFL': 0.2})

nSteps = 1000
writeInterval = 10

timeSteps, = primal.run([0, 1e-4], nSteps, writeInterval)

adjointFields = [CellField.zeros('{0}a'.format(phi.name), phi.mesh, phi.dimensions) for phi in primal.fields]

for checkpoint in range(0, nSteps/writeInterval):
    adjointIndex = nSteps-1 - checkpoint*writeInterval
    primalIndex = adjointIndex - writeInterval + 1
    primalIndex = nSteps - checkpoint*(writeInterval + 1)
    garbage, jacobians = primal.run(timeSteps[primalIndex], writeInterval, adjoint=True)
    for step in range(0, writeInterval):
        for index in len(0, adjointFields):
            adjointFields[index].field = jacobians[writeInterval-1 - step][index].transpose().dot(adjointFields[index].field)
    
    for adjointPhi in adjointFields:
        adjointPhi.write(timeSteps[primalIndex][0])


    

    





