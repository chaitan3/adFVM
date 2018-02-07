#!/usr/bin/python2
import numpy as np
import os

import fds
from adFVM.interface import SerialRunner

class Shadowing(SerialRunner):
    def __init__(self, *args, **kwargs):
        super(Shadowing, self).__init__(*args, **kwargs)

    def solve(self, initFields, parameter, nSteps, run_id):
        case = self.base + 'temp/' + run_id + "/"
        self.copyCase(case)
        data = self.runCase(initFields, (parameter, nSteps), case)
        return data

def main():
    base = 'cases/3d_cylinder/'
    time = 2.0
    template = 'templates/3d_cylinder_fds.py'
    nProcs = 16

    runner = Shadowing(base, time, template, nProcs=nProcs)

    nSegments = 20
    nSteps = 1000
    #nSteps = 400?
    nExponents = 20
    runUpSteps = 0
    parameter = 0.0
    checkpointPath = base + 'checkpoint/'
    if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)

    fields = runner.readFields(base, time)
    fds.shadowing(runner.solve, fields, parameter, nExponents, nSegments, nSteps, runUpSteps, checkpoint_path=checkpointPath)
    
    # adjoint?

if __name__ == '__main__':
    main()
