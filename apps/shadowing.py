#!/usr/bin/python
import numpy as np
import os
import shutil

import fds
from adFVM.interface import SerialRunner

class Shadowing(SerialRunner):
    def __init__(self, *args, **kwargs):
        super(Shadowing, self).__init__(*args, **kwargs)

    def solve(self, initFields, parameter, nSteps, run_id):
        case = self.base + 'temp/' + run_id + "/"
        self.copyCase(case)
        data = self.runPrimal(initFields, (parameter, nSteps), case)
	shutil.rmtree(case)
        return data

    def adjoint(self, initPrimalFields, paramter, nSteps, initAdjointFields, run_id):
        case = self.base + 'temp/' + run_id + "/"
        self.copyCase(case)
        data = self.runAdjoint(initPrimalFields, (parameter, nSteps), initAdjointFields, case)
        return 

def main():
    base = 'cases/3d_cylinder/'
    time = 2.0
    dt = 6e-9
    template = 'templates/3d_cylinder_fds.py'
    nProcs = 1

    runner = Shadowing(base, time, dt, template, nProcs=nProcs, flags=['-g', '--gpu_double'])

    nSegments = 400
    nSteps = 500
    #nSteps = 400?
    nExponents = 20
    runUpSteps = 0
    parameter = 0.0
    checkpointPath = base + 'checkpoint/'
    if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)

    fields = runner.readFields(base, time)
    J, dJds_tan = fds.shadowing(runner.solve, fields, parameter, nExponents, nSegments, nSteps, runUpSteps, checkpoint_path=checkpointPath)
    #dJds_adj = fds.adjoint_shadowing(runner.solve, runner.adjoint, parameter, nExponents, checkpointPath)

if __name__ == '__main__':
    main()
