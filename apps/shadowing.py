#!/usr/bin/python2
import numpy as np

import fds
from adFVM.interface import SerialRunner

class Shadowing(SerialRunner):
    def __init__(self, *args):
        super(Shadowing, self).__init__(*args)

    def solve(self, initFields, parameter, nSteps, runID):
        case = self.base + 'temp/' + runID
        self.copyCase(case)
        data = self.runCase(initFields, (parameter, nSteps), case)
        return data

def main():
    base = 'cases/3d_cylinder/'
    time = 2.0
    template = 'templates/3d_cylinder_fds.py'

    runner = Shadowing(base, time, template)

    nSegments = 2
    nSteps = 10
    nExponents = 2
    runUpSteps = 5
    parameter = 0.0

    fields = runner.readFields(base, time)
    fds.shadowing(runner.solve, fields, parameter, nExponents, nSegments, nSteps, runUpSteps, checkpoint_path=base + 'checkpoint/')

if __name__ == '__main__':
    main()
