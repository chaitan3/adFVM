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
    base = 'cases/vane/laminar/'
    time = 3.0
    template = 'templates/vane_runner.py'

    runner = Shadowing(base, time, template)
    fds.shadowing(runner.solve)


if __name__ == '__main__':
    main()
