import subprocess
import os
import sys
import shutil
import glob

from adFVM import config
from adFVM.mesh import Mesh
from adFVM.field import IOField

cases_path = '../cases/'
apps_path = '../apps/'

def test_adjoint():
    case_path = os.path.join(cases_path, 'cylinder')
    problem = os.path.join('../templates/', 'cylinder_test')
    primal = os.path.join(apps_path, 'problem.py')
    adjoint = os.path.join(apps_path, 'adjoint.py')
    
    try:
        subprocess.check_output([primal, problem, '-c'])
        subprocess.check_output([primal, problem, 'perturb'])

        subprocess.check_output([adjoint, problem, '-c'])

        with open(os.path.join(case_path, 'objective.txt')) as f:
            data = f.readlines()
        fdSens = float(data[1].split(' ')[-2])
        adjSens = float(data[2].split(' ')[-2])
        diff = abs(fdSens-adjSens)/abs(fdSens)

        assert diff < 1e-3
    finally:
        map(shutil.rmtree, glob.glob(os.path.join(case_path, '1.*')))
        map(os.remove, glob.glob(os.path.join(case_path, '*.txt')))
        map(os.remove, glob.glob(os.path.join(case_path, '*.pkl')))

def test_adjoint_mpi():
    case_path = os.path.join(cases_path, 'cylinder')
    problem = os.path.join('../templates/', 'cylinder_test')
    primal = os.path.join(apps_path, 'problem.py')
    adjoint = os.path.join(apps_path, 'adjoint.py')

    try:
        subprocess.check_output(['decomposePar', '-case', case_path, '-time', '1'])
        subprocess.check_output(['mpirun', '-np', '4', primal, problem, '-c'])
        subprocess.check_output(['mpirun', '-np', '4', primal, problem, 'perturb'])

        subprocess.check_output(['mpirun', '-np', '4', adjoint, problem, '-c'])

        with open(os.path.join(case_path, 'processor0', 'objective.txt')) as f:
            data = f.readlines()
        fdSens = float(data[1].split(' ')[-2])
        adjSens = float(data[2].split(' ')[-2])
        diff = abs(fdSens-adjSens)/abs(fdSens)

        assert diff < 1e-3
    finally:
        map(shutil.rmtree, glob.glob(os.path.join(case_path, 'processor*')))

if __name__ == '__main__':
    #test_adjoint()
    test_adjoint_mpi()

