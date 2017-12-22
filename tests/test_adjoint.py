import subprocess
import os
import sys
import shutil
import glob

from adFVM import config
from adFVM.mesh import Mesh
from adFVM.field import IOField

cases_path = '../cases/'

class TestAdjoint(unittest.TestCase):
    def test_adjoint(self):
        case_path = os.path.join(cases_path, 'cylinder')
        problem = os.path.join(templates_path, 'cylinder_test')
        primal = os.path.join(apps_path, 'problem.py')
        adjoint = os.path.join(apps_path, 'adjoint.py')
        
        subprocess.check_output([primal, problem])
        subprocess.check_output([primal, problem, 'perturb'])

        subprocess.check_output([adjoint, problem])

        with open(os.path.join(case_path, 'objective.txt')) as f:
            data = f.readlines()
        fdSens = float(data[1].split(' ')[-1])
        adjSens = float(data[2].split(' ')[-1])
        diff = abs(fdSens-adjSens)/abs(fdSens)

        try:
            self.assertAlmostEqual(0., diff, delta=1e-2)
        finally:
            map(shutil.rmtree, glob.glob(os.path.join(case_path, '1.*')))
            map(os.remove, glob.glob(os.path.join(case_path, '*.txt')))

    def test_adjoint_mpi(self):
        case_path = os.path.join(cases_path, 'cylinder')
        problem = os.path.join(templates_path, 'cylinder_test')
        primal = os.path.join(apps_path, 'problem.py')
        adjoint = os.path.join(apps_path, 'adjoint.py')

        subprocess.check_output(['decomposePar', '-case', case_path, '-time', '1'])
        for pkl in glob.glob(os.path.join(case_path, '*.pkl')):
            shutil.copy(pkl, os.path.join(case_path, 'processor0'))
        
        subprocess.check_output(['mpirun', '-np', '4', primal, problem])
        subprocess.check_output(['mpirun', '-np', '4', primal, problem, 'perturb'])

        subprocess.check_output(['mpirun', '-np', '4', adjoint, problem])

        with open(os.path.join(case_path, 'processor0', 'objective.txt')) as f:
            data = f.readlines()
        fdSens = float(data[1].split(' ')[-1])
        adjSens = float(data[2].split(' ')[-1])
        diff = abs(fdSens-adjSens)/abs(fdSens)

        try:
            self.assertAlmostEqual(0., diff, delta=1e-2)
        finally:
            map(shutil.rmtree, glob.glob(os.path.join(case_path, 'processor*')))

if __name__ == '__main__':
    unittest.main(verbosity=2, buffer=True)

