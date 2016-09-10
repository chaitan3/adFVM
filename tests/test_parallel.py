import subprocess
import os
import sys
import shutil
import glob

from test import *
from adFVM import config
from adFVM.mesh import Mesh
from adFVM.field import IOField

def test_mpi_comm_method(case_path):
    mesh = Mesh.create(case_path)
    IOField.setMesh(mesh)
    with IOField.handle(1.):
        U = IOField.read('U')
        U.partialComplete()
    with IOField.handle(2.):
        U.write()
    return

class TestParallel(unittest.TestCase):
    def test_mpi_comm(self):
        case_path = os.path.join(cases_path, 'forwardStep')
        mesh = Mesh.create(case_path)
        IOField.setMesh(mesh)
        with IOField.handle(0.):
            U = IOField.read('U')
            U.partialComplete()
        U.field = np.random.rand(*U.field.shape)
        with IOField.handle(1.):
            U.write()
        subprocess.check_call(['decomposePar', '-case', case_path, '-time', '1'])

        subprocess.check_call(['mpirun', '-np', '4', 'python2', __file__, 'RUN', 'test_mpi_comm_method', case_path])

        try:
            checkFields(self, case_path, 'U', '1.0', '2.0', relThres=1e-12, nProcs=4)
        finally:
            shutil.rmtree(os.path.join(case_path, '1'))
            map(shutil.rmtree, glob.glob(os.path.join(case_path, 'processor*')))

    def test_mpi(self):
        solver = os.path.join(apps_path, 'pyRCF.py')
        case_path = os.path.join(cases_path, 'forwardStep')
        endTime = '0.2'
        args = ['-t', endTime, '-w', '1000', '-f', 'AnkitENO', '-v', '--Cp', '2.5', '--dt', '1e-4']
        #subprocess.check_call([solver, case_path, '0.0'] + args)
        folder = float(endTime)
        if folder.is_integer():
            folder = int(folder)
        folder = str(folder)
        subprocess.check_call(['mv', os.path.join(case_path, folder), os.path.join(case_path, '10')])

        subprocess.check_call(['decomposePar', '-case', case_path, '-time', '0'])
        for pkl in glob.glob(os.path.join(case_path, '*.pkl')):
            shutil.copy(pkl, os.path.join(case_path, 'processor0'))
        subprocess.check_call(['mpirun', '-np', '4', solver, case_path, '0.0'] + args)
        subprocess.check_call(['reconstructPar', '-case', case_path, '-time', endTime])

        try:
            checkFields(self, case_path, 'U', endTime, '10.0')
        finally:
            shutil.rmtree(os.path.join(case_path, endTime))
            shutil.rmtree(os.path.join(case_path, '10'))
            map(shutil.rmtree, glob.glob(os.path.join(case_path, 'processor*')))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'RUN':
        func = locals()[sys.argv[2]]
        func(*sys.argv[3:])
    else:
        unittest.main(verbosity=2, buffer=True)
