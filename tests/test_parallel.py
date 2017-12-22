import subprocess
import os
import sys
import shutil
import glob
import time
import numpy as np

from adFVM import config
from adFVM.mesh import Mesh
from adFVM.field import IOField

cases_path = '../cases/'
scripts_path = '../scripts'
apps_path = '../apps'

def test_mpi_comm_method(case_path):
    mesh = Mesh.create(case_path)
    IOField.setMesh(mesh)
    with IOField.handle(1.):
        U = IOField.read('U')
        U.partialComplete()
    with IOField.handle(2.):
        U.write()
    return

def checkFields(case, field, time1, time2, relThres=1e-9, nProcs=1):
    diff = os.path.join(scripts_path, 'field', 'diff_fields.py')
    if nProcs == 1:
        output = subprocess.check_output([diff, case, field, time1, time2])
    else:
        output = subprocess.check_output(['mpirun', '-np', str(nProcs), diff, case, field, time1, time2])
    output = output.split('\n')
    absDiff = float(output[-4].split(' ')[1])
    relDiff = float(output[-3].split(' ')[1])
    assert relDiff < relThres

def test_mpi_comm():
    case_path = os.path.join(cases_path, 'forwardStep')
    mesh = Mesh.create(case_path)
    IOField.setMesh(mesh)
    with IOField.handle(0.):
        U = IOField.read('U')
        U.partialComplete()
    U.field = np.random.rand(*U.field.shape)
    with IOField.handle(1.):
        U.write()
    time.sleep(1)

    subprocess.check_output(['decomposePar', '-case', case_path, '-time', '1'])

    subprocess.check_output(['mpirun', '-np', '4', 'python2', __file__, 'RUN', 'test_mpi_comm_method', case_path])

    try:
        checkFields(case_path, 'U', '1.0', '2.0', relThres=1e-12, nProcs=4)
    finally:
        shutil.rmtree(os.path.join(case_path, '1'))
        map(shutil.rmtree, glob.glob(os.path.join(case_path, 'processor*')))

def test_mpi():
    solver = os.path.join(apps_path, 'problem.py')
    case = os.path.join('../templates/', 'forwardStep_test')
    endTime = '0.001'
    subprocess.check_output([solver, case, '-c'])
    case_path = os.path.join(cases_path, 'forwardStep')
    endTime = '{0:.11f}'.format(float(endTime))
    subprocess.check_output(['mv', os.path.join(case_path, endTime), os.path.join(case_path, '10')])

    subprocess.check_output(['decomposePar', '-case', case_path, '-time', '0'])
    #subprocess.check_output([os.path.join(scripts_path, 'decompose.py'), case_path, '4', '0.0'])
    for pkl in glob.glob(os.path.join(case_path, '*.pkl')):
        shutil.copy(pkl, os.path.join(case_path, 'processor0'))
    subprocess.check_output(['mpirun', '-np', '4', solver, case])
    subprocess.check_output(['reconstructPar', '-case', case_path, '-time', endTime])

    try:
        checkFields(case_path, 'U', endTime, '10.0')
    finally:
        shutil.rmtree(os.path.join(case_path, endTime))
        shutil.rmtree(os.path.join(case_path, '10'))
        map(shutil.rmtree, glob.glob(os.path.join(case_path, 'processor*')))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'RUN':
        func = locals()[sys.argv[2]]
        func(*sys.argv[3:])
    else:
        #test_mpi_comm()
        test_mpi()
