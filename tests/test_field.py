from __future__ import print_function
import copy
import shutil
import glob
import os
import sys
import numpy as np
import subprocess
import pytest

from adFVM import config
from adFVM.field import Field, CellField, IOField
from adFVM.mesh import Mesh
from deep_eq import deep_eq

def test_field():
    case = '../cases/convection/'
    mesh = Mesh.create(case)
    Field.setMesh(mesh)
    n = mesh.nFaces

    Ur = np.random.rand(n, 1)
    Vr = np.random.rand(n, 1) + 2
    
    U = Field('U', Ur, (1,))
    V = Field('V', Vr, (1,))

    Wr = (Ur + Vr/Ur)*Vr**0.5
    W = (U + V/U)*V**0.5
    assert np.allclose(W.field, Wr)


@pytest.mark.skip
def test_field_io(case, hdf5):
    config.hdf5 = hdf5
    mesh = Mesh.create(case)
    Field.setMesh(mesh)

    time = 1.0

    field = np.random.rand(mesh.nInternalCells, 3)
    boundary = copy.deepcopy(mesh.defaultBoundary)
    nFaces = mesh.getPatchFaceRange('inlet')[2]
    boundary['inlet'] = {
            'type':'CBC_TOTAL_PT',
            'pt': 'uniform 20',
            'value': np.random.rand(nFaces, 3)
            }
    U = IOField('U', field, (3,), boundary)
    U.partialComplete()

    field = np.random.rand(mesh.nInternalCells, 1)
    boundary = copy.deepcopy(mesh.defaultBoundary)
    T = IOField('T', field, (1,), boundary)
    boundary['outlet'] = {
            'type':'fixedValue',
            'value': 'uniform 10'
            }

    with IOField.handle(time):
        U.write()
        T.write()

    with IOField.handle(time):
        Tn = IOField.read('T')
        Un = IOField.read('U')
        Un.partialComplete()

    config.hdf5 = False
    assert (deep_eq(Tn.boundary, T.boundary))
    assert (deep_eq(Un.boundary, U.boundary))
    np.allclose(Tn.field, T.field)
    np.allclose(Un.field, U.field)

@pytest.mark.skip
def test_field_io_mpi(case):
    mesh = Mesh.create(case)
    Field.setMesh(mesh)

    time = 1.0

    config.hdf5 = False
    with IOField.handle(time):
        U = IOField.read('U')
        U.partialComplete()

    config.hdf5 = True
    with IOField.handle(time, case=case):
        Uh = IOField.read('U')
        Uh.partialComplete()

    assert np.allclose(U.field, Uh.field)
    assert deep_eq(U.boundary, Uh.boundary)

def test_foam():
    case = '../cases/forwardStep/'
    try:
        test_field_io(case, False)
    finally:
        shutil.rmtree(os.path.join(case, '1'))

def test_hdf5():
    case = '../cases/forwardStep/'
    try:
        subprocess.check_output(['../scripts/conversion/hdf5.py', case])
        test_field_io(case, True)
    finally:
        map(os.remove, glob.glob(os.path.join(case, '*.hdf5')))

def test_hdf5_mpi():
    case = '../cases/forwardStep/'
    try:
        mesh = Mesh.create(case)
        IOField.setMesh(mesh)
        time = 1.0
        field = np.random.rand(mesh.nInternalCells, 3)
        boundary = copy.deepcopy(mesh.defaultBoundary)
        U = IOField('U', field, (3,), boundary)
        with IOField.handle(time):
            U.write()

        subprocess.check_output(['decomposePar', '-time', '1', '-case', case])
        subprocess.check_output(['mpirun', '-np', '4', '../scripts/conversion/hdf5.py', case, str(time)])
        subprocess.check_output(['mpirun', '-np', '4', sys.executable, __file__, 'RUN', 'test_field_io_mpi', case])
    finally:
        try:
            shutil.rmtree(os.path.join(case, '1'))
        except OSError:
            pass
        map(os.remove, glob.glob(os.path.join(case, '*.hdf5')))
        map(shutil.rmtree, glob.glob(os.path.join(case, 'processor*')))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'RUN':
        func = locals()[sys.argv[2]]
        func(*sys.argv[3:])
    else:
        #test_field()
        #test_foam()
        #test_hdf5()
        test_hdf5_mpi()
