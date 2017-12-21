import numpy as np
import os
import subprocess


test_path = os.path.dirname(__file__)
adFVM_path = os.path.join(test_path, '..')
scripts_path = os.path.join(adFVM_path, 'scripts')
apps_path = os.path.join(adFVM_path, 'apps')
cases_path = os.path.join(adFVM_path, 'cases')
templates_path = os.path.join(adFVM_path, 'templates')

def evaluate(output, inputs, value, self):
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(value, list):
        value = [value]
    f = SolverFunction(inputs, output, self, 'test', BCs=False, source=False)
    return f(*value)

def checkArray(self, res, ref, maxThres=1e-7):#, sumThres=1e-4):
    self.assertEqual(res.shape, ref.shape)
    diff = np.abs(res-ref)
    self.assertAlmostEqual(0, diff.max(), delta=maxThres)
    #self.assertAlmostEqual(0, diff.sum(), delta=sumThres)

def checkVolSum(self, res, ref, relThres=1e-4, mesh=None):
    if not mesh:
        mesh = self.mesh
    vols = mesh.origMesh.volumes
    if len(res.shape) == 3:
        vols = vols.flatten().reshape((-1,1,1))
    diffV = np.abs(res-ref)*vols
    refV = np.abs(ref)*vols
    rel = diffV.sum()/refV.sum()
    self.assertAlmostEqual(0, rel, delta=relThres)


def checkFields(self, case, field, time1, time2, relThres=1e-6, nProcs=1):
    diff = os.path.join(scripts_path, 'field', 'diff_fields.py')
    if nProcs == 1:
        output = subprocess.check_output([diff, case, field, time1, time2])
    else:
        output = subprocess.check_output(['mpirun', '-np', str(nProcs), diff, case, field, time1, time2])
    output = output.split('\n')
    absDiff = float(output[-4].split(' ')[1])
    relDiff = float(output[-3].split(' ')[1])
    self.assertAlmostEqual(0, relDiff, delta=relThres)
