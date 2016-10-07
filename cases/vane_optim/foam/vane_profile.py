import numpy as np
from scipy import interpolate, spatial
import matplotlib.pyplot as plt

from cStringIO import StringIO
import copy
import os, sys, shutil
import glob

from adFVM.mesh import writeField
from adFVM import config

foam_dir = os.environ['FOAM_APPBIN'] + '/'
apps_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
scripts_dir = os.path.join(apps_dir, '..', 'scripts')

def fit_bspline(coords, mesh):
    t, u = interpolate.splprep(coords, s=0)
    tev = np.linspace(0, 1, 10000)
    x = interpolate.splev(tev, t)
    dist = spatial.distance.cdist(np.array(x).T, mesh)
    indices = dist.argmin(axis=0)
    tn = tev[indices]
    #plt.plot(x[0][indices], x[1][indices], 'b.')
    #plt.plot(coords[0], coords[1], 'r.')
    #plt.plot(mesh[:,0], mesh[:,1], 'k.')
    #plt.axis('scaled')
    return t, tn

c_index = [-2, -4]
c_scale = [2e-3, 8e-3]
c_count = len(c_index)
def perturb_bspline(param, index, t, tn):
    n_param = len(param)/2
    param = param[index*n_param:(index+1)*n_param]
    per = np.array(param).reshape(c_count, 2)
    c = copy.deepcopy(t)
    x = interpolate.splev(tn, t)
    plt.plot(x[0], x[1], 'k.', label='orig')
    n = 0
    for i, s in zip(c_index, c_scale):
        coord0 = c[1][0][i]
        coord1 = c[1][1][i]
        factor = np.exp((-(coord0-c[1][0])**2
                        -(coord1-c[1][1])**2)
                        /s**2)
        c[1][0] += per[n,0]*factor
        c[1][1] += per[n,1]*factor
        n += 1
    x = interpolate.splev(tn, c)
    plt.plot(x[0], x[1], 'b.', label='perturbed')
    return x

def create_displacement(param, base, case):
    with open(base + 'vane_coords.txt', 'r') as f:
        v1 = []
        v2 = []
        n = int(f.readline().split(' ')[0])
        for l in f.readlines():
            t = l.split()
            v1.append(t[0])
            v2.append(t[1])
    v1 = np.array(v1, dtype=float)/1000
    v2 = np.array(v2, dtype=float)/1000
    suction = [v1[:n], v2[:n]]
    pressure = [v1[n:], v2[n:]]

    #for prof in [suction, pressure]:
    #    plt.scatter(prof[0], prof[1])
    #    for i in range(0, len(prof[0])):
    #        plt.annotate(str(i), (prof[0][i], prof[1][i]))
    #plt.axis('scaled')
    #plt.show()

    try:
        shutil.rmtree(case + '1')
    except:
        pass
    shutil.copytree(case + '0', case + '1')

    config.fileFormat = 'ascii'

    repl = {}
    index = 0
    for coords, patch in [(suction, 'suction'), (pressure, 'pressure')]:
        points = np.loadtxt(base + '{}_points.txt'.format(patch))
        t, tn = fit_bspline(coords, points[:,[0,1]])
        newPoints = perturb_bspline(param, index, t, tn)
        pointDisplacement = np.vstack((newPoints[0]-points[:,0], newPoints[1]-points[:,1], 0*points[:,2])).T
        handle = StringIO()
        writeField(handle, pointDisplacement, 'vector', 'value')
        repl[patch] = handle.getvalue()
        index += 1

    plt.axis('scaled')
    plt.legend()
    plt.show()
    #plt.savefig(case + 'perturbation.png')

    dispFile = case + '1/pointDisplacement'
    with open(dispFile) as f:
        data = f.readlines()
    for index, line in enumerate(data):
        data[index] = line.replace('PRESSURE', repl['pressure'])
        data[index] = data[index].replace('SUCTION', repl['suction'])
    with open(dispFile, 'w') as f:
        for line in data:
            f.write(line)
    return

def extrude_mesh(case, spawn_job):
    shutil.copyfile(case + 'system/createPatchDict.patch', case + 'system/createPatchDict')
    spawn_job([foam_dir + 'createPatch', '-overwrite', '-case', case], False)
    spawn_job([foam_dir + 'extrudeMesh'], False, cwd=case)
    spawn_job([foam_dir + 'transformPoints', '-translate', '"(0 0 -0.1)"', '-case', case], False)
    shutil.copyfile(case + 'system/createPatchDict.cyclic', case + 'system/createPatchDict')
    spawn_job([foam_dir + 'createPatch', '-overwrite', '-case', case], False)
    spawn_job([foam_dir + 'decomposePar', '-time', 'constant', '-case', case], False)
        
def perturb_mesh(case, spawn_job):
    # serial 
    spawn_job([foam_dir + 'moveDynamicMesh', '-case', case], False)
    shutil.move(case + '1.0001/polyMesh/points', case + 'constant/polyMesh/points')
    extrude_mesh(case, spawn_job)
    # direct parallel?
    spawn_job([sys.executable, os.path.join(scripts_dir, 'conversion', 'hdf5.py'), case, 'None'])
    return

def gen_mesh_param(param, base, case, spawn_job):
    case = case +'/'
    shutil.copytree(base + 'constant', case + 'constant')
    shutil.copytree(base + '0', case + '0')
    shutil.copytree(base + 'system', case + 'system')
    create_displacement(param, base, case)
    perturb_mesh(case, spawn_job)
    map(os.remove, glob.glob('*.obj'))
    return

if __name__ == '__main__':
    import subprocess
    param = [+1e-4,-4e-4, 0, 0, -1e-4, -4e-4, 0, 0]
    gen_mesh_param(param, './', './param0', subprocess.check_call)
