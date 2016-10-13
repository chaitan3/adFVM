import numpy as np
from scipy import interpolate, spatial
import matplotlib.pyplot as plt

from cStringIO import StringIO
import cPickle as pickle
import copy
import os, sys, shutil
import glob

from adFVM.mesh import writeField
from adFVM import config

foam_dir = os.environ['FOAM_APPBIN'] + '/'
apps_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
scripts_dir = os.path.join(apps_dir, '..', 'scripts')
#scripts_dir = os.path.join('/home/talnikar/adFVM/scripts/')

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
c_scale = [1e-3, 4e-3]
c_count = len(c_index)

def perturb_bspline(param, c, t, index):
    n_param = len(param)/2
    param_index = param[index*n_param:(index+1)*n_param]
    per = np.array(param_index).reshape(c_count, 2)
    n = 0
    for i, s in zip(c_index, c_scale):
        coord0 = t[1][0][i]
        coord1 = t[1][1][i]
        factor = np.exp((-(coord0-c[1][0])**2
                        -(coord1-c[1][1])**2)
                        /s**2)
        c[1][0] += per[n,0]*factor
        c[1][1] += per[n,1]*factor
        n += 1
    return 

def create_displacement(param, base, case):
    from mpi4py import MPI
    mpi = MPI.COMM_WORLD
    rank = mpi.rank
    if mpi.Get_size() > 1:
        case = case + 'processor{}/'.format(rank)

    if rank == 0:
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
    else:
        suction = pressure = None
    suction = mpi.bcast(suction, root=0)
    pressure = mpi.bcast(pressure, root=0)

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

    #config.fileFormat = 'ascii'

    repl = {}
    index = 0
    t = []
    spline_points = []
    spline_coeffs = []
    for coords, patch in [(suction, 'suction'), (pressure, 'pressure')]:
        points = np.loadtxt(case + '{}_points.txt'.format(patch))
        if points.shape[0] == 0:
            points = points.reshape((0, 3))
        t, tn = fit_bspline(coords, points[:,[0,1]])
        spline_points.append(points)
        spline_coeffs.append((t, tn))

    for index, patch in enumerate(['suction', 'pressure']):
        points = spline_points[index]
        t, tn = spline_coeffs[index]
        #if rank == 0:
        #    plt.plot(points[:,0], points[:,1], 'k.', label='orig')
        if points.shape[0] > 0:
            c = copy.deepcopy(t)
            perturb_bspline(param, c, spline_coeffs[0][0], 0)
            perturb_bspline(param, c, spline_coeffs[1][0], 1)
            newPoints = interpolate.splev(tn, c)
                #if rank == 0:
            #    plt.plot(newPoints[0], newPoints[1], 'b.', label='perturbed')
        else:
            newPoints = [np.zeros((0.,)), np.zeros((0.,))]
        pointDisplacement = np.vstack((newPoints[0]-points[:,0], newPoints[1]-points[:,1], 0*points[:,2])).T
        pointDisplacement = np.ascontiguousarray(pointDisplacement)
        handle = StringIO()
        writeField(handle, pointDisplacement, 'vector', 'value')
        repl[patch] = handle.getvalue()
        index += 1
    
    if rank == 0:
    #    plt.axis('scaled')
    #    plt.xlim([0.026,0.040])
    #    plt.ylim([-0.055,-0.030])
    #    plt.legend()
    #    #plt.show()
        np.savetxt(case + 'params.txt', param)
    #    plt.savefig(case + 'perturbation.png')

    dispFile = case + '1/pointDisplacement'
    with open(dispFile) as f:
        data = f.readlines()
    for index, line in enumerate(data):
        if 'value' in line and 'uniform' in line:
            if '111' in line:
                data[index] =  repl['pressure']
            elif '222' in line:
                data[index] =  repl['suction']
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
    #spawn_job([foam_dir + 'moveMesh', '-case', case], False)
    #shutil.move(case + '1.0001/polyMesh/points', case + 'constant/polyMesh/points')
    #extrude_mesh(case, spawn_job)
    # parallel
    #spawn_job([foam_dir + 'moveMesh', '-case', case])
    spawn_job([foam_dir + 'moveMesh', '-case', case, '-parallel'])

    spawn_job([sys.executable, os.path.join(scripts_dir, 'conversion', 'hdf5mesh.py'), case, '1.0001'])
    return

def gen_mesh_param(param, base, case, spawn_job):
    case = case +'/'
    shutil.copytree(base + 'constant', case + 'constant')
    shutil.copytree(base + '0', case + '0')
    shutil.copytree(base + 'system', case + 'system')
    for points in glob.glob(base + '*_points.txt'):
        shutil.copyfile(points, case + os.path.basename(points))
    for folder in glob.glob(base + 'processor*'):
        shutil.copytree(folder, case + os.path.basename(folder))

    with open('params.pkl', 'w') as f:
        pickle.dump([param, base, case], f)
    #create_displacement(param, base, case)
    spawn_job([sys.executable, __file__, 'create_displacement', 'params.pkl'])
    perturb_mesh(case, spawn_job)

    map(os.remove, glob.glob('*.obj'))
    return

if __name__ == '__main__':
    func = locals()[sys.argv[1]]
    paramsFile = sys.argv[2]
    with open(paramsFile) as f:
        params = pickle.load(f)
    func(*params)
