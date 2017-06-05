import numpy as np
from scipy import interpolate, spatial, integrate
import matplotlib.pyplot as plt

from cStringIO import StringIO
import cPickle as pickle
import copy
import os, sys, shutil
import glob

from adFVM.mesh import writeField
from adFVM import config

foam_dir = os.environ['FOAM_APPBIN'] + '/'
scripts_dir = '/home/talnikar/adFVM/scripts/'

def fit_bspline(coords, mesh):

    t, u = interpolate.splprep(coords, s=0)
    tev = np.linspace(0, 1, 10000)
    x = interpolate.splev(tev, t)
    dist = spatial.distance.cdist(np.array(x).T, mesh)
    indices = dist.argmin(axis=0)
    tn = tev[indices]
    #plt.plot(x[0][indices], x[1][indices], 'b.')
    #plt.plot(coords[0], coords[1], 'r.')
    #plt.axis('scaled')
    return t, tn

c_index = [-1, -4]
c_scale = [1e-3, 4e-3]
c_bound = [0.5, 0.5]
c_count = len(c_index)

def perturb_bspline(param, c, t, index):
    n_param = len(param)/2
    param_index = param[index*n_param:(index+1)*n_param]
    per = np.array(param_index).reshape(c_count, 2)
    n = 0
    for i, s, b in zip(c_index, c_scale, c_bound):
        coord0 = t[1][0][i]
        coord1 = t[1][1][i]
        factor = b*np.exp((-(coord0-c[1][0])**2
                        -(coord1-c[1][1])**2)
                        /s**2)
        c[1][0] += per[n,0]*factor
        c[1][1] += per[n,1]*factor
        n += 1
    return c[1][0][c_index], c[1][1][c_index]

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

    #with open(base + '../vane_coords_baseline.txt', 'r') as f:
    #    v1 = []
    #    v2 = []
    #    n = int(f.readline().split(' ')[0])
    #    for l in f.readlines():
    #        t = l.split()
    #        v1.append(t[0])
    #        v2.append(t[1])
    #v1 = np.array(v1, dtype=float)/1000
    #v2 = np.array(v2, dtype=float)/1000
    #suction2 = [v1[:n], v2[:n]]
    #pressure2 = [v1[n:], v2[n:]]

    # transformation
    def transform(coords, rev=False):
        t1 = 0.036461,-0.052305
        t2 = 0.0356125, -0.0494455
        theta = np.arctan((t2[0]-t1[0])/(t2[1]-t1[1]))
        if rev:
            theta = -theta
        tcoords = np.array(coords)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t1 = np.array(t1).reshape(-1,1)
        if rev:
            return np.matmul(rot, tcoords) + t1
        else:
            return np.matmul(rot, tcoords-t1)

    tsuction = transform(suction)
    tpressure = transform(pressure)
    #plt.scatter(tsuction[0], tsuction[1])
    #plt.scatter(tpressure[0], tpressure[1])
    #plt.axis('scaled')
    #plt.show()

    points = np.hstack((tpressure[:,-5:-1], tsuction[:,-5:][:,::-1]))
    t = interpolate.splrep(points[0], points[1], k=4, s=5e-7)
    x = np.linspace(points[0,0], points[0,-1], 10000)
    y = interpolate.splev(x, t)
    #plt.plot(x, y)
    #plt.scatter(points[0], points[1])
    #plt.axis('scaled')
    #plt.show()
    ts = []
    ys = []
    for i in range(1, 5):
        per_points = np.loadtxt('../vane_coords_l{}.txt'.format(i))
        points2 = np.hstack((tpressure[:,-5:-2], per_points.T, tsuction[:,-5:-2][:,::-1]))
        ts.append(interpolate.splrep(points2[0], points2[1], k=4, s=1e-6))
        ys.append(interpolate.splev(x, ts[-1]))
        #plt.plot(x, ys[-1])
        #plt.scatter(points2[0], points2[1])
        #plt.axis('scaled')
        #plt.show()
    #plt.show()


    #ti = interpolate.splrep(x, yd)
    #tn = interpolate.splantider(ti)
    #yn = points[1, 0] + interpolate.splev(x, tn)
    #tn, u = interpolate.splprep([x, yn])
    #yn = interpolate.splev(np.linspace(0, 1, 1000), tn)
    #plt.plot(yn[0], yn[1])
    #yn = y[0] + integrate.cumtrapz(yd, x, initial=0)
    yn = (1-sum(param))*y + sum([param[i]*ys[i] for i in range(0,4)])
    tn = interpolate.splrep(x, yn)
    plt.plot(x, yn)
    plt.plot(x, y)
    plt.axis('scaled')
    plt.show()

    try:
        shutil.rmtree(case + '1')
    except:
        pass
    shutil.copytree(case + '0', case + '1')

    config.fileFormat = 'ascii'

    repl = {}
    #import pdb;pdb.set_trace()
    for coords, patch in [(suction, 'suction'), (pressure, 'pressure')]:
    #for coords, coords2, patch in [(suction, suction2, 'suction'), (pressure, pressure2, 'pressure')]:
        meshPoints = np.loadtxt(case + '{}_points.txt'.format(patch))
        newPoints = meshPoints[:,[0,1]].copy()
        indices = np.logical_and(meshPoints[:,0] > coords[0][-5], meshPoints[:,1] < coords[1][-5])

        tpoints = transform(meshPoints[indices][:,[0,1]].T)
        data = np.vstack((x, y)).T
        dist = np.cumsum(np.sqrt(((data[1:]-data[:-1])**2).sum(axis=1)))
        p1 = dist/dist[-1]
        ind  = spatial.distance.cdist(data, tpoints.T).argmin(axis=0)
        p1 = p1[ind]
        data = np.vstack((x, yn)).T
        dist = np.cumsum(np.sqrt(((data[1:]-data[:-1])**2).sum(axis=1)))
        p2 = dist/dist[-1]

        p = np.abs(p2[:,None]-p1[None,:])
        pi = np.argmin(p, axis=0)
        xn = x[pi]
        ynp = interpolate.splev(xn, tn)
        newPoints[indices] = transform((xn, ynp), rev=True).T

        pointDisplacement = np.vstack((newPoints[:,0]-meshPoints[:,0], newPoints[:,1]-meshPoints[:,1], 0*meshPoints[:,2])).T
        pointDisplacement = np.ascontiguousarray(pointDisplacement)
        handle = StringIO()
        writeField(handle, pointDisplacement, 'vector', 'value')
        repl[patch] = handle.getvalue()
    
    plt.axis('scaled')
    plt.axis('off')
    plt.xlim([0.026,0.040])
    plt.ylim([-0.055,-0.040])
    plt.legend()
    np.savetxt(case + 'params.txt', param)
    #plt.savefig(case + 'perturbation.png')
    import re
    #num = re.search('param[0-9]+', case).group(0)
    num = 'test'
    plt.savefig(base + 'perturbation_{}.png'.format(num))
    #plt.show()

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
    spawn_job([foam_dir + 'createPatch', '-overwrite', '-case', case])
    spawn_job([foam_dir + 'extrudeMesh'], cwd=case)
    spawn_job([foam_dir + 'transformPoints', '-translate', '(0 0 -0.01)', '-case', case])
    shutil.copyfile(case + 'system/createPatchDict.cyclic', case + 'system/createPatchDict')
    spawn_job([foam_dir + 'createPatch', '-overwrite', '-case', case])
    map(os.remove, glob.glob('*.obj'))
        
def perturb_mesh(base, case, spawn_job):
    # serial 
    spawn_job([foam_dir + 'moveMesh', '-case', case])
    shutil.move(case + '1.0001/polyMesh/points', case + 'constant/polyMesh/points')
    extrude_mesh(case, spawn_job)
    time = '3.0'
    spawn_job([scripts_dir + 'field/map_fields.py', base + '3d_baseline/', case, time, time])
    spawn_job([foam_dir + 'decomposePar', '-time', time, '-case', case])
    # convert to hdf5
    spawn_job([scripts_dir + 'conversion/hdf5serial.py', case, time])
    spawn_job([scripts_dir + 'conversion/hdf5swap.py', case + 'mesh.hdf5', case + '3.hdf5'])

    # if done this way, no mapping and hdf5 conversion needed
    # serial for laminar
    #spawn_job([foam_dir + 'moveMesh', '-case', case])
    # parallel
    #spawn_job([foam_dir + 'moveMesh', '-case', case, '-parallel'])
    #spawn_job([sys.executable, os.path.join(scripts_dir, 'conversion', 'hdf5mesh.py'), case, '1.0001'])
    return 

def gen_mesh_param(param, base, case, spawn_job, perturb=True):
    case = case +'/'
    shutil.copytree(base + 'constant', case + 'constant')
    shutil.copytree(base + '0', case + '0')
    shutil.copytree(base + 'system', case + 'system')
    for points in glob.glob(base + '*_points.txt'):
        shutil.copyfile(points, case + os.path.basename(points))
    for folder in glob.glob(base + 'processor*'):
        shutil.copytree(folder, case + os.path.basename(folder))

    create_displacement(param, base, case)
    #with open('params.pkl', 'w') as f:
    #    pickle.dump([param, base, case], f)
    #spawn_job([sys.executable, __file__, 'create_displacement', 'params.pkl'])
    if perturb:
        perturb_mesh(base, case, spawn_job)

    return

if __name__ == '__main__':
    func = locals()[sys.argv[1]]
    paramsFile = sys.argv[2]
    with open(paramsFile) as f:
        params = pickle.load(f)
    func(*params)
