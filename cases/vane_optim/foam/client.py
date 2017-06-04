from subprocess import check_call, STDOUT
from subprocess import check_output
import os
import pickle

def get_mesh(param, work_dir):
    server = 'kolmogorov.mit.edu'
    base = '/home/talnikar/adFVM/cases/vane_optim/foam/laminar/'
    case = base + os.path.basename(work_dir) + '/'
    mesh_log = work_dir + 'mesh.log'
    with open(work_dir + 'params.pkl', 'w') as f:
        pickle.dump([param, base, case], f)
    #configure timeouts
    #copy profile to server
    try:
        check_output(['scp', work_dir + 'params.pkl', '{0}:{1}'.format(server, case)])
    except:
        raise Exception('Could not copy profile to server')

    #run mesh generator
    try:
        ret = check_output(['ssh', server, '\"python {}/test.py {}\"'.format(base, case + 'params.pkl')], stderr=STDOUT)
        write_log(mesh_log, ret)
    except:
        raise Exception('Could not run mesh generator')
    
    #copy the mesh
    try:
        for f in ['mesh.hdf5', '3.hdf5']:
            check_output(['scp', '{0}:{1}'.format(server, case + f), work_dir + f])
    except:
        raise Exception('Could not copy mesh from server')
    #delete mesh for future runs
    #try:
    #    check_call(['ssh', server, 'rm', mesh_dir + mesh_file])
    #except:
    #    raise Exception('Could not delete mesh from server')

def write_log(mesh_log, output):
    f = open(mesh_log, 'a')
    f.write(output)
    f.close()

