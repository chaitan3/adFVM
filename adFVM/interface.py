import os
import sys
import subprocess
import numpy as np
import shutil
import glob

from .mesh import Mesh
import h5py

def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class Runner(object):
    fieldNames = ['rho', 'rhoU', 'rhoE']
    reference = [1., 200., 2e5]
    program = os.path.expanduser('~') + '/adFVM/apps/problem.py'

    def __init__(self):
        pass

    def copyCase(self, case):
        os.makedirs(case)
        shutil.copy(self.base + 'mesh.hdf5', case)
        shutil.copytree(self.base + 'gencode', case + 'gencode')
        return

    def randomFields(self):
        return np.random.randn(self.fieldsShape)

    def spawnJob(self, args, **kwargs):
        return subprocess.call(['mpirun', '-np', str(self.nProcs)] + args, **kwargs)

    def spawnSlurmJob(exe, args, **kwargs):
        from fds.slurm import grab_from_SLURM_NODELIST
        interprocess = kwargs['interprocess']
        del kwargs['interprocess']
        nodes = grab_from_SLURM_NODELIST(1, interprocess)
        print('spawnJob', nodes, exe, args)
        returncode = subprocess.call(['mpirun', '--host', ','.join(nodes.grabbed_nodes)
                           , exe] + args, **kwargs)
        nodes.release()
        #returncode = subprocess.call(['mpirun', '-np', str(nProcessors), exe] + args, **kwargs)
        return returncode

class SerialRunner(Runner):
    def __init__(self, base, time, problem, nProcs=1):
        self.base = base
        self.time = time
        self.problem = problem
        self.stime = Mesh.getTimeString(time)
        self.nProcs = nProcs
        self.internalCells = self.getInternalCells(base)
        self.fieldsShape = len(self.internalCells)*5

    def getInternalCells(self, case):
        internalCells = []
        with h5py.File(case + 'mesh.hdf5', 'r') as mesh:
            nCount = mesh['parallel/end'][:]-mesh['parallel/start'][:]
            nInternalCells = nCount[:,4]
            nGhostCells = nCount[:,2]-nCount[:,3]
            start = 0
            for i in range(0, self.nProcs):
                n = nInternalCells[i] 
                internalCells.append(np.arange(start, start + n))
                start += n + nGhostCells[i]
        return np.concatenate(internalCells)

    def readFields(self, case, time):
        fields = []
        with h5py.File(case + Mesh.getTimeString(time) + '.hdf5', 'r') as phi:
            for name in Runner.fieldNames:
                fields.append(phi[name + '/field'][:][self.internalCells])
        fields = [x/y for x, y in zip(fields, Runner.reference)]
        return np.hstack(fields).ravel()

    def writeFields(self, fields, case, time):
        fields = fields.reshape((fields.shape[0]/5, 5))
        fields = fields[:,[0]], fields[:,1:4], fields[:,[4]]
        fields = [x*y for x, y in zip(fields, Runner.reference)]
        timeFile = case + Mesh.getTimeString(time) + '.hdf5' 
        shutil.copy(self.base + self.stime + '.hdf5', timeFile)
        with h5py.File(timeFile, 'r+') as phi:
            for index, name in enumerate(Runner.fieldNames):
                field = phi[name + '/field'][:]
                field[self.internalCells] = fields[index]
                phi[name + '/field'][:] = field
        return

    def runCase(self, initFields, (parameter, nSteps), case):
        # write initial field
        self.writeFields(initFields, case, self.time)

        # modify problem file
        problemFile = case + os.path.basename(self.problem)
        with open(self.problem, 'r') as f:
            lines = f.readlines()
        with open(problemFile, 'w') as f:
            for line in lines:
                writeLine = line.replace('NSTEPS', str(nSteps))
                writeLine = writeLine.replace('STARTTIME', str(self.time))
                writeLine = writeLine.replace('PARAMETER', str(parameter))
                f.write(writeLine)

        outputFile = case  + 'output.log'
        with open(outputFile, 'w') as f:
            returncode = self.spawnJob([Runner.program, problemFile], stdout=f, stderr=f, cwd=case)
        if returncode:
            raise Exception('Execution failed, check error log in :', case)
        objectiveSeries = np.loadtxt(case + 'timeSeries.txt')

        # read final fields
        times = [float(x[:-5]) for x in os.listdir(case) if isfloat(x[:-5]) and x.endswith('.hdf5')]
        lastTime = sorted(times)[-1]
        finalFields = self.readFields(case, lastTime)
        # read objective values
        return finalFields, objectiveSeries 

class ParallelRunner(Runner):
    def __init__(self):
        raise NotImplemented

    def getParallelInfo():
        import h5py
        from mpi4py import MPI
        mpi = MPI.COMM_WORLD
        rank = mpi.rank 
        #print rank, mpi.Get_size(), MPI.Get_processor_name()
        sys.stdout.flush()
        
        with h5py.File(case + 'mesh.hdf5', 'r', driver='mpio', comm=mpi) as mesh:
            nCount = mesh['parallel/end'][rank]-mesh['parallel/start'][rank]
            nInternalCells = nCount[4]
            nGhostCells = nCount[2]-nCount[3]
            nCells = nInternalCells + nGhostCells
            cellStart = mpi.exscan(nCells)
            if cellStart == None:
                cellStart = 0
            cellEnd = cellStart + nInternalCells

        size = nInternalCells*5
        start = mpi.exscan(size)
        end = mpi.scan(size)
        size = mpi.bcast(end, root=nProcessors-1)
        return cellStart, cellEnd, start, end, size, mpi

    def readFields(self, case, time, fieldFile):
        import h5py
        time = float(time)
        cellStart, cellEnd, start, end, size, mpi = getParallelInfo()
        fields = []
        with h5py.File(case + getTime(time) + '.hdf5', 'r', driver='mpio', comm=mpi) as phi:
            for name in fieldNames:
                fields.append(phi[name + '/field'][cellStart:cellEnd])
        fields = [x/y for x, y in zip(fields, reference)]
        field = np.hstack(fields).ravel()
        with h5py.File(fieldFile, 'w', driver='mpio', comm=mpi) as handle:
            fieldData = handle.create_dataset('field', shape=(size,), dtype=field.dtype)
            fieldData[start:end] = field
        return

    def writeFields(self, fieldFile, caseDir, ntime):
        import h5py
        ntime = float(ntime)
        cellStart, cellEnd, start, end, size, mpi = getParallelInfo()
        with h5py.File(fieldFile, 'r', driver='mpio', comm=mpi) as handle:
            fields = handle['field'][start:end]
        fields = fields.reshape((fields.shape[0]/5, 5))
        fields = fields[:,[0]], fields[:,1:4], fields[:,[4]]
        fields = [x*y for x, y in zip(fields, reference)]
        timeFile = caseDir + getTime(ntime) + '.hdf5' 
        with h5py.File(timeFile, 'r+', driver='mpio', comm=mpi) as phi:
            for index, name in enumerate(fieldNames):
                field = phi[name + '/field']
                field[cellStart:cellEnd] = fields[index]
                phi[name + '/field'][:] = field
        return


    def runCase(initFields, parameter, nSteps, run_id, interprocess):
        #cobalt.interprocess = interprocess

        # generate case folders
        caseDir = getHostDir(run_id)
        if not os.path.exists(caseDir):
            os.makedirs(caseDir)
        shutil.copy(case + 'mesh.hdf5', caseDir)
        for pkl in glob.glob(case + '*.pkl'):
            shutil.copy(pkl, caseDir)
        timeFile = caseDir + stime + '.hdf5' 
        shutil.copy(case + stime + '.hdf5', timeFile)
        
        # write initial field
        outputFile = caseDir  + 'writeFields.log'
        with open(outputFile, 'w') as f:
            if spawnJob(python, [fileName, 'RUN', 'writeFields', initFields, caseDir, str(time)], stdout=f, stderr=f, interprocess=interprocess):
                raise Exception('initial field conversion failed')
        print('initial field written', initFields)

        # modify problem file
        shutil.copy(case + problem, caseDir)
        problemFile = caseDir + problem
        with open(problemFile, 'r') as f:
            lines = f.readlines()
        with open(problemFile, 'w') as f:
            for line in lines:
                writeLine = line.replace('NSTEPS', str(nSteps))
                writeLine = writeLine.replace('STARTTIME', str(time))
                writeLine = writeLine.replace('CASEDIR', '\'{}\''.format(caseDir))
                writeLine = writeLine.replace('PARAMETER', str(parameter))
                f.write(writeLine)

        outputFile = caseDir  + 'output.log'
        errorFile = caseDir  + 'error.log'
        with open(outputFile, 'w') as f, open(errorFile, 'w') as fe:
            #if spawnJob(python, [problemFile], stdout=f, stderr=f):
            if spawnJob(python, [program, problemFile, '--coresPerNode', str(nProcsPerNode)], stdout=f, stderr=fe, interprocess=interprocess):
                raise Exception('Execution failed, check error log:', outputFile)
        print('execution finished', caseDir)

        # read final fields
        times = [float(x[:-5]) for x in os.listdir(caseDir) if isfloat(x[:-5]) and x.endswith('.hdf5')]
        lastTime = sorted(times)[-1]
        finalFields = caseDir + 'output.h5'
        outputFile = caseDir  + 'getInternalFields.log'
        with open(outputFile, 'w') as f:
            if spawnJob(python, [fileName, 'RUN', 'getInternalFields', caseDir, str(lastTime), finalFields], stdout=f, stderr=f, interprocess=interprocess):
                raise Exception('final field conversion failed')
        print('final field written', finalFields)

        # read objective values
        objectiveSeries = np.loadtxt(caseDir + 'timeSeries.txt')
        print caseDir

        #cobalt.interprocess = None
        return finalFields, objectiveSeries
