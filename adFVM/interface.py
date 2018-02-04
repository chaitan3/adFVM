#!/usr/bin/python2 -u
import os
import sys
import subprocess
import numpy as np
import shutil
import glob

def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

python = sys.executable

fieldNames = ['rho', 'rhoU', 'rhoE']
program = source + 'adFVM/apps/problem.py'
reference = [1., 200., 2e5]

class Runner(object):
    def getHostDir(run_id):
        return '{}/temp/{}/'.format(case, run_id)

    def spawnJob(exe, args, **kwargs):
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
    internalCells = []
    with h5py.File(case + 'mesh.hdf5', 'r') as mesh:
        nCount = mesh['parallel/end'][:]-mesh['parallel/start'][:]
        nInternalCells = nCount[:,4]
        nGhostCells = nCount[:,2]-nCount[:,3]
        start = 0
        for i in range(0, nProcessors):
            n = nInternalCells[i] 
            internalCells.append(np.arange(start, start + n))
            start += n + nGhostCells[i]
    internalCells = np.concatenate(internalCells)

    fieldNames = ['rho', 'rhoU', 'rhoE']
    program = source + 'apps/problem.py'

    reference = [1., 200., 2e5]
    def getInternalFields(case, time):
        fields = []
        with h5py.File(case + getTime(time) + '.hdf5', 'r') as phi:
            for name in fieldNames:
                fields.append(phi[name + '/field'][:][internalCells])
        fields = [x/y for x, y in zip(fields, reference)]
        return np.hstack(fields).ravel()

    def writeFields(fields, caseDir, ntime):
        fields = fields.reshape((fields.shape[0]/5, 5))
        fields = fields[:,[0]], fields[:,1:4], fields[:,[4]]
        fields = [x*y for x, y in zip(fields, reference)]
        timeFile = caseDir + getTime(ntime) + '.hdf5' 
        shutil.copy(case + stime + '.hdf5', timeFile)
        with h5py.File(timeFile, 'r+') as phi:
            for index, name in enumerate(fieldNames):
                field = phi[name + '/field'][:]
                field[internalCells] = fields[index]
                phi[name + '/field'][:] = field

    def runCase(initFields, parameters, nSteps, run_id):

        # generate case folders
        caseDir = '{}/temp/{}/'.format(case, run_id)
        mesh.case = caseDir
        if not os.path.exists(caseDir):
            os.makedirs(caseDir)
        shutil.copy(case + problem, caseDir)
        shutil.copy(case + 'mesh.hdf5', caseDir)
        for pkl in glob.glob(case + '*.pkl'):
            shutil.copy(pkl, caseDir)

        # write initial field
        writeFields(initFields, caseDir, time)

        # modify problem file
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
        for rep in range(0, 5):
            try:
                with open(outputFile, 'w') as f:
                    returncode = subprocess.call(['srun', '--exclusive', '-n', str(nProcessors),
                                      '-N', '1', '--resv-ports',
                                      program, problemFile, '--voyager'],
                                      stdout=f, stderr=f)
                if returncode:
                    raise Exception('Execution failed, check error log:', outputFile)
                objectiveSeries = np.loadtxt(caseDir + 'timeSeries.txt')
                break 
            except Exception as e:
                print caseDir, 'rep', rep, str(e)
                import time as timer
                timer.sleep(2)

        # read final fields
        times = [float(x[:-5]) for x in os.listdir(caseDir) if isfloat(x[:-5]) and x.endswith('.hdf5')]
        lastTime = sorted(times)[-1]
        finalFields = getInternalFields(caseDir, lastTime)
        # read objective values
        print caseDir

        return finalFields, objectiveSeries 

class ParallelRunner(Runner):
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

    def readFields(case, time, fieldFile):
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

    def writeFields(fieldFile, caseDir, ntime):
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
