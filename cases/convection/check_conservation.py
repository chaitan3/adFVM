from templates.convection import *
from compute import getTotalPressureAndEntropy

mesh = primal.mesh
times = mesh.getTimes()
startTime, endTime = times[0], times[-1]

def getSystem(rho, rhoU, rhoE, s):
    volumes = mesh.origMesh.volumes
    rhoV = (rho*volumes).sum(axis=0)
    rhoUV = (rhoU*volumes).sum(axis=0)
    rhoEV = (rhoE*volumes).sum(axis=0)
    sV = (s*volumes).sum(axis=0)
    return np.concatenate([rhoV, rhoUV, rhoEV, sV])

primal.readFields(startTime)
rho, rhoU, rhoE = primal.conservative(primal.U, primal.T, primal.p)
_, _, _, s = getTotalPressureAndEntropy(primal.U, primal.T, primal.p)
start = getSystem(rho.field, rhoU.field, rhoE.field, s.field)
maxF = np.hstack((rho.field, rhoU.field, rhoE.field, s.max())).max(axis=0) 

primal.readFields(endTime)
rho, rhoU, rhoE = primal.conservative(primal.U, primal.T, primal.p)
_, _, _, s = getTotalPressureAndEntropy(primal.U, primal.T, primal.p)
end = getSystem(rho.field, rhoU.field, rhoE.field, s)
_, _, _, sEnd = getTotalPressureAndEntropy(primal.U, primal.T, primal.p)

# constant source
time = endTime-startTime
rhoS, rhoUS, rhoES = source([rho, rhoU, rhoE], mesh, startTime)
added = getSystem(rhoS*time, rhoUS*time, rhoES*time, 0)

print 'CONSERVATION'
print 'source:', added
res = end-start
absDiff = np.abs(res-added)
relDiff = absDiff/maxF
print 'absolute diff:', absDiff
print 'relative diff:', relDiff
print

print 'ENTROPY'
res = sEnd-sStart
absDiff = np.abs(res-added)
relDiff = absDiff/maxF
print ''
