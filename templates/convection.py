primal = RCF('cases/convection/', {'R': 8.314, 'Cp': 1006., 'gamma': 1.4, 'mu': 0., 'Pr': 0.7, 'CFL': 0.2})

def objective(fields):
    rho, rhoU, rhoE = fields
    mesh = rho.mesh
    mid = np.array([0.75, 0.5, 0.5])
    indices = range(0, mesh.nInternalCells)
    G = np.exp(-100*np.linalg.norm(mid-mesh.cellCentres[indices], axis=1)**2).reshape(-1,1)*mesh.volumes[indices]
    return ad.sum(rho.field[indices]*G)/(nSteps + 1)

def perturb(fields, eps=1E-2):
    rho, rhoU, rhoE = fields
    mesh = rho.mesh
    mid = np.array([0.5, 0.5, 0.5])
    indices = range(0, mesh.nInternalCells)
    G = eps*ad.array(np.exp(-100*np.linalg.norm(mid-mesh.cellCentres[indices], axis=1)**2).reshape(-1,1))
    rho.field[indices] += G


