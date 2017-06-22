from . import config, riemann, interp
from .tensor import Tensor, ZeroTensor, Function
from .field import Field, IOField
from .op import  div, snGrad, grad, internal_sum
from .solver import Solver
from .interp import central, secondOrder

import numpy as np
#import adFVMcpp

logger = config.Logger(__name__)

class RCF(Solver):
    defaultConfig = Solver.defaultConfig.copy()
    defaultConfig.update({
                             #specific
                             'Cp': 1004.5, 
                             'gamma': 1.4, 
                             'mu': lambda T:  1.4792e-06*T**1.5/(T+116.), 
                             'Pr': 0.7, 
                             'CFL': 1.2,
                             'stepFactor': 1.2,
                             'timeIntegrator': 'SSPRK', 'nStages': 3,
                             'objectiveDragInfo': None,
                             'objectivePLInfo': None,
                             # eulerHLLC DOES NOT WORK
                             'riemannSolver': 'eulerRoe',
                             #'boundaryRiemannSolver': 'eulerLaxFriedrichs',
                             'boundaryRiemannSolver': 'eulerRoe',
                             'readConservative': False,
                             'faceReconstructor': 'SecondOrder',
                        })

    def __init__(self, case, **userConfig):
        super(RCF, self).__init__(case, **userConfig)

        self.Cv = self.Cp/self.gamma
        self.R = self.Cp - self.Cv
        self.kappa = lambda mu, T: mu*(self.Cp/self.Pr)
        self.riemannSolver = getattr(riemann, self.riemannSolver)
        self.boundaryRiemannSolver = getattr(riemann, self.boundaryRiemannSolver)
        self.faceReconstructor = getattr(interp, self.faceReconstructor)

        self.names = ['rho', 'rhoU', 'rhoE']
        self.dimensions = [(1,), (3,), (1,)]

        self.initSource()

        self.Uref = 33.
        self.Tref = 300.
        self.pref = 1e5
        self.tref = 1.
        self.Jref = 1.
        return

    def compileInit(self):
        super(RCF, self).compileInit()
        #self.faceReconstructor = self.faceReconstructor(self)
        return

        # only reads fields
    @config.timeFunction('Time for reading fields')
    def readFields(self, t, suffix=''):

        with IOField.handle(t):
            U = IOField.read('U' + suffix, skipField=False)
            T = IOField.read('T' + suffix, skipField=False)
            p = IOField.read('p' + suffix, skipField=False)
            if self.readConservative:
                rho = IOField.read('rho' + suffix, skipField=False)
                rhoU = IOField.read('rhoU' + suffix, skipField=False)
                rhoE = IOField.read('rhoE' + suffix, skipField=False)
                U.field, T.field, p.field = [phi.field for phi in self.primitive(rho, rhoU, rhoE)]
            if self.dynamicMesh:
                self.mesh.read(IOField._handle)
        fields = [U, T, p]
        # IO Fields, with phi attribute as a CellField
        if self.firstRun:
            self.U, self.T, self.p = fields
            self.fields = fields
        else:
            self.updateFields(fields)
        self.firstRun = False
        return list(self.conservative(*self.fields))

    # reads and updates ghost cells
    def initFields(self, fields):
        newFields = adFVMcpp.ghost(*[phi.field for phi in fields])
        #import pdb;pdb.set_trace()
        return self.getFields(newFields, IOField, refFields=fields)
    
    @config.timeFunction('Time for writing fields')
    def writeFields(self, fields, t):
        n = len(self.names)
        fields, rest = fields[:n], fields[n:]
        fields = self.initFields(fields)
        U, T, p = self.primitive(*fields)
        for phi, phiN in zip(self.fields, [U, T, p]):
            phi.field = phiN.field

        with IOField.handle(t):
            for phi in fields + self.fields + rest:
                phi.write(skipProcessor=True)
            if self.dynamicMesh:
                self.mesh.write(IOField._handle)
        return

    def initOrder(self, fields):
        return [fields[2], fields[0], fields[1]]

    # operations
    def primitive(self, rho, rhoU, rhoE):
        logger.info('converting fields to primitive')
        U = rhoU/rho
        E = rhoE/rho
        e = E - 0.5*U.magSqr()
        p = (self.gamma-1)*rho*e
        T = e*(1./self.Cv)
        if isinstance(U, Field):
            U.name, T.name, p.name = 'U', 'T', 'p'
        return U, T, p

    def conservative(self, U, T, p):
        logger.info('converting fields to conservative')
        e = self.Cv*T
        rho = p/(e*(self.gamma-1))
        rhoE = rho*(e + 0.5*U.magSqr())
        rhoU = U*rho
        if isinstance(U, Field):
            rho.name, rhoU.name, rhoE.name = self.names
        return rho, rhoU, rhoE

    def getFlux(self, U, T, p, Normals):
        rho, rhoU, rhoE = self.conservative(U, T, p)
        Un = U.dot(Normals)
        rhoFlux = rho*Un
        rhoUFlux = rhoU*Un + p*Normals
        rhoEFlux = (rhoE + p)*Un
        return rhoFlux, rhoUFlux, rhoEFlux

    def getRho(self, T, p):
        e = self.Cv*T
        rho = p/(e*(self.gamma-1))
        return rho

    def viscousFlux(self, TL, TR, UF, TF, gradUF, gradTF):
        mesh = self.mesh
        mu = self.mu(TF)
        kappa = self.kappa(mu, TF)

        qF = kappa*snGrad(TL, TR, mesh);
        tmp2 = ZeroTensor((3,))
        tmp3 = ZeroTensor((1,))
        for i in range(0, 3):
            for j in range(0, 3):
                tmp2[i] += (gradUF[i,j] + gradUF[j,i])*mesh.normals[j]
            tmp3 += gradUF[i,i];
        sigmaF = mu*(tmp2-2./3*tmp3*mesh.normals)
        sigmadotUF = ZeroTensor((1,))
        rhoUFlux = -sigmaF
        rhoEFlux = -(qF + sigmaF.dot(UF));
        return rhoUFlux, rhoEFlux

    def flux(self, *args):
        mesh = self.mesh
        if not hasattr(self, '_flux'):
            UL, UR = Tensor((3,)), Tensor((3,))
            TL, TR = Tensor((1,)), Tensor((1,))
            pL, pR = Tensor((1,)), Tensor((1,))
            gradUL, gradUR = Tensor((3,3)), Tensor((3,3))
            gradTL, gradTR = Tensor((1,3)), Tensor((1,3))
            gradpL, gradpR = Tensor((1,3)), Tensor((1,3))
            ULF, URF = secondOrder(UL, UR, gradUL, mesh, 0), secondOrder(UR, UL, gradUR, mesh, 1)
            TLF, TRF = secondOrder(TL, TR, gradTL, mesh, 0), secondOrder(TR, TL, gradTR, mesh, 1)
            pLF, pRF = secondOrder(pL, pR, gradpL, mesh, 0), secondOrder(pR, pL, gradpR, mesh, 1)


            rhoLF, rhoULF, rhoELF = self.conservative(ULF, TLF, pLF)
            rhoRF, rhoURF, rhoERF = self.conservative(URF, TRF, pRF)

            rhoFlux, rhoUFlux, rhoEFlux = self.riemannSolver(self.gamma, pLF, pRF, TLF, TRF, ULF, URF, \
                rhoLF, rhoRF, rhoULF, rhoURF, rhoELF, rhoERF, mesh.normals)

            UF = 0.5*(ULF + URF)
            TF = 0.5*(TLF + TRF)
            gradTF = 0.5*(gradTL + gradTR)
            gradUF = 0.5*(gradUL + gradUR)

            ret = self.viscousFlux(TL, TR, UF, TF, gradUF, gradTF)
            rhoUFlux += ret[0]
            rhoEFlux += ret[1]

            drhoL, drhoR = div(rhoFlux, mesh)
            drhoUL, drhoUR = div(rhoUFlux, mesh)
            drhoEL, drhoER = div(rhoEFlux, mesh)
            #this->operate->div(&rhoEFlux, drhoE, i, neighbour);

            self._flux = Function([UL, UR, TL, TR, pL, pR, gradUL, gradUR, gradTL, gradTR, gradpL, gradpR],
                                  [drhoL, drhoR, drhoUL, drhoUR, drhoEL, drhoER], mesh)

        return self._flux(*args)
          
    def equation(self, rho, rhoU, rhoE):
        logger.info('computing RHS/LHS')
        mesh = self.mesh.origMesh

        U, T, p = self.primitive(rho, rhoU, rhoE)
        U.field = np.concatenate((U.field, np.zeros((mesh.nGhostCells, 3))))
        T.field = np.concatenate((T.field, np.zeros((mesh.nGhostCells, 1))))
        p.field = np.concatenate((p.field, np.zeros((mesh.nGhostCells, 1))))
        # BC for 

        # gradient evaluated using gauss integration rule
        gradU = grad(central(U, mesh), ghost=True, numpy=True)
        gradT = grad(central(T, mesh), ghost=True, numpy=True)
        gradp = grad(central(p, mesh), ghost=True, numpy=True)
        #gradU = grad(U, ghost=True, op=True)
        #gradT = grad(T, ghost=True, op=True)
        #gradp = grad(p, ghost=True, op=True)
        #self.local = gradp.field
        #self.remote = gradpO.field

        fields = []
        for phi in [U, T, p, gradU, gradT, gradp]:
            fields.append(phi.field[mesh.owner])
            fields.append(phi.field[mesh.neighbour])
        for phi in mesh.gradFields:
            fields.append(getattr(mesh, phi))
        flux = self.flux(*fields)

        # CFL based time step
        self.aF = ((self.gamma-1)*self.Cp*TF).sqrt()
        if self.stage == 1:
            maxaF = (UF.dotN()).abs() + self.aF
            self.dtc = 2*self.CFL/internal_sum(maxaF, mesh, absolute=True)

        return [div(rhoFlux) - self.sourceFields[0], \
                #ddt(rhoU, self.dt) + div(rhoUFlux) + grad(pF) - div(sigmaF) - source[1],
                div(rhoUFlux - sigmaF) - self.sourceFields[1], \
                div(rhoEFlux - qF - sigmadotUF) - self.sourceFields[2]]


    def boundary(self, rhoN, rhoUN, rhoEN):
        logger.info('correcting boundary')
        UN, TN, pN = self.primitive(rhoN, rhoUN, rhoEN)
        U, T, p = self.getBCFields()
        #p.resetField()
        #U.resetField()
        #T.resetField()
        U.field = UN.field
        T.field = TN.field
        p.field = pN.field
        p.setInternalField(pN.field)
        U.setInternalField(UN.field)
        T.setInternalField(TN.field)
        U.updateProcessorCells([U, T, p])
        return list(self.conservative(U, T, p))
    
