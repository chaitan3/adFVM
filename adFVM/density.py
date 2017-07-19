from . import config, riemann, interp
from .tensor import Tensorize
from .variable import Variable, Function
from .field import Field, IOField
from .op import  div, absDiv, snGrad, grad, internal_sum
from .solver import Solver
from .interp import central, secondOrder
from . import BCs
from . import postpro

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
                             # eulerHLLC DOES NOT WORK
                             'riemannSolver': 'eulerRoe',
                             'objectivePLInfo': None,
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

        #self.initSource()

        self.Uref = 33.
        self.Tref = 300.
        self.pref = 1e5
        self.tref = 1.
        self.Jref = 1.
        return

    def compileInit(self):
        #super(RCF, self).compileInit()

        #self.faceReconstructor = self.faceReconstructor(self)
        Function.createCodeDir(self.mesh.caseDir)

        if config.compile:
            Function.clean()
            mesh = self.mesh.symMesh

            # init function
            rho, rhoU, rhoE = Variable((mesh.nInternalCells, 1)), Variable((mesh.nInternalCells, 3)), Variable((mesh.nInternalCells, 1)),
            U, T, p = Tensorize(self.primitive)(rho, rhoU, rhoE)
            UN, TN, pN = self.U.completeField(U), self.T.completeField(T), self.p.completeField(p)
            rhoN, rhoUN, rhoEN = Tensorize(self.conservative)(UN, TN, pN)
            self.mapBoundary = Function([rho, rhoU, rhoE], [rhoN, rhoUN, rhoEN])


            rho, rhoU, rhoE = Variable((mesh.nInternalCells, 1)), Variable((mesh.nInternalCells, 3)), Variable((mesh.nInternalCells, 1)),
            self.map = Function([rho, rhoU, rhoE], self.equation(rho, rhoU, rhoE))

        Function.compile()
        Function._module.init(*([self.mesh.origMesh] + [phi.boundary for phi in self.fields] + [self.__class__.defaultConfig]))
        #TensorFunction._module.finalize()
        #import atexit
        #atexit.register(TensorFunction._module.finalize)

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
        newFields = self.mapBoundary(*[phi.field for phi in fields])
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

    # operations, dual
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

    # used in symbolic
    def getFlux(self, U, T, p, Normals):
        rho, rhoU, rhoE = self.conservative(U, T, p)
        Un = U.dot(Normals)
        rhoFlux = rho*Un
        rhoUFlux = rhoU*Un + p*Normals
        rhoEFlux = (rhoE + p)*Un
        return rhoFlux, rhoUFlux, rhoEFlux

    def viscousFlux(self, TL, TR, UF, TF, gradUF, gradTF):
        mesh = self.mesh.symMesh
        mu = self.mu(TF)
        kappa = self.kappa(mu, TF)

        qF = kappa*snGrad(TL, TR, mesh);
        tmp2 = (gradUF + gradUF.transpose()).tensordot(mesh.normals)
        tmp3 = gradUF.trace()

        sigmaF = mu*(tmp2-2./3*tmp3*mesh.normals)
        rhoUFlux = -sigmaF
        rhoEFlux = -(qF + sigmaF.dot(UF));
        return rhoUFlux, rhoEFlux

    #symbolic funcs

    def gradients(self, U, T, p, neighbour=True, boundary=False):
        mesh = self.mesh.symMesh
        if boundary:
            UF = U.extract(mesh.neighbour)
            TF = T.extract(mesh.neighbour)
            pF = p.extract(mesh.neighbour)
        else:
            UF = central(U, mesh)
            TF = central(T, mesh)
            pF = central(p, mesh)

        gradU = grad(UF, mesh, neighbour)
        gradT = grad(TF, mesh, neighbour)
        gradp = grad(pF, mesh, neighbour)

        return gradU, gradT, gradp
        #inputs = [getattr(mesh, attr) for attr in mesh.gradFields] + \
        #         [getattr(mesh, attr) for attr in mesh.intFields]

        #return TensorFunction(name, [U, T, p] + inputs, [gradU, gradT, gradp])


        
    def flux(self, U, T, p, gradU, gradT, gradp, characteristic=False, neighbour=True):
        mesh = self.mesh.symMesh
        P, N = mesh.owner, mesh.neighbour

        ULF = secondOrder(U, gradU, mesh, 0)
        TLF = secondOrder(T, gradT, mesh, 0)
        pLF = secondOrder(p, gradp, mesh, 0)
        rhoLF, rhoULF, rhoELF = self.conservative(ULF, TLF, pLF)

        if characteristic:
            URF, TRF, pRF = U.extract(N), T.extract(N), p.extract(N)
            rhoRF, rhoURF, rhoERF = self.conservative(URF, TRF, pRF)
            rhoFlux, rhoUFlux, rhoEFlux = self.boundaryRiemannSolver(self.gamma, pLF, pRF, TLF, TRF, ULF, URF, \
            rhoLF, rhoRF, rhoULF, rhoURF, rhoELF, rhoERF, mesh.normals)
            UF = URF
            TF = TRF
            gradUF = gradU.extract(N)
            gradTF = gradT.extract(N)
        else:
            URF = secondOrder(U, gradU, mesh, 1)
            TRF = secondOrder(T, gradT,  mesh, 1)
            pRF = secondOrder(p, gradp, mesh, 1)
            rhoRF, rhoURF, rhoERF = self.conservative(URF, TRF, pRF)
            rhoFlux, rhoUFlux, rhoEFlux = self.riemannSolver(self.gamma, pLF, pRF, TLF, TRF, ULF, URF, \
            rhoLF, rhoRF, rhoULF, rhoURF, rhoELF, rhoERF, mesh.normals)
            UF = 0.5*(ULF + URF)
            TF = 0.5*(TLF + TRF)
            gradTF = 0.5*(gradT.extract(P) + gradT.extract(N))
            gradUF = 0.5*(gradU.extract(P) + gradU.extract(N))

        ret = self.viscousFlux(T.extract(P), T.extract(N), UF, TF, gradUF, gradTF)
        rhoUFlux += ret[0]
        rhoEFlux += ret[1]

        drho = div(rhoFlux, mesh, neighbour)
        drhoU = div(rhoUFlux, mesh, neighbour)
        drhoE = div(rhoEFlux, mesh, neighbour)

        aF = (self.Cp*TF*(self.gamma-1)).sqrt()
        maxaF = abs(UF.dot(mesh.normals)) + aF
        dtc = absDiv(maxaF, mesh, neighbour)

        return drho, drhoU, drhoE, dtc

        #inputs = [getattr(mesh, attr) for attr in mesh.gradFields] + \
        #         [getattr(mesh, attr) for attr in mesh.intFields]

        #return TensorFunction(name, [U, T, p, gradU, gradT, gradp] + inputs,
        #                          [drho, drhoU, drhoE, dtc])

    def boundaryFlux(self, U, T, p, gradU, gradT, gradp):
        mesh = self.mesh.symMesh
        P, N = mesh.owner, mesh.neighbour

        # boundary extraction could be done using cellstartface
        UR, TR, pR = U.extract(N), T.extract(N), p.extract(N) 
        gradUR, gradTR = gradU.extract(N), gradT.extract(N)
        TL = T.extract(P)

        rhoFlux, rhoUFlux, rhoEFlux = self.getFlux(UR, TR, pR, mesh.normals)
        ret = self.viscousFlux(TL, TR, UR, TR, gradUR, gradTR)
        rhoUFlux += ret[0]
        rhoEFlux += ret[1]

        drho = div(rhoFlux, mesh, False)
        drhoU = div(rhoUFlux, mesh, False)
        drhoE = div(rhoEFlux, mesh, False)

        aF = (self.Cp*TR*(self.gamma-1)).sqrt()
        maxaF = abs(UR.dot(mesh.normals)) + aF
        dtc = absDiv(maxaF, mesh, False)

        return drho, drhoU, drhoE, dtc
        #inputs = [getattr(mesh, attr) for attr in mesh.gradFields] + \
        #         [getattr(mesh, attr) for attr in mesh.intFields]

        #return TensorFunction("boundaryFlux", [U, T, p, gradU, gradT, gradp] + inputs,
        #                          [drho, drhoU, drhoE, dtc])

    def equation(self, rho, rhoU, rhoE):
        logger.info('computing RHS/LHS')
        mesh = self.mesh.symMesh

        U, T, p = Variable((mesh.nCells, 3)), Variable((mesh.nCells, 1)), Variable((mesh.nCells, 1))
        U, T, p = Tensorize(self.primitive)(rho, rhoU, rhoE, outputs=(U, T, p))
        gradU, gradT, gradp = Variable((mesh.nCells, 3, 3)), Variable((mesh.nCells, 1, 3)), Variable((mesh.nCells, 1, 3))
        for patchID in self.mesh.boundary:
            startFace, endFace, nFaces = self.mesh.getPatchFaceRange(patchID)
            patchType = self.mesh.boundary[patchID]['type']
            if patchType in config.coupledPatches:
                gradU, gradT, gradp = Tensorize(self.gradients)(U, T, p, neighbour=False, boundary=False, outputs=(gradU, gradT, gradp))
            else:
                gradU, gradT, gradp = Tensorize(self.gradients)(U, T, p, neighbour=False, boundary=True, outputs=(gradU, gradT, gradp))

        exit(1)

        return [Field('drho', drho, (1,)), Field('drho', drhoU, (1,)), Field('drho', drhoE, (1,))] 

        # CFL based time step
        #self.aF = ((self.gamma-1)*self.Cp*TF).sqrt()
        #if self.stage == 1:
        #    maxaF = (UF.dotN()).abs() + self.aF
        #    self.dtc = 2*self.CFL/internal_sum(maxaF, mesh, absolute=True)

        #return [div(rhoFlux) - self.sourceFields[0], \
        #        #ddt(rhoU, self.dt) + div(rhoUFlux) + grad(pF) - div(sigmaF) - source[1],
        #        div(rhoUFlux - sigmaF) - self.sourceFields[1], \
        #        div(rhoEFlux - qF - sigmadotUF) - self.sourceFields[2]]
