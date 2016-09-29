from . import config, riemann, interp
from .config import ad
from .field import Field, IOField
from .op import  div, snGrad, grad, internal_sum
from .solver import Solver
from .interp import central

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
        self.faceReconstructor = self.faceReconstructor(self)
        return

    def primitive(self, rho, rhoU, rhoE):
        logger.info('converting fields to primitive')
        U = rhoU/rho
        E = rhoE/rho
        e = E - 0.5*U.magSqr()
        p = (self.gamma-1)*rho*e
        T = e*(1./self.Cv)
        U.name, T.name, p.name = 'U', 'T', 'p'
        return U, T, p

    def conservative(self, U, T, p):
        logger.info('converting fields to conservative')
        e = self.Cv*T
        rho = p/(e*(self.gamma-1))
        E = e + 0.5*U.magSqr()
        rhoU = U*rho
        rhoE = rho*E
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

    # only reads fields
    @config.timeFunction('Time for reading fields')
    def readFields(self, t, suffix=''):
        firstRun = self.init is None

        with IOField.handle(t):
            U = IOField.read('U' + suffix, skipField=firstRun)
            T = IOField.read('T' + suffix, skipField=firstRun)
            p = IOField.read('p' + suffix, skipField=firstRun)
            if self.readConservative:
                rho = IOField.read('rho' + suffix, skipField=firstRun)
                rhoU = IOField.read('rhoU' + suffix, skipField=firstRun)
                rhoE = IOField.read('rhoE' + suffix, skipField=firstRun)
                U.field, T.field, p.field = [phi.field for phi in self.primitive(rho, rhoU, rhoE)]
            if self.dynamicMesh:
                self.mesh.read(IOField._handle)
        fields = [U, T, p]
        # IO Fields, with phi attribute as a CellField
        if firstRun:
            self.U, self.T, self.p = fields
            self.fields = fields
        else:
            self.updateFields(fields)
        return

    # reads and updates ghost cells
    def initFields(self, t, **kwargs):
        self.readFields(t, **kwargs)
        self.U.field, self.T.field, self.p.field = self.init(self.U.field, self.T.field, self.p.field)
        return self.conservative(self.U, self.T, self.p)
    
    @config.timeFunction('Time for writing fields')
    def writeFields(self, fields, t):
        U, T, p = self.primitive(*fields[:3])
        self.setFields([U, T, p])

        with IOField.handle(t):
            for phi in fields + self.fields:
                phi.write()
            if self.dynamicMesh:
                self.mesh.write(IOField._handle)
        return
          
    def equation(self, rho, rhoU, rhoE):
        logger.info('computing RHS/LHS')
        mesh = self.mesh

        U, T, p = self.primitive(rho, rhoU, rhoE)
        self.setBCFields([U, T, p])

        ## face reconstruction
        #rhoLF, rhoRF = TVD_dual(rho, gradRho)
        #rhoULF, rhoURF = TVD_dual(rhoU, gradRhoU)
        #rhoELF, rhoERF = TVD_dual(rhoE, gradRhoE)
        #ULF, TLF, pLF = self.primitive(rhoLF, rhoULF, rhoELF)
        #URF, TRF, pRF = self.primitive(rhoRF, rhoURF, rhoERF)

        # gradient evaluated using gauss integration rule
        gradU = grad(central(U, mesh), ghost=True)
        gradT = grad(central(T, mesh), ghost=True)
        gradp = grad(central(p, mesh), ghost=True)
        #gradU = grad(U, ghost=True, op=True)
        #gradT = grad(T, ghost=True, op=True)
        #gradp = grad(p, ghost=True, op=True)
        #self.local = gradp.field
        #self.remote = gradpO.field

        # for zeroGradient boundary
        UB, TB, pB = self.getBCFields()
        UB.grad, TB.grad, pB.grad = gradU, gradT, gradp
        #self.local = gradp.field[:mesh.nInternalCells]

        # face reconstruction
        #reconstuctor = Reconstruct(mesh, TVD)
        #ULF, URF = reconstructor.dual(U, gradU)
        #TLF, TRF = reconstructor.dual(T, gradT)
        #pLF, pRF = reconstructor.dual(p, gradp)
        ##ULF, URF = central(U, mesh), central(U, mesh)
        ##TLF, TRF = central(T, mesh), central(T, mesh)
        ##pLF, pRF = central(p, mesh), central(p, mesh)

        #rhoLF, rhoULF, rhoELF = self.conservative(ULF, TLF, pLF)
        #rhoRF, rhoURF, rhoERF = self.conservative(URF, TRF, pRF)

        ## don't apply TVD to boundary faces
        ## instead use the standard scalar dissipation?

        #rhoFlux, rhoUFlux, rhoEFlux, aF, UnF = self.riemannSolver(mesh, self.gamma, \
        #        pLF, pRF, TLF, TRF, ULF, URF, \
        #        rhoLF, rhoRF, rhoULF, rhoURF, rhoELF, rhoERF)
        rhoFlux = Field('rho', ad.zeros((mesh.nFaces,) + rho.dimensions), rho.dimensions)
        rhoUFlux = Field('rhoU', ad.zeros((mesh.nFaces,) + rhoU.dimensions), rhoU.dimensions)
        rhoEFlux = Field('rhoE', ad.zeros((mesh.nFaces,) + rhoE.dimensions), rhoE.dimensions)
        UF = Field('U', ad.zeros((mesh.nFaces,) + U.dimensions), U.dimensions)
        TF = Field('T', ad.zeros((mesh.nFaces,) + T.dimensions), T.dimensions)
        indices = self.faceReconstructor.indices

        # INTERNAL FLUX: includes internal faces
        # face reconstruction

        (ULIF, URIF), (TLIF, TRIF), (pLIF, pRIF) = self.faceReconstructor.dualSystem(
                (U, gradU), (T, gradT), (p, gradp)
        )

        rhoLIF, rhoULIF, rhoELIF = self.conservative(ULIF, TLIF, pLIF)
        rhoRIF, rhoURIF, rhoERIF = self.conservative(URIF, TRIF, pRIF)
        Normals = mesh.Normals.getField(indices)

        rhoIFlux, rhoUIFlux, rhoEIFlux = self.riemannSolver(self.gamma, \
                pLIF, pRIF, TLIF, TRIF, ULIF, URIF, \
                rhoLIF, rhoRIF, rhoULIF, rhoURIF, rhoELIF, rhoERIF, Normals)
        rhoFlux.setField(indices, rhoIFlux)
        rhoUFlux.setField(indices, rhoUIFlux)
        rhoEFlux.setField(indices, rhoEIFlux)
        UF.setField(indices, 0.5*(ULIF + URIF))
        TF.setField(indices, 0.5*(TLIF + TRIF))

        # BOUNDARY FLUX
        if self.faceReconstructor.boundary:
            # no riemann stuff
            indices = self.faceReconstructor.Bindices
            cellIndices = indices - mesh.nInternalFaces + mesh.nInternalCells
            UBF, TBF, pBF = U.getField(cellIndices), T.getField(cellIndices), p.getField(cellIndices)
            BNormals = mesh.Normals.getField(indices)
            rhoBFlux, rhoUBFlux, rhoEBFlux = self.getFlux(UBF, TBF, pBF, BNormals)
            rhoFlux.setField(indices, rhoBFlux)
            rhoUFlux.setField(indices, rhoUBFlux)
            rhoEFlux.setField(indices, rhoEBFlux)
            UF.setField(indices, UBF)
            TF.setField(indices, TBF)
        # CHARACTERISTIC BOUNDARY FLUX 
        if self.faceReconstructor.characteristic:
            # first order interpolation? really?
            indices = self.faceReconstructor.Cindices
            Iindices = mesh.owner[indices]
            cellIndices = indices - mesh.nInternalFaces + mesh.nInternalCells
            ULCF, TLCF, pLCF = U.getField(Iindices), T.getField(Iindices), p.getField(Iindices)
            URCF, TRCF, pRCF = U.getField(cellIndices), T.getField(cellIndices), p.getField(cellIndices)
            rhoLCF, rhoULCF, rhoELCF = self.conservative(ULCF, TLCF, pLCF)
            rhoRCF, rhoURCF, rhoERCF = self.conservative(URCF, TRCF, pRCF)
            CNormals = mesh.Normals.getField(indices)
            rhoCFlux, rhoUCFlux, rhoECFlux = self.boundaryRiemannSolver(self.gamma, \
                    pLCF, pRCF, TLCF, TRCF, ULCF, URCF, \
                    rhoLCF, rhoRCF, rhoULCF, rhoURCF, rhoELCF, rhoERCF, CNormals)
            rhoFlux.setField(indices, rhoCFlux)
            rhoUFlux.setField(indices, rhoUCFlux)
            rhoEFlux.setField(indices, rhoECFlux)
            UF.setField(indices, 0.5*(ULCF + URCF))
            TF.setField(indices, 0.5*(TLCF + TRCF))

        # viscous part
        mu = self.mu(TF)
        kappa = self.kappa(mu, TF)
        #qF = snGrad(T)*kappa
        gradTF = central(gradT, mesh)
        gradTCF = gradTF + snGrad(T)*mesh.Normals - (gradTF.dotN())*mesh.Normals
        qF = kappa*gradTCF.dotN()
        
        #gradUTF = central(gradU.transpose(), mesh)
        #sigmaF = (snGrad(U) + gradUTF.dotN() - (2./3)*mesh.Normals*gradUTF.trace())*mu
        gradUF = central(gradU, mesh)
        gradUCF = gradUF + snGrad(U).outer(mesh.Normals) - (gradUF.dotN()).outer(mesh.Normals)
        sigmaF = mu*((gradUCF + gradUCF.transpose()).dotN() - (2./3)*gradUCF.trace()*mesh.Normals)
        sigmadotUF = sigmaF.dot(UF)

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
        U.setInternalField(UN.field)
        T.setInternalField(TN.field)
        p.setInternalField(pN.field)
        return self.conservative(U, T, p)
    
