#include "objective.hpp"

scalar objectiveNone(RCF* rcf, const arr& U, const arr& T, const arr& p) {
    return 0;
}

scalar objectiveDrag(RCF* rcf, const arr& U, const arr& T, const arr& p) {
    const Mesh& mesh = *(rcf->mesh);
    string patchID = rcf->objectiveDragInfo;
    integer startFace, nFaces;
    tie(startFace, nFaces) = mesh.boundaryFaces.at(patchID);
    scalar drag = 0;

    for (integer i = 0; i < nFaces; i++) {
        integer f = startFace + i;
        integer c = mesh.nInternalCells - mesh.nInternalFaces + f;
        integer o = mesh.owner(f);
        scalar nx = mesh.normals(f, 0);
        scalar d = 0;
        for (integer j = 0; j < 3; j++) {
            d += pow(mesh.cellCentres(c, j)-mesh.cellCentres(o, j), 2);
        }
        d = sqrt(d);
        scalar U0 = U(c, 0);
        scalar U0i = U(o, 0);
        scalar mungUx = (rcf->*rcf->mu)(T(c))*(U0-U0i)/d;
        drag += (p(c)*nx-mungUx)*mesh.areas(f);
    }
    return drag;
}

/*def getPlane(solver):*/
    //#point = np.array([0.0032,0.0,0.0], config.precision)
    //point = np.array([0.032,0.0,0.0], config.precision)
    //normal = np.array([1.,0.,0.], config.precision)
    //interCells, interArea = intersectPlane(solver.mesh, point, normal)
    //#print interCells.shape, interArea.sum()
    //solver.postpro.extend([(ad.ivector(), interCells), (ad.bcmatrix(), interArea)])
    //return solver.postpro[-2][0], solver.postpro[-1][0], normal
    
//def objectivePressureLoss(fields, mesh):
    //#if not hasattr(objectivePressureLoss, interArea):
    //#    objectivePressureLoss.cells, objectivePressureLoss.area = getPlane(primal)
    //#cells, area = objectivePressureLoss.cells, objectivePressureLoss.area
    //ptin = 104190.
    //cells, area, normal = getPlane(primal)
    //rho, rhoU, rhoE = fields
    //solver = rhoE.solver
    //g = solver.gamma
    //U, T, p = solver.primitive(rho, rhoU, rhoE)
    //pi, rhoi, Ui = p.field[cells], rho.field[cells], U.field[cells]
    //rhoUi, ci = rhoi*Ui, ad.sqrt(g*pi/rhoi)
    //rhoUni, Umagi = dot(rhoUi, normal), ad.sqrt(dot(Ui, Ui))
    //Mi = Umagi/ci
    //pti = pi*(1 + 0.5*(g-1)*Mi*Mi)**(g/(g-1))
    //#res = ad.sum((ptin-pti)*rhoUni*area)/(ad.sum(rhoUni*area) + config.VSMALL)
    //res = ad.sum((ptin-pti)*rhoUni*area)#/(ad.sum(rhoUni*area) + config.VSMALL)
    //return res 

//objective = objectiveDrag
/*#objective = objectivePressureLoss*/


