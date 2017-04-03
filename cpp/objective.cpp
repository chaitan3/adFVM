#include "objective.hpp"

scalar objectiveNone(RCF* rcf, const mat& U, const vec& T, const vec& p) {
    return 0;
}

scalar objectiveDrag(RCF* rcf, const mat& U, const vec& T, const vec& p) {
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

   
scalar objectivePressureLoss(RCF* rcf, const mat& U, const vec& T, const vec& p) {
    integer nCells = rcf->objectivePLInfo["cells"].size()/sizeof(integer);
    integer* cells = (integer*) rcf->objectivePLInfo.at("cells").data();
    uscalar* areas = (uscalar*) rcf->objectivePLInfo.at("areas").data();
    uscalar* normal = (uscalar*) rcf->objectivePLInfo.at("normal").data();
    scalar ptin = stod(rcf->objectivePLInfo.at("ptin"));
    scalar g = rcf->gamma;
    scalar pl = 0;

    for (integer i = 0; i < nCells; i++) {
        integer index = cells[i];
        scalar pi = p(index);
        scalar Ti = T(index);
        const scalar* Ui = &U(index);
        scalar rhoi = pi/(rcf->Cv*Ti*(g - 1));
        scalar ci = sqrt(g*pi/rhoi);

        scalar rhoUni = 0;
        scalar Umagi = 0;
        for (integer j = 0; j < 3; j++) {
            rhoUni += rhoi*Ui[j]*normal[j];
            Umagi += Ui[j]*Ui[j];
        }
        scalar Mi = sqrt(Umagi)/ci;
        scalar pti = pi*pow(1 + 0.5*(g-1)*Mi*Mi, g/(g-1));
        pl += (ptin-pti)*rhoUni*areas[i];
    }
    return pl;
}

//objective = objectiveDrag
/*#objective = objectivePressureLoss*/


