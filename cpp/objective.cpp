#include "objective.hpp"

#include <string>
#include <sstream>
#include <vector>
#include <iterator>

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
            *(result++) = item;
        }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}


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
        //for (integer j = 0; j < 3; j++) {
        //    d += pow(mesh.cellCentres(c, j)-mesh.cellCentres(o, j), 2);
        //}
        //d = sqrt(d);
        d = mesh.deltas(f);
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
    scalar w = 0;

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
        pl += (ptin-pti)*rhoUni*areas[i]/ptin;
        w += rhoUni*areas[i];
    }
    //return pl;
    scalar gpl, gw;
    AMPI_Allreduce(&pl, &gpl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    AMPI_Allreduce(&w, &gw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return gpl/gw;
}

scalar objectiveHeatTransfer(RCF *rcf, const mat& U, const vec& T, const vec& p) {
    const Mesh& mesh = *(rcf->mesh);
    string patches = rcf->objectiveDragInfo;
    scalar ht = 0;
    scalar w = 0;
    for (string patchID : split(patches, '|')) {
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patchID);
        for (integer i = 0; i < nFaces; i++) {
            integer f = startFace + i;
            integer c = mesh.nInternalCells - mesh.nInternalFaces + f;
            integer o = mesh.owner(f);

            // HACK
            if (patchID == "pressure") {
                if ((mesh.faceCentres(f, 0) < 0.33757) || (mesh.faceCentres(f, 1) > 0.04692))
                continue;
            }
            if (patchID == "suction") {
                if ((mesh.faceCentres(f, 0) < 0.035241) || (mesh.faceCentres(f, 1) > 0.044337))
                continue;
            }

            scalar d = mesh.deltas(f);
            scalar Tw = T(c);
            scalar k = (rcf->*rcf->mu)(Tw)*rcf->Cp/rcf->Pr;
            scalar dtdn = (Tw-T(o))/d;
            ht += k*dtdn*mesh.areas(f);
            w += mesh.areas(f);
        }
    }
    scalar ght, gw;
    AMPI_Allreduce(&ht, &ght, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    AMPI_Allreduce(&w, &gw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return ght/gw;
}

scalar objectiveOptim(RCF *rcf, const mat& U, const vec& T, const vec& p) {
    scalar ht = objectiveHeatTransfer(rcf, U, T, p);
    scalar pc = objectivePressureLoss(rcf, U, T, p);
    scalar k = (rcf->*rcf->mu)(300)*rcf->Cp/rcf->Pr;
    scalar Nu = -ht*0.71e-3/(120.*k);
    scalar obj = Nu/2000. + pc*0.4;
    //cout << "Nu " << Nu << endl;
    //cout << "pc " << pc << endl;
    //cout << "obj " << obj << endl;
    return obj;
}
