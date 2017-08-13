#include "mesh.hpp"

void Mesh::buildBeforeWrite() {
    this->cellFaces = move(arrType<integer, 6>(this->nInternalCells));
    this->cells = move((arrType<integer, 8>(this->nInternalCells)));
    integer i, j, k, l, m, n;
    ivec indices(this->nInternalCells, true);
    //#pragma omp parallel for private(i)
    for (i = 0; i < this->nFaces; i++) {
        integer p = this->owner(i);
        this->cellFaces(p, indices(p)) = i;
        indices(p) += 1;
    }
    //#pragma omp parallel for private(i)
    for (i = 0; i < this->nInternalFaces; i++) {
        integer p = this->neighbour(i);
        this->cellFaces(p, indices(p)) = i;
        indices(p) += 1;
    }
    #pragma omp parallel for private(i, j, k, m, l, n)
    for (i = 0; i < this->nInternalCells; i++) {
        integer firstFace[4];
        integer nextFace[4];
        integer point, found;
        integer f = this->cellFaces(i, 0);
        for (j = 0; j < 4; j++) {
            firstFace[j] = this->faces(f, 1+j);
        }
        integer* cellPoint = &this->cells(i);
        for (j = 0; j < 4; j++) {
            point = firstFace[j];
            found = 0;
            for (n = 1; n < 6; n++) {
                f = this->cellFaces(i, n);
                for (k = 0; k < 4; k++) {
                    nextFace[k] = this->faces(f, 1+k);
                }
                for (k = 0; k < 4; k++) {
                    if (nextFace[k] == point) {
                        l = (k + 1) % 4;
                        for (m = 0; m < 4; m++) {
                            if (firstFace[m] == nextFace[l]) {
                                l = (k - 1) % 4;
                                break;
                            }
                        }
                        cellPoint[4+j] = nextFace[l];
                        found = 1;
                        break;
                    }
                if (found)
                    break;
                }
            }
        }

        for (j = 0; j < 4; j++) {
            cellPoint[j] = firstFace[j];
        }
    }
    PyObject_SetAttrString(this->mesh, "cellFaces", putArray(this->cellFaces));
    PyObject_SetAttrString(this->mesh, "cells", putArray(this->cells));
}

void Mesh::build() {
    this->normals = move(mat(this->nFaces));
    this->faceCentres = move(mat(this->nFaces));
    this->areas = move(vec(this->nFaces));
    integer i, j, c, f;
    #pragma omp parallel for private(f, i, j)
    for (f = 0; f < this->nFaces; f++) {
        scalar *a = &this->points(this->faces(f, 1));
        scalar *b = &this->points(this->faces(f, 2));
        scalar *c = &this->points(this->faces(f, 3));
        scalar v1[3], v2[3];
        for (i = 0; i < 3; i++) {
            v1[i] = a[i]-b[i];
            v2[i] = b[i]-c[i];
        }
        scalar *normal = &this->normals(f);
        normal[0] = v1[1]*v2[2]-v1[2]*v2[1];
        normal[1] = v1[2]*v2[0]-v1[0]*v2[2];
        normal[2] = v1[0]*v2[1]-v1[1]*v2[0];
        scalar Ns = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
        for (i = 0; i < 3; i++) {
            normal[i] /= Ns;
        }
        scalar* sumCentre = &this->faceCentres(f);
        scalar faceCentre[3] = {0, 0, 0};
        scalar* area = &this->areas(f);
        area[0] = 0;
        for (i = 0; i < 3; i++) {
            sumCentre[i] = 0;
            for (j = 1; j < 5; j++) {
                faceCentre[i] += this->points(this->faces(f, j), i);
            }
            faceCentre[i] /= 4;
        }
        for (j = 1; j < 5; j++) {
            scalar* point = &this->points(this->faces(f, j));
            scalar* nextPoint = &this->points(this->faces(f, (j % 4) + 1));
            scalar avgPoint[3] = {0, 0, 0};
            scalar N[3], v1[3], v2[3];
            for (integer i = 0; i < 3; i++) {
                avgPoint[i] += (point[i] + nextPoint[i] + faceCentre[i])/3;
                v1[i] = nextPoint[i]-point[i];
                v2[i] = faceCentre[i]-point[i];
            }
            N[0] = v1[1]*v2[2]-v1[2]*v2[1];
            N[1] = v1[2]*v2[0]-v1[0]*v2[2];
            N[2] = v1[0]*v2[1]-v1[1]*v2[0];
            scalar Ns = sqrt(N[0]*N[0] + N[1]*N[1] + N[2]*N[2]);
            area[0] += Ns/2;
            for (integer i = 0; i < 3; i++) {
                sumCentre[i] += Ns*avgPoint[i]/2;
            }
        }
        for (i = 0; i < 3; i++) {
            sumCentre[i] /= area[0];
        }
    }
    PyObject_SetAttrString(this->mesh, "normals", putArray(this->normals));
    PyObject_SetAttrString(this->mesh, "faceCentres", putArray(this->faceCentres));
    PyObject_SetAttrString(this->mesh, "areas", putArray(this->areas));

    this->cellCentres = move(mat(this->nInternalCells));
    this->volumes = move(vec(this->nInternalCells));
    #pragma omp parallel for private(c, i, j)
    for (c = 0; c < this->nInternalCells; c++) {
        scalar* volume = &this->volumes(c);
        scalar* sumCentre = &this->cellCentres(c);
        scalar cellCentre[3] = {0, 0, 0};
        volume[0] = 0;
        for (i = 0; i < 3; i++) {
            sumCentre[i] = 0;
            for (integer j = 0; j < 6; j++) {
                cellCentre[i] += this->faceCentres(this->cellFaces(c, j), i);
            }
            cellCentre[i] /= 6;
        }
        for (j = 0; j < 6; j++) {
            integer f = this->cellFaces(c, j);
            scalar* faceCentre = &this->faceCentres(f);
            scalar area = this->areas(f);
            scalar* N = &this->normals(f);
            scalar v = 0;
            scalar avgCentre[3];
            for (integer i = 0; i < 3; i++) {
                scalar height = cellCentre[i]-faceCentre[i];
                scalar areaN = area*N[i];
                v += areaN*height;
                avgCentre[i] = 3./4*faceCentre[i] + 1./4*cellCentre[i];
            }
            v = abs(v/3);
            volume[0] += v;
            for (i = 0; i < 3; i++) {
                sumCentre[i] += v*avgCentre[i];
            }
        }
        for (i = 0; i < 3; i++) {
            sumCentre[i] /= volume[0];
        }
    }
    PyObject_SetAttrString(this->mesh, "cellCentres", putArray(this->cellCentres));
    PyObject_SetAttrString(this->mesh, "volumes", putArray(this->volumes));

    PyObject* ret = PyObject_CallMethod(this->mesh, "createGhostCells", NULL);
    PyObject_SetAttrString(this->mesh, "nLocalCells", ret);
    getMeshArray(this->mesh, "neighbour", this->neighbour);
    getMeshArray(this->mesh, "cellCentres", this->cellCentres);

    this->deltas = move(vec(nFaces));
    this->deltasUnit = move(mat(nFaces));
    this->weights = move(vec(nFaces));
    this->linearWeights = move(arrType<scalar, 2>(nFaces));
    this->quadraticWeights = move(arrType<scalar, 2, 3>(nFaces));
    #pragma omp parallel for private(f, i, j)
    for (f = 0; f < this->nFaces; f++) {
        integer p = this->owner(f);
        integer n = this->neighbour(f);
        scalar* P = &this->cellCentres(p);
        scalar* N = &this->cellCentres(n);
        scalar delta[3];
        for (i = 0; i < 3; i++) {
            delta[i] = P[i]-N[i];
        }
        scalar d = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
        this->deltas(f) = d;
        for (i = 0; i < 3; i++) {
            this->deltasUnit(f, i) = delta[i]/d;
        }
        scalar* F = &this->faceCentres(f);
        scalar nD = 0, pD = 0;
        scalar* normal = &this->normals(f);
        scalar nF[3], pF[3];
        for (i = 0; i < 3; i++) {
            nF[i] = (F[i]-N[i]);
            pF[i] = (F[i]-P[i]);
            nD += nF[i]*normal[i];
            pD += pF[i]*normal[i];
        }
        nD = abs(nD);
        pD = abs(pD);
        this->weights(f) = nD/(nD + pD);
        //this->weights(f) = nD;
        scalar w1 = 0, w2 = 0;
        d = 0;
        for (i = 0; i < 3; i++) {
            w1 += -delta[i]*pF[i];
            w2 += delta[i]*nF[i];
            d += delta[i]*delta[i];
        }
        w1 /= d;
        w2 /= d;
        this->linearWeights(f, 0) = w1/3;
        this->linearWeights(f, 1) = w2/3;
        for (i = 0; i < 3; i++) {
            this->quadraticWeights(f, 0, i) = 2./3*pF[i] + 1./3*(pF[i]+w1*delta[i]);
            this->quadraticWeights(f, 1, i) = 2./3*nF[i] + 1./3*(nF[i]-w2*delta[i]);
        }
    }
    PyObject_SetAttrString(this->mesh, "deltas", putArray(this->deltas));
    PyObject_SetAttrString(this->mesh, "deltasUnit", putArray(this->deltasUnit));
    PyObject_SetAttrString(this->mesh, "weights", putArray(this->weights));
    PyObject_SetAttrString(this->mesh, "linearWeights", putArray(this->linearWeights));
    PyObject_SetAttrString(this->mesh, "quadraticWeights", putArray(this->quadraticWeights));

    this->cellNeighbours = move(arrType<integer, 6>(this->nInternalCells));
    #pragma omp parallel for private(c, i, j)
    for (c = 0; c < this->nInternalCells; c++) {
        integer* neigh = &this->cellNeighbours(c);
        for (j = 0; j < 6; j++) {
            integer f = this->cellFaces(c, j);
            integer p = this->owner(f);
            integer n = this->neighbour(f);
            if (p != c) {
                neigh[j] = p;
            } else {
                if (n < this->nInternalCells) {
                    neigh[j] = n;
                } else {
                    neigh[j] = -1;
                }
            }
        }
    }
    PyObject_SetAttrString(this->mesh, "cellNeighboursMatOp", putArray(this->cellNeighbours));
}

struct memory mem = {0, 0};
Mesh *meshp = NULL;

//

#ifdef PY3
    #define initFunc PyInit_cmesh
#else
    #define initFunc initcmesh
#endif
#define modName "cmesh"

PyObject* buildMesh(PyObject *self, PyObject *args) {

    PyObject *meshObject = PyTuple_GetItem(args, 0);
    Py_INCREF(meshObject);

    //meshp = new Mesh(meshObject);
    meshp->build();
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* buildMeshBeforeWrite(PyObject *self, PyObject *args) {

    PyObject *meshObject = PyTuple_GetItem(args, 0);
    //Py_INCREF(meshObject);

    meshp = new Mesh(meshObject);
    meshp->buildBeforeWrite();
    Py_INCREF(Py_None);
    return Py_None;
}

PyMethodDef Methods[] = {
    {"build",  buildMesh, METH_VARARGS, "Execute a shell command."},
    {"buildBeforeWrite",  buildMeshBeforeWrite, METH_VARARGS, "Execute a shell command."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#ifdef PY3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,  /* m_base */
    modName,                 /* m_name */
    NULL,                   /* m_doc */
    -1,                     /* m_size */
    Methods            /* m_methods */
};
#endif

PyMODINIT_FUNC
initFunc(void)
{
    PyObject *m;

    #ifdef PY3
        m = PyModule_Create(&moduledef);
    #else
        m = Py_InitModule(modName, Methods);
        if (m == NULL)
            return;
    #endif
    import_array();

    #ifdef PY3
        return m;
    #endif
}
