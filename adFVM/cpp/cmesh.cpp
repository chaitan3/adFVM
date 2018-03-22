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
    assert (ret != NULL);
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
            //delta[i] = N[i]-P[i];
            delta[i] = P[i]-N[i];
        }
        scalar d = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
        this->deltas(f) = d;
        for (i = 0; i < 3; i++) {
            this->deltasUnit(f, i) = -delta[i]/d;
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

        // central difference + upwind gradient
        this->linearWeights(f, 0) = w1/3;
        this->linearWeights(f, 1) = w2/3;
        for (i = 0; i < 3; i++) {
            this->quadraticWeights(f, 0, i) = 2./3*pF[i] + 1./3*(pF[i]+w1*delta[i]);
            this->quadraticWeights(f, 1, i) = 2./3*nF[i] + 1./3*(nF[i]-w2*delta[i]);
        }
        
        // upwind
        //this->linearWeights(f, 0) = 0;
        //this->linearWeights(f, 1) = 0;
        //for (i = 0; i < 3; i++) {
        //    this->quadraticWeights(f, 0, i) = pF[i];
        //    this->quadraticWeights(f, 1, i) = nF[i];
        //}
    }
    PyObject_SetAttrString(this->mesh, "deltas", putArray(this->deltas));
    PyObject_SetAttrString(this->mesh, "deltasUnit", putArray(this->deltasUnit));
    PyObject_SetAttrString(this->mesh, "weights", putArray(this->weights));
    PyObject_SetAttrString(this->mesh, "linearWeights", putArray(this->linearWeights));
    PyObject_SetAttrString(this->mesh, "quadraticWeights", putArray(this->quadraticWeights));

    this->cellOwner = move(arrType<integer, 6>(this->nInternalCells));
    this->cellNeighbours = move(arrType<integer, 6>(this->nInternalCells));
    this->cellNeighboursFull = move(arrType<integer, 6>(this->nInternalCells));
    #pragma omp parallel for private(c, i, j)
    for (c = 0; c < this->nInternalCells; c++) {
        integer* neigh = &this->cellNeighbours(c);
        integer* neighFull = &this->cellNeighboursFull(c);
        integer* own = &this->cellOwner(c);
        for (j = 0; j < 6; j++) {
            integer f = this->cellFaces(c, j);
            integer p = this->owner(f);
            integer n = this->neighbour(f);
            if (p != c) {
                own[j] = 0;
                neigh[j] = p;
                neighFull[j] = p;
            } else {
                own[j] = 1;
                neighFull[j] = n;
                if (n < this->nInternalCells) {
                    neigh[j] = n;
                } else {
                    neigh[j] = -1;
                }
            }
        }
    }
    PyObject_SetAttrString(this->mesh, "cellNeighboursMatOp", putArray(this->cellNeighbours));
    PyObject_SetAttrString(this->mesh, "cellNeighbours", putArray(this->cellNeighboursFull));
    PyObject_SetAttrString(this->mesh, "cellOwner", putArray(this->cellOwner));
}

Mesh *meshp = NULL;

#ifdef CPU_FLOAT32
    #define MODULE cmesh_gpu
#else
    #define MODULE cmesh
#endif
#ifdef PY3
    #define initFunc GET_MODULE(PyInit_,MODULE)
#else
    #define initFunc GET_MODULE(init,MODULE)
#endif
#define modName VALUE(MODULE)
//#pragma message(PRINT(modName))
//#pragma message(PRINT(MODULE))

PyObject* buildMesh(PyObject *self, PyObject *args) {

    //PyObject *meshObject = PyTuple_GetItem(args, 0);
    //Py_INCREF(meshObject);
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

PyObject* computeSensitivity(PyObject* self, PyObject* args) {
    PyObject* gradients;
    PyObject* perturbation;
    if (!PyArg_ParseTuple(args, "OO", &gradients, &perturbation)) {
        cout << "parsing arguments failed" << endl;
        return NULL;
    }
    if (!PyList_Check(gradients) || !PyList_Check(perturbation)) {
        cout << "arguments not lists" << endl;
        return NULL;
    }
    int i, j;
    int n = PyList_Size(gradients);
    scalar sensitivity = 0;
    for (i = 0; i < n; i++) {
        PyArrayObject* grad = (PyArrayObject*) PyList_GetItem(gradients, i);
        PyArrayObject* per = (PyArrayObject*) PyList_GetItem(perturbation, i);
        assert(PyArray_IS_C_CONTIGUOUS(grad));
        assert(PyArray_IS_C_CONTIGUOUS(per));
        assert(PyArray_ITEMSIZE(grad) == sizeof(scalar));
        assert(PyArray_ITEMSIZE(per) == sizeof(scalar));
        int m = PyArray_SIZE(grad);
        assert(PyArray_SIZE(per) == m);
        scalar* grad_data = (scalar*)PyArray_DATA(grad);
        scalar* per_data = (scalar*)PyArray_DATA(per);
        #pragma omp parallel for private(j) reduction(+: sensitivity)
        for (j = 0; j < m; j++) {
            sensitivity += grad_data[j]*per_data[j];
        }
    }
    return Py_BuildValue("d", sensitivity);
}

PyObject* computeEnergy(PyObject* self, PyObject* args) {
    int i;
    PyObject* solver;
    PyObject* rhoaObj,*rhoUaObj,*rhoEaObj;
    PyObject* rhoObj,*rhoUObj,*rhoEObj;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &solver, &rhoaObj, &rhoUaObj, &rhoEaObj, &rhoObj, &rhoUObj, &rhoEObj)) {
        cout << "parsing arguments failed" << endl;
        return NULL;
    }
    scalar energy = 0;
    scalar g = getScalar(solver, "gamma");
    
    PyArrayObject* rhoaArr = (PyArrayObject*) rhoaObj;
    assert(PyArray_IS_C_CONTIGUOUS(rhoaArr));
    assert(PyArray_ITEMSIZE(rhoaArr) == sizeof(scalar));
    int m = PyArray_SIZE(rhoaArr);
    assert(PyArray_SIZE(rhoaArr) == m);
    PyArrayObject* rhoUaArr = (PyArrayObject*) rhoUaObj;
    assert(PyArray_IS_C_CONTIGUOUS(rhoUaArr));
    assert(PyArray_ITEMSIZE(rhoUaArr) == sizeof(scalar));
    assert(PyArray_SIZE(rhoUaArr) == m*3);
    PyArrayObject* rhoEaArr = (PyArrayObject*) rhoEaObj;
    assert(PyArray_IS_C_CONTIGUOUS(rhoEaArr));
    assert(PyArray_ITEMSIZE(rhoEaArr) == sizeof(scalar));
    assert(PyArray_SIZE(rhoEaArr) == m);
    PyArrayObject* rhoArr = (PyArrayObject*) rhoObj;
    assert(PyArray_IS_C_CONTIGUOUS(rhoArr));
    assert(PyArray_ITEMSIZE(rhoArr) == sizeof(scalar));
    assert(PyArray_SIZE(rhoArr) == m);
    PyArrayObject* rhoUArr = (PyArrayObject*) rhoUObj;
    assert(PyArray_IS_C_CONTIGUOUS(rhoUArr));
    assert(PyArray_ITEMSIZE(rhoUArr) == sizeof(scalar));
    assert(PyArray_SIZE(rhoUArr) == m*3);
    PyArrayObject* rhoEArr = (PyArrayObject*) rhoEObj;
    assert(PyArray_IS_C_CONTIGUOUS(rhoEArr));
    assert(PyArray_ITEMSIZE(rhoEArr) == sizeof(scalar));
    assert(PyArray_SIZE(rhoEArr) == m);

    scalar* rhoa = (scalar*)PyArray_DATA(rhoaArr);
    scalar* rhoUa = (scalar*)PyArray_DATA(rhoUaArr);
    scalar* rhoEa = (scalar*)PyArray_DATA(rhoEaArr);
    scalar* rho = (scalar*)PyArray_DATA(rhoArr);
    scalar* rhoU = (scalar*)PyArray_DATA(rhoUArr);
    scalar* rhoE = (scalar*)PyArray_DATA(rhoEArr);
    #pragma omp parallel for private(i) reduction(+: energy)
    for (i = 0; i < meshp->nInternalCells; i++) {
        scalar v = meshp->volumes(i);
        scalar u1 = rhoU[i*3]/rho[i];
        scalar u2 = rhoU[i*3+1]/rho[i];
        scalar u3 = rhoU[i*3+2]/rho[i];

        scalar q2 = u1*u1+u2*u2+u3*u3;
        scalar p = (rhoE[i]-rho[i]*q2/2)*(g-1);
        scalar H = g*p/(rho[i]*(g-1)) + q2/2;
        scalar w1 = rhoa[i]/v;
        scalar w2 = rhoUa[i*3]/v;
        scalar w3 = rhoUa[i*3+1]/v;
        scalar w4 = rhoUa[i*3+2]/v;
        scalar w5 = rhoEa[i]/v;
        scalar A11 = rho[i];
        scalar A12 = rho[i]*u1;
        scalar A13 = rho[i]*u2;
        scalar A14 = rho[i]*u3;
        scalar A15 = rhoE[i];
        scalar A22 = rho[i]*u1*u1+p;
        scalar A23 = rho[i]*u1*u2;
        scalar A24 = rho[i]*u1*u3;
        scalar A25 = rho[i]*H*u1;
        scalar A33 = rho[i]*u2*u2+p;
        scalar A34 = rho[i]*u2*u3;
        scalar A35 = rho[i]*H*u2;
        scalar A44 = rho[i]*u3*u3+p;
        scalar A45 = rho[i]*H*u3;
        scalar A55 = rho[i]*H*H-g*p*p/(rho[i]*(g-1));
        energy += (A11*w1*w1 + A22*w2*w2 + A33*w3*w3 + A44*w4*w4 + A55*w5*w5 + \
                  2*(A12*w1*w2 + A13*w1*w3 + A14*w1*w4 + A15*w1*w5 + \
                     A23*w2*w3 + A24*w2*w4 + A25*w2*w5 + \
                     A34*w3*w4 + A35*w3*w5 + \
                     A45*w4*w5
                    ))*v;
    }
    
    return Py_BuildValue("d", energy);
}

PyMethodDef Methods[] = {
    {"build",  buildMesh, METH_VARARGS, "Execute a shell command."},
    {"buildBeforeWrite",  buildMeshBeforeWrite, METH_VARARGS, "Execute a shell command."},
    {"computeSensitivity",  computeSensitivity, METH_VARARGS, "Execute a shell command."},
    {"computeEnergy",  computeEnergy, METH_VARARGS, "Execute a shell command."},
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

void cmesh_exit() {
    delete meshp;
}

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
    Py_AtExit(cmesh_exit);

    #ifdef PY3
        return m;
    #endif
}
