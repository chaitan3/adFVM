#include "mesh.hpp"

void Mesh::build() {
    this->normals = move(mat(this->nFaces));
    this->faceCentres = move(mat(this->nFaces));
    this->areas = move(vec(this->nFaces));
    for (integer f = 0; f < this->nFaces; f++) {
        scalar *a = &this->points(this->faces(f, 1));
        scalar *b = &this->points(this->faces(f, 2));
        scalar *c = &this->points(this->faces(f, 3));
        scalar v1[3], v2[3];
        for (integer i = 0; i < 3; i++) {
            v1[i] = a[i]-b[i];
            v2[i] = b[i]-c[i];
        }
        scalar *normal = &this->normals(f);
        normal[0] = v1[1]*v2[2]-v1[2]*v2[1];
        normal[1] = v1[2]*v2[0]-v1[0]*v2[2];
        normal[2] = v1[0]*v2[1]-v1[1]*v2[0];
        scalar Ns = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
        for (integer i = 0; i < 3; i++) {
            normal[i] /= Ns;
        }
        scalar* sumCentre = &this->faceCentres(f);
        scalar faceCentre[3] = {0, 0, 0};
        scalar* area = &this->areas(f);
        area[0] = 0;
        for (integer i = 0; i < 3; i++) {
            sumCentre[i] = 0;
            for (integer j = 1; j < 5; j++) {
                faceCentre[i] += this->points(this->faces(f, j), i);
            }
            faceCentre[i] /= 4;
        }
        for (integer j = 1; j < 5; j++) {
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
        for (integer i = 0; i < 3; i++) {
            sumCentre[i] /= area[0];
        }
    }

    this->cellCentres = move(mat(this->nInternalCells));
    this->volumes = move(vec(this->nInternalCells));
    for (integer c = 0; c < this->nInternalCells; c++) {
        scalar* volume = &this->volumes(c);
        scalar* sumCentre = &this->cellCentres(c);
        scalar cellCentre[3] = {0, 0, 0};
        volume[0] = 0;
        for (integer i = 0; i < 3; i++) {
            sumCentre[i] = 0;
            for (integer j = 0; j < 6; j++) {
                cellCentre[i] += this->faceCentres(this->cellFaces(c, j), i);
            }
            cellCentre[i] /= 6;
        }
        for (integer j = 0; j < 6; j++) {
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
            for (integer i = 0; i < 3; i++) {
                sumCentre[i] += v*avgCentre[i];
            }
        }
        for (integer i = 0; i < 3; i++) {
            sumCentre[i] /= volume[0];
        }
    }
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

    meshp = new Mesh(meshObject);
    meshp->build();
    PyObject* outputs = PyTuple_New(5);
    PyTuple_SetItem(outputs, 0, putArray(meshp->normals));
    PyTuple_SetItem(outputs, 1, putArray(meshp->faceCentres));
    PyTuple_SetItem(outputs, 2, putArray(meshp->areas));
    PyTuple_SetItem(outputs, 3, putArray(meshp->cellCentres));
    PyTuple_SetItem(outputs, 4, putArray(meshp->volumes));
    return outputs;
}

PyMethodDef Methods[] = {
    {"build",  buildMesh, METH_VARARGS, "Execute a shell command."},
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
