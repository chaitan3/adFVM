#include <metis.h>
#include "part_mesh.h"

PyObject* part_mesh(PyObject *self, PyObject *args) {
    PyArrayObject *eptr, *eind;
    PyArrayObject *epart, *npart;
    
    idx_t nparts;
    idx_t ncommon = 4;
    idx_t ne;
    idx_t nn;
    idx_t objval;

    if (!PyArg_ParseTuple(args, "iiOOiOO", &ne, &nn, &eptr, &eind, &nparts, &epart, &npart))
        return Py_None;

    int ret = METIS_PartMeshDual(&ne, &nn, PyArray_DATA(eptr), PyArray_DATA(eind), NULL, NULL, \
                       &ncommon, &nparts, NULL, NULL, \
                       &objval, PyArray_DATA(epart), PyArray_DATA(npart));
    return Py_BuildValue("i", ret);
}



