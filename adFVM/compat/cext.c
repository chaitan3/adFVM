#include "cext.h"
#include "part_mesh.h"

static PyMethodDef module_methods[] = {
    {"part_mesh", part_mesh, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcext(void)
{
    PyObject *m = Py_InitModule3("cext", module_methods, NULL);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}
