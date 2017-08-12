#include "mesh.hpp"

void Mesh::build() {
    cout << "hello" << endl;
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
    return Py_None;
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
