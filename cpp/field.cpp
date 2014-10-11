#include "field.hpp"

Field::Field (const string name, const Mesh& mesh, const double time): 
    mesh(mesh), name(name) {
    Py_Initialize();

    PyObject* mainmod = PyImport_AddModule("__main__");
    PyObject* maindict = PyModule_GetDict(mainmod);
    this->fieldModule = PyImport_ImportModuleEx("field", maindict, maindict, NULL);
    assert(this->fieldModule);
    this->fieldClass = PyObject_GetAttrString(this->fieldModule, "CellField");
    PyObject *read = PyObject_GetAttrString(this->fieldClass, "read");
    assert(this->fieldClass);
    PyObject *nameString = PyString_FromString(name.c_str());
    PyObject *timeDouble = PyFloat_FromDouble(time);
    PyObject *args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, nameString);
    PyTuple_SetItem(args, 1, mesh.mesh);
    PyTuple_SetItem(args, 2, timeDouble);
    this->pyField = PyObject_CallObject(read,  args);
    assert(this->pyField);
    Py_DECREF(read);
    Py_DECREF(nameString);
    Py_DECREF(timeDouble);
    Py_DECREF(args);

    getArray(this->pyField, "field", this->field);
    this->boundary = getBoundary(this->pyField, "boundary");
}

Field::~Field () {
    Py_DECREF(this->pyField);
    Py_DECREF(this->fieldClass);
    Py_DECREF(this->fieldModule);
    if (Py_IsInitialized())
        Py_Finalize();
}



