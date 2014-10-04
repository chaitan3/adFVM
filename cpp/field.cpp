#include "field.hpp"

Field::Field (const string name, const Mesh& mesh, const double time) {
    this->name = name;
    this->mesh = &mesh;

    PyObject *module = PyString_FromString("field");
    this->fieldModule = PyImport_Import(module);
    assert(this->fieldModule);
    Py_DECREF(module);
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
    Py_DECREF(nameString);
    Py_DECREF(timeDouble);
    Py_DECREF(args);

    getArray(this->pyField, "field", this->field);
    getBoundary(this->pyField, "boundary", this->boundary);
}

Field::~Field () {
    Py_DECREF(this->pyField);
    Py_XDECREF(this->fieldClass);
    Py_DECREF(this->fieldModule);

    Py_Finalize();
}



