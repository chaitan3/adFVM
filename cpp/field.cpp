#include "field.hpp"

Field::Field (const string name, const Mesh& mesh, const double time): 
    mesh(mesh), name(name) {
    Py_Initialize();

    PyObject* mainmod = PyImport_AddModule("__main__");
    PyObject* maindict = PyModule_GetDict(mainmod);
    this->fieldModule = PyImport_ImportModuleEx("field", maindict, maindict, NULL);
    assert(this->fieldModule);
    this->fieldClass = PyObject_GetAttrString(this->fieldModule, "CellField");
    assert(this->fieldClass);
    PyObject_SetAttrString(this->fieldClass, "solver", mesh.mesh);
    PyObject_SetAttrString(this->fieldClass, "mesh", mesh.mesh);
    PyObject *read = PyObject_GetAttrString(this->fieldClass, "read");
    PyObject *args = PyTuple_New(2);
    PyObject *nameString = PyString_FromString(name.c_str());
    PyObject *timeDouble = PyFloat_FromDouble(time);
    PyTuple_SetItem(args, 0, nameString);
    PyTuple_SetItem(args, 1, timeDouble);
    this->pyField = PyObject_CallObject(read,  args);
    assert(this->pyField);
    Py_DECREF(read);
    Py_DECREF(nameString);
    Py_DECREF(timeDouble);
    Py_DECREF(args);

    getArray(this->pyField, "field", this->field);
    this->boundary = getBoundary(this->pyField, "boundary");
}

void Field::write(const double time) {
    putArray(this->pyField, "field", this->field);
    PyObject *pyWrite = PyObject_GetAttrString(this->pyField, "write");
    PyObject *args = PyTuple_New(1);
    PyObject *timeDouble = PyFloat_FromDouble(time);
    PyTuple_SetItem(args, 0, timeDouble);
    PyObject_CallObject(pyWrite, args);
    Py_DECREF(pyWrite);
    Py_DECREF(timeDouble);
    Py_DECREF(args);
}

void Field::updateGhostCells() {
    for (auto& patch: this->boundary) {
        string patchType = patch.second["type"];
        string patchID = patch.first;
        int size = this->field.rows();
        const map<string, string>& patchInfo = mesh.boundary.at(patchID);
        int startFace = stoi(patchInfo.at("startFace"));
        int nFaces = stoi(patchInfo.at("nFaces"));
        int cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces;

        if (patchType == "cyclic") {
            string neighbourPatchID = patchInfo.at("neighbourPatch");
            int neighbourStartFace = stoi(mesh.boundary.at(neighbourPatchID).at("startFace"));
            this->field.block(0, cellStartFace, size, nFaces) = slice(this->field, SELECT(mesh.owner, neighbourStartFace, nFaces));
            cout << "cyclic" << endl;
        }
        else if (patchType == "zeroGradient") {
            this->field.block(0, cellStartFace, size, nFaces) = slice(this->field, SELECT(mesh.owner, startFace, nFaces));
            cout << "zeroGradient" << endl;
        }
        else {
            cout << "patch not found" << endl;
        }
    }
}

Field::~Field () {
    if (!Py_IsInitialized())
        return;
    Py_DECREF(this->pyField);
    Py_DECREF(this->fieldClass);
    Py_DECREF(this->fieldModule);
    Py_Finalize();
}



