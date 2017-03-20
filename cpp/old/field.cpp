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

Field::Field (const Mesh& mesh, const arr& phi):
    mesh(mesh) {
    if (phi.cols() == mesh.nInternalCells) {
        this->field = arr::Zero(phi.rows(), mesh.nCells);
        this->boundary = mesh.defaultBoundary;
        SELECT(this->field, 0, mesh.nInternalCells) = phi;
        this->updateGhostCells();
    } else {
        this->field = phi;
    }
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
        Ref<arr> patchFaces = SELECT(this->field, cellStartFace, nFaces);
        const Ref<const iarr>& internalFaces = SELECT(mesh.owner, startFace, nFaces);

        if (patchType == "cyclic") {
            string neighbourPatchID = patchInfo.at("neighbourPatch");
            int neighbourStartFace = stoi(mesh.boundary.at(neighbourPatchID).at("startFace"));
            patchFaces = slice(this->field, SELECT(mesh.owner, neighbourStartFace, nFaces));
        } else if (patchType == "zeroGradient" || patchType == "empty" || patchType == "inletOutlet") {
            patchFaces = slice(this->field, internalFaces);
        } else if (patchType == "symmetryPlane" || patchType == "slip") {
            if (size == 3) {
                const Ref<const arr>& v = SELECT(mesh.normals, startFace, nFaces);
                patchFaces -= ROWMUL(v, DOT(v, patchFaces));
            } else {
                patchFaces = slice(this->field, internalFaces);
            }
        } else if (patchType == "fixedValue") {
            if (size == 3) {
                patchFaces = arr::Zero(3, nFaces);
                patchFaces.row(0) = 3*VectorXd::Ones(nFaces);
            } else {
                patchFaces = arr::Ones(1, nFaces);
            }
        } else {
            cout << "patch not found " << patchType << " " << patchID << endl;
        }
    }
}

Field::~Field () {
    //if (!Py_IsInitialized())
    //    return;
    //Py_DECREF(this->pyField);
    //Py_DECREF(this->fieldClass);
    //Py_DECREF(this->fieldModule);
    //Py_Finalize();
}



