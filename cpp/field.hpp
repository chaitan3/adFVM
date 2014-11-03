#ifndef FIELD_HPP
#define FIELD_HPP

#include "interface.hpp"

class Field {
    public:
        const string name;
        const Mesh& mesh;
        arr field;
        Boundary boundary;

        PyObject *fieldModule;
        PyObject *fieldClass;
        PyObject *pyField;
    
        ~Field ();
        void write(const double);
        void updateGhostCells();
        Field (const string, const Mesh&, const double);
};

#endif
