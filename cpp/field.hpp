#ifndef FIELD_HPP
#define FIELD_HPP
#include "interface.hpp"

class Field {
    public:
        string name;
        const Mesh* mesh;
        mat field;
        Boundary boundary;

        PyObject *fieldModule;
        PyObject *fieldClass;
        PyObject *pyField;
    
        Field () {};
        ~Field ();
        Field (const string, const Mesh&, const double);
};

#endif
