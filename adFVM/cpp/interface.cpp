#define NO_IMPORT_ARRAY
#include "interface.hpp"

#ifdef PY3
char* PyString_AsString(PyObject* result) {
    char *my_result;
    PyObject * temp_bytes = PyUnicode_AsEncodedString(result, "ASCII", "strict"); // Owned reference
    if (temp_bytes != NULL) {
        my_result = PyBytes_AS_STRING(temp_bytes); // Borrowed pointer
        my_result = strdup(my_result);
        Py_DECREF(temp_bytes);
        return my_result;
    } else {
        return NULL;
    }
}
#endif

PyObject* PyTuple_CreateNone(int n) {
    PyObject* outputs = PyTuple_New(n);
    for (int i = 0; i < n; i++) {
        Py_INCREF(Py_None);
        PyTuple_SetItem(outputs, i, Py_None);
    }
    return outputs;
}

map<string, int> PyOptions_Parse(PyObject* dict) {
    PyObject *key2, *value2;
    map<string, int> options;
    Py_ssize_t pos2 = 0;
    while (PyDict_Next(dict, &pos2, &key2, &value2)) {
        string ckey2 = PyString_AsString(key2);
        options[ckey2] = PyObject_IsTrue(value2);
    }
    return options;
}

long long mil = 0;
long long current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    long long curr_time = te.tv_sec*1000000LL + te.tv_usec; // caculate microseconds
    return curr_time;
}
