#ifndef Py_ARRAYOBJECT_H
#define Py_ARRAYOBJECT_H

#include "ndarrayobject.h"
#include "npy_interrupt.h"

#ifdef NPY_NO_PREFIX
#include "noprefix.h"
#endif

//Check for exact native types that for sure do not
//support array related methods. Useful for faster checks when
//validating if an object supports these methods
#define ISEXACT_NATIVE_PYTYPE(op) (PyList_CheckExact(op) ||  (Py_None == op) || PyTuple_CheckExact(op) || PyFloat_CheckExact(op) || PyInt_CheckExact(op) || PyString_CheckExact(op) || PyUnicode_CheckExact(op))

//Check for exact native types that for sure do not
//support buffer protocol. Useful for faster checks when 
//validating if an object supports the buffer protocol.
#define NEVERSUPPORTS_BUFFER_PROTOCOL(op) ( PyList_CheckExact(op) ||  (Py_None == op) || PyTuple_CheckExact(op) )
#endif
