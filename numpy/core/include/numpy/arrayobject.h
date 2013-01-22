#ifndef Py_ARRAYOBJECT_H
#define Py_ARRAYOBJECT_H

#include <Python.h>
#include "ndarrayobject.h"
#include "npy_interrupt.h"

#ifdef NPY_NO_PREFIX
#include "noprefix.h"
#endif

/*
 * Check for exact native types that for sure do not
 * support array related methods. Useful for faster checks when
 * validating if an object supports these methods
 */
#if PY_VERSION_HEX >= 0x03000000
/* PyInt_CheckExact is not in Python 3 */
#define ISEXACT_NATIVE_PYTYPE(op) (PyList_CheckExact(op) ||  (Py_None == op) || PyTuple_CheckExact(op) || PyFloat_CheckExact(op) || PyString_CheckExact(op) || PyUnicode_CheckExact(op))
#else
#define ISEXACT_NATIVE_PYTYPE(op) (PyList_CheckExact(op) ||  (Py_None == op) || PyTuple_CheckExact(op) || PyFloat_CheckExact(op) || PyInt_CheckExact(op) || PyString_CheckExact(op) || PyUnicode_CheckExact(op))
#endif
/* Check for exact native types that for sure do not
 * support buffer protocol. Useful for faster checks when 
 * validating if an object supports the buffer protocol.
 */
#define NEVERSUPPORTS_BUFFER_PROTOCOL(op) ( PyList_CheckExact(op) ||  (Py_None == op) || PyTuple_CheckExact(op) )
#endif
