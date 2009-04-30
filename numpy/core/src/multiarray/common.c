#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

/*#include <stdio.h>*/
#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "common.h"

/*
 * new reference
 * doesn't alter refcount of chktype or mintype ---
 * unless one of them is returned
 */
NPY_NO_EXPORT PyArray_Descr *
_array_small_type(PyArray_Descr *chktype, PyArray_Descr* mintype)
{
    PyArray_Descr *outtype;
    int outtype_num, save_num;

    if (PyArray_EquivTypes(chktype, mintype)) {
        Py_INCREF(mintype);
        return mintype;
    }


    if (chktype->type_num > mintype->type_num) {
        outtype_num = chktype->type_num;
    }
    else {
        if (PyDataType_ISOBJECT(chktype) &&
            PyDataType_ISSTRING(mintype)) {
            return PyArray_DescrFromType(NPY_OBJECT);
        }
        else {
            outtype_num = mintype->type_num;
        }
    }

    save_num = outtype_num;
    while (outtype_num < PyArray_NTYPES &&
          !(PyArray_CanCastSafely(chktype->type_num, outtype_num)
            && PyArray_CanCastSafely(mintype->type_num, outtype_num))) {
        outtype_num++;
    }
    if (outtype_num == PyArray_NTYPES) {
        outtype = PyArray_DescrFromType(save_num);
    }
    else {
        outtype = PyArray_DescrFromType(outtype_num);
    }
    if (PyTypeNum_ISEXTENDED(outtype->type_num)) {
        int testsize = outtype->elsize;
        int chksize, minsize;
        chksize = chktype->elsize;
        minsize = mintype->elsize;
        /*
         * Handle string->unicode case separately
         * because string itemsize is 4* as large
         */
        if (outtype->type_num == PyArray_UNICODE &&
            mintype->type_num == PyArray_STRING) {
            testsize = MAX(chksize, 4*minsize);
        }
        else {
            testsize = MAX(chksize, minsize);
        }
        if (testsize != outtype->elsize) {
            PyArray_DESCR_REPLACE(outtype);
            outtype->elsize = testsize;
            Py_XDECREF(outtype->fields);
            outtype->fields = NULL;
            Py_XDECREF(outtype->names);
            outtype->names = NULL;
        }
    }
    return outtype;
}

