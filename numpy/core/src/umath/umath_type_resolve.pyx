from cpython.ref import Py_INCREF
from cpython cimport PyObject

from libc.stdio cimport printf

from numpy_base cimport (PyUFuncObject, NPY_CASTING, PyArrayObject,
                         PyArray_Descr, NPY_OBJECT, PyArray_DescrFromType,
                         PyArray_API)

from numpy cimport import_array



cdef public int object_ufunc_type_resolver(PyUFuncObject *ufunc,
                                           NPY_CASTING casting,
                                           PyArrayObject **operands,
                                           PyObject* type_tup,
                                           PyArray_Descr **out_dtypes):
    cdef int i
    cdef int nop = ufunc.nin + ufunc.nout

    printf("hello\n\n")
    printf("%p\n", PyArray_API)
    printf("%p\n", PyArray_DescrFromType)
    out_dtypes[0] = PyArray_DescrFromType(NPY_OBJECT)

    print("----------- ufuunc_type_resolver")

    if out_dtypes[0] is NULL:
        return -1

    for i in range(1, nop):
        Py_INCREF(<object>out_dtypes[0])
        out_dtypes[i] = out_dtypes[0]

    return 0


#import_array()
