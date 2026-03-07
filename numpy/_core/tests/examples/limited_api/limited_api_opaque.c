#ifndef _Py_OPAQUE_PYOBJECT
#error "This file must be compiled with -D_Py_OPAQUE_PYOBJECT"
#endif

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

static PyObject *limited_api_opaque_nonzero(PyObject *mod, PyArrayObject *self)
{
    PyArray_NonzeroFunc* nonzero = PyDataType_GetArrFuncs(PyArray_DESCR(self))->nonzero;

    NpyIter* iter;
    NpyIter_IterNextFunc *iternext;
    char** dataptr;
    npy_intp nonzero_count;
    npy_intp* strideptr,* innersizeptr;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(self) == 0) {
        return PyLong_FromLong(0);
    }

    /*
     * Create and use an iterator to count the nonzeros.
     *   flag NPY_ITER_READONLY
     *     - The array is never written to.
     *   flag NPY_ITER_EXTERNAL_LOOP
     *     - Inner loop is done outside the iterator for efficiency.
     *   flag NPY_ITER_NPY_ITER_REFS_OK
     *     - Reference types are acceptable.
     *   order NPY_KEEPORDER
     *     - Visit elements in memory order, regardless of strides.
     *       This is good for performance when the specific order
     *       elements are visited is unimportant.
     *   casting NPY_NO_CASTING
     *     - No casting is required for this operation.
     */
    iter = NpyIter_New(self, NPY_ITER_READONLY|
                             NPY_ITER_EXTERNAL_LOOP|
                             NPY_ITER_REFS_OK,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    if (iter == NULL) {
        return NULL;
    }

    /*
     * The iternext function gets stored in a local variable
     * so it can be called repeatedly in an efficient manner.
     */
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return NULL;
    }
    /* The location of the data pointer which the iterator may update */
    dataptr = NpyIter_GetDataPtrArray(iter);
    /* The location of the stride which the iterator may update */
    strideptr = NpyIter_GetInnerStrideArray(iter);
    /* The location of the inner loop size which the iterator may update */
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    nonzero_count = 0;
    do {
        /* Get the inner loop data/stride/count values */
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
        while (count--) {
            if (nonzero(data, self)) {
                ++nonzero_count;
            }
            data += stride;
        }

        /* Increment the iterator to the next inner loop */
    } while(iternext(iter));

    NpyIter_Deallocate(iter);

    return PyLong_FromLong(nonzero_count);
}

static PyMethodDef limited_api_opaque_methods[] = {
    {"nonzero", (PyCFunction)limited_api_opaque_nonzero, METH_O,
     "Count the number of non-zero elements in the array."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

PyABIInfo_VAR(abi_info);

static PyModuleDef_Slot limited_api_opaque_slots[] = {
    {Py_mod_abi, &abi_info},
    {Py_mod_name, "limited_api_opaque"},
    {Py_mod_methods, limited_api_opaque_methods},
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
    {0, NULL},
};

PyMODEXPORT_FUNC
PyModExport_limited_api_opaque(void)
{
    import_array();
    import_umath();
    return limited_api_opaque_slots;
}