#ifndef Py_TARGET_ABI3T
#error "This file must be compiled with -DPy_TARGET_ABI3T"
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

/*
 * Test PyArray_ITER_NEXT, PyArray_ITER_RESET, PyArray_ITER_DATA,
 * and PyArray_ITER_NOTDONE by summing all elements using the
 * legacy iterator macros.
 */
static PyObject *
limited_api_opaque_iter_next(PyObject *mod, PyArrayObject *self)
{
    PyObject *iter_obj = PyArray_IterNew((PyObject *)self);
    if (iter_obj == NULL) {
        return NULL;
    }
    double sum = 0.0;
    while (PyArray_ITER_NOTDONE(iter_obj)) {
        sum += *(double *)PyArray_ITER_DATA(iter_obj);
        PyArray_ITER_NEXT(iter_obj);
    }
    Py_DECREF(iter_obj);
    return PyFloat_FromDouble(sum);
}

/*
 * Test PyArray_ITER_GOTO1D by accessing a specific flat index.
 */
static PyObject *
limited_api_opaque_iter_goto1d(PyObject *mod, PyObject *args)
{
    PyArrayObject *arr;
    npy_intp index;
    if (!PyArg_ParseTuple(args, "O!n", &PyArray_Type, &arr, &index)) {
        return NULL;
    }
    PyObject *iter_obj = PyArray_IterNew((PyObject *)arr);
    if (iter_obj == NULL) {
        return NULL;
    }
    PyArray_ITER_GOTO1D(iter_obj, index);
    double val = *(double *)PyArray_ITER_DATA(iter_obj);
    Py_DECREF(iter_obj);
    return PyFloat_FromDouble(val);
}

/*
 * Test PyArray_ITER_RESET by iterating, resetting, and iterating again.
 * Returns the sum from the second pass (should equal the first).
 */
static PyObject *
limited_api_opaque_iter_reset(PyObject *mod, PyArrayObject *self)
{
    PyObject *iter_obj = PyArray_IterNew((PyObject *)self);
    if (iter_obj == NULL) {
        return NULL;
    }
    /* First pass: skip through */
    while (PyArray_ITER_NOTDONE(iter_obj)) {
        PyArray_ITER_NEXT(iter_obj);
    }
    /* Reset and sum */
    PyArray_ITER_RESET(iter_obj);
    double sum = 0.0;
    while (PyArray_ITER_NOTDONE(iter_obj)) {
        sum += *(double *)PyArray_ITER_DATA(iter_obj);
        PyArray_ITER_NEXT(iter_obj);
    }
    Py_DECREF(iter_obj);
    return PyFloat_FromDouble(sum);
}

/*
 * Test PyArray_MultiIter_NEXT, PyArray_MultiIter_RESET,
 * PyArray_MultiIter_DATA, and PyArray_MultiIter_NOTDONE
 * by computing the element-wise sum of two broadcastable arrays.
 * Returns the total sum of (a + b) for all broadcast elements.
 */
static PyObject *
limited_api_opaque_multi_iter_next(PyObject *mod, PyObject *args)
{
    PyArrayObject *a, *b;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a,
                          &PyArray_Type, &b)) {
        return NULL;
    }
    PyObject *multi = PyArray_MultiIterNew(2, a, b);
    if (multi == NULL) {
        return NULL;
    }
    double sum = 0.0;
    while (PyArray_MultiIter_NOTDONE(multi)) {
        double va = *(double *)PyArray_MultiIter_DATA(multi, 0);
        double vb = *(double *)PyArray_MultiIter_DATA(multi, 1);
        sum += va + vb;
        PyArray_MultiIter_NEXT(multi);
    }
    /* Test reset: iterate again and verify same sum */
    PyArray_MultiIter_RESET(multi);
    double sum2 = 0.0;
    while (PyArray_MultiIter_NOTDONE(multi)) {
        double va = *(double *)PyArray_MultiIter_DATA(multi, 0);
        double vb = *(double *)PyArray_MultiIter_DATA(multi, 1);
        sum2 += va + vb;
        PyArray_MultiIter_NEXT(multi);
    }
    Py_DECREF(multi);
    if (sum != sum2) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MultiIter reset produced different sum");
        return NULL;
    }
    return PyFloat_FromDouble(sum);
}

/*
 * Test PyArray_ITER_GOTO by jumping to a coordinate and reading the value.
 */
static PyObject *
limited_api_opaque_iter_goto(PyObject *mod, PyObject *args)
{
    PyArrayObject *arr;
    PyObject *coord_tuple;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &arr,
                          &PyTuple_Type, &coord_tuple)) {
        return NULL;
    }
    int nd = PyArray_NDIM(arr);
    if (PyTuple_Size(coord_tuple) != nd) {
        PyErr_SetString(PyExc_ValueError, "coordinate length mismatch");
        return NULL;
    }
    npy_intp destination[NPY_MAXDIMS_LEGACY_ITERS];
    for (int i = 0; i < nd; i++) {
        destination[i] = PyLong_AsLong(PyTuple_GetItem(coord_tuple, i));
        if (destination[i] == -1 && PyErr_Occurred()) {
            return NULL;
        }
    }
    PyObject *iter_obj = PyArray_IterNew((PyObject *)arr);
    if (iter_obj == NULL) {
        return NULL;
    }
    PyArray_ITER_GOTO(iter_obj, destination);
    double val = *(double *)PyArray_ITER_DATA(iter_obj);
    Py_DECREF(iter_obj);
    return PyFloat_FromDouble(val);
}

/*
 * Test PyArray_MultiIter_GOTO by jumping to a coordinate
 * and returning (a_val, b_val) at that position.
 */
static PyObject *
limited_api_opaque_multi_iter_goto(PyObject *mod, PyObject *args)
{
    PyArrayObject *a, *b;
    PyObject *coord_tuple;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &a,
                          &PyArray_Type, &b,
                          &PyTuple_Type, &coord_tuple)) {
        return NULL;
    }
    PyObject *multi = PyArray_MultiIterNew(2, a, b);
    if (multi == NULL) {
        return NULL;
    }
    int nd = _PyMIT(multi)->nd;
    if (PyTuple_Size(coord_tuple) != nd) {
        Py_DECREF(multi);
        PyErr_SetString(PyExc_ValueError, "coordinate length mismatch");
        return NULL;
    }
    npy_intp destination[NPY_MAXDIMS_LEGACY_ITERS];
    for (int i = 0; i < nd; i++) {
        destination[i] = PyLong_AsLong(PyTuple_GetItem(coord_tuple, i));
        if (destination[i] == -1 && PyErr_Occurred()) {
            Py_DECREF(multi);
            return NULL;
        }
    }
    PyArray_MultiIter_GOTO(multi, destination);
    double va = *(double *)PyArray_MultiIter_DATA(multi, 0);
    double vb = *(double *)PyArray_MultiIter_DATA(multi, 1);
    Py_DECREF(multi);
    return Py_BuildValue("dd", va, vb);
}

/*
 * Test PyArray_MultiIter_GOTO1D by jumping to a flat index
 * and returning (a_val, b_val) at that position.
 */
static PyObject *
limited_api_opaque_multi_iter_goto1d(PyObject *mod, PyObject *args)
{
    PyArrayObject *a, *b;
    npy_intp index;
    if (!PyArg_ParseTuple(args, "O!O!n", &PyArray_Type, &a,
                          &PyArray_Type, &b, &index)) {
        return NULL;
    }
    PyObject *multi = PyArray_MultiIterNew(2, a, b);
    if (multi == NULL) {
        return NULL;
    }
    PyArray_MultiIter_GOTO1D(multi, index);
    double va = *(double *)PyArray_MultiIter_DATA(multi, 0);
    double vb = *(double *)PyArray_MultiIter_DATA(multi, 1);
    Py_DECREF(multi);
    return Py_BuildValue("dd", va, vb);
}

/*
 * Test PyArray_MultiIter_NEXTi by advancing only the first iterator
 * and returning its data pointer value after N steps.
 */
static PyObject *
limited_api_opaque_multi_iter_nexti(PyObject *mod, PyObject *args)
{
    PyArrayObject *a, *b;
    int steps;
    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &a,
                          &PyArray_Type, &b, &steps)) {
        return NULL;
    }
    PyObject *multi = PyArray_MultiIterNew(2, a, b);
    if (multi == NULL) {
        return NULL;
    }
    for (int i = 0; i < steps; i++) {
        PyArray_MultiIter_NEXTi(multi, 0);
    }
    double val = *(double *)PyArray_MultiIter_DATA(multi, 0);
    Py_DECREF(multi);
    return PyFloat_FromDouble(val);
}

static PyMethodDef limited_api_opaque_methods[] = {
    {"nonzero", (PyCFunction)limited_api_opaque_nonzero, METH_O,
     "Count the number of non-zero elements in the array."},
    {"iter_next", (PyCFunction)limited_api_opaque_iter_next, METH_O,
     "Sum array elements using PyArray_ITER_NEXT."},
    {"iter_goto1d", (PyCFunction)limited_api_opaque_iter_goto1d, METH_VARARGS,
     "Get element at flat index using PyArray_ITER_GOTO1D."},
    {"iter_reset", (PyCFunction)limited_api_opaque_iter_reset, METH_O,
     "Sum array elements after reset using PyArray_ITER_RESET."},
    {"multi_iter_next", (PyCFunction)limited_api_opaque_multi_iter_next,
     METH_VARARGS,
     "Sum broadcast (a+b) using PyArray_MultiIter_NEXT."},
    {"iter_goto", (PyCFunction)limited_api_opaque_iter_goto, METH_VARARGS,
     "Get element at coordinate using PyArray_ITER_GOTO."},
    {"multi_iter_goto", (PyCFunction)limited_api_opaque_multi_iter_goto,
     METH_VARARGS,
     "Get (a, b) at coordinate using PyArray_MultiIter_GOTO."},
    {"multi_iter_goto1d", (PyCFunction)limited_api_opaque_multi_iter_goto1d,
     METH_VARARGS,
     "Get (a, b) at flat index using PyArray_MultiIter_GOTO1D."},
    {"multi_iter_nexti", (PyCFunction)limited_api_opaque_multi_iter_nexti,
     METH_VARARGS,
     "Advance only iter 0 N steps using PyArray_MultiIter_NEXTi."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

PyABIInfo_VAR(abi_info);

static PySlot limited_api_opaque_slots[] = {
    PySlot_STATIC_DATA(Py_mod_abi, &abi_info),
    PySlot_STATIC_DATA(Py_mod_name, "limited_api_opaque"),
    PySlot_STATIC_DATA(Py_mod_methods, limited_api_opaque_methods),
    PySlot_STATIC_DATA(Py_mod_gil, Py_MOD_GIL_NOT_USED),
    PySlot_END,
};

PyMODEXPORT_FUNC
PyModExport_limited_api_opaque(void)
{
    import_array();
    import_umath();
    return limited_api_opaque_slots;
}
