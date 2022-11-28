#ifndef NUMPY_CORE_SRC_MULTIARRAY_REFCOUNT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_REFCOUNT_H_


NPY_NO_EXPORT int
PyArray_ClearBuffer(
        PyArray_Descr *descr, char *data,
        npy_intp stride, npy_intp size, int aligned);

NPY_NO_EXPORT int
PyArray_ClearArray(PyArrayOBject *arr);

/*
 * The following functions worke directly on object or structured dtypes.
 * Their use is generally incorrect.  In all cases the above functions should
 * be used.
 * In some old code, these were (are still) used by copying data first and
 * calling INCREF in hindsight.  This use is generally incorrect since it
 * does not translate to user DTypes.
 * The code should rather use `PyArray_GetStridedCopyFn()` (or copyswapn
 * as a start, although that probably needs replacement for new DTypes as
 * well).  (Comment as of NumPy 1.25)
 */

NPY_NO_EXPORT void
PyArray_Item_INCREF(char *data, PyArray_Descr *descr);

NPY_NO_EXPORT void
PyArray_Item_XDECREF(char *data, PyArray_Descr *descr);

NPY_NO_EXPORT int
PyArray_INCREF(PyArrayObject *mp);

NPY_NO_EXPORT int
PyArray_XDECREF(PyArrayObject *mp);

NPY_NO_EXPORT void
PyArray_FillObjectArray(PyArrayObject *arr, PyObject *obj);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_REFCOUNT_H_ */
