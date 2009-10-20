#ifndef _NPY_CALCULATION_H_
#define _NPY_CALCULATION_H_

NPY_NO_EXPORT PyObject*
PyArray_ArgMax(PyArrayObject* self, int axis, PyArrayObject *out);

NPY_NO_EXPORT PyObject*
PyArray_ArgMin(PyArrayObject* self, int axis, PyArrayObject *out);

NPY_NO_EXPORT PyObject*
PyArray_Max(PyArrayObject* self, int axis, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Min(PyArrayObject* self, int axis, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Ptp(PyArrayObject* self, int axis, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Mean(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

NPY_NO_EXPORT PyObject *
PyArray_Round(PyArrayObject *a, int decimals, PyArrayObject *out);

NPY_NO_EXPORT PyObject*
PyArray_Trace(PyArrayObject* self, int offset, int axis1, int axis2,
                int rtype, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Clip(PyArrayObject* self, PyObject* min, PyObject* max, PyArrayObject *out);

NPY_NO_EXPORT PyObject*
PyArray_Conjugate(PyArrayObject* self, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Round(PyArrayObject* self, int decimals, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Std(PyArrayObject* self, int axis, int rtype, PyArrayObject* out,
                int variance);

NPY_NO_EXPORT PyObject *
__New_PyArray_Std(PyArrayObject *self, int axis, int rtype, PyArrayObject *out,
                  int variance, int num);

NPY_NO_EXPORT PyObject*
PyArray_Sum(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_CumSum(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Prod(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_CumProd(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_All(PyArrayObject* self, int axis, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Any(PyArrayObject* self, int axis, PyArrayObject* out);

#endif
