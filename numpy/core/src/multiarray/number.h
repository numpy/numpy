#ifndef _NPY_ARRAY_NUMBER_H_
#define _NPY_ARRAY_NUMBER_H_

typedef struct {
    PyObject *add;
    PyObject *subtract;
    PyObject *multiply;
    PyObject *divide;
    PyObject *remainder;
    PyObject *divmod;
    PyObject *power;
    PyObject *square;
    PyObject *reciprocal;
    PyObject *_ones_like;
    PyObject *sqrt;
    PyObject *cbrt;
    PyObject *negative;
    PyObject *positive;
    PyObject *absolute;
    PyObject *invert;
    PyObject *left_shift;
    PyObject *right_shift;
    PyObject *bitwise_and;
    PyObject *bitwise_xor;
    PyObject *bitwise_or;
    PyObject *less;
    PyObject *less_equal;
    PyObject *equal;
    PyObject *not_equal;
    PyObject *greater;
    PyObject *greater_equal;
    PyObject *floor_divide;
    PyObject *true_divide;
    PyObject *logical_or;
    PyObject *logical_and;
    PyObject *floor;
    PyObject *ceil;
    PyObject *maximum;
    PyObject *minimum;
    PyObject *rint;
    PyObject *conjugate;
} NumericOps;

extern NPY_NO_EXPORT NumericOps n_ops;
extern NPY_NO_EXPORT PyNumberMethods array_as_number;

NPY_NO_EXPORT PyObject *
array_int(PyArrayObject *v);

NPY_NO_EXPORT int
PyArray_SetNumericOps(PyObject *dict);

NPY_NO_EXPORT PyObject *
PyArray_GetNumericOps(void);

NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyArrayObject *m1, PyObject *m2, PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out);

NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out);

#endif
