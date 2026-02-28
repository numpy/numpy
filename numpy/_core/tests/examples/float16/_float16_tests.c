#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/float16.h"

/*
 * Conversions
 */
static PyObject *
float16tests_float16_to_float(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int hbits;
    if (!PyArg_ParseTuple(args, "I", &hbits)) {
        return NULL;
    }

    npy_half h = hbits;
    float f = npy_float16_to_float(h);
    return PyFloat_FromDouble(f);
}

static PyObject *
float16tests_float16_to_double(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int hbits;
    if (!PyArg_ParseTuple(args, "I", &hbits)) {
        return NULL;
    }

    npy_half h = hbits;
    double d = npy_float16_to_double(h);
    return PyFloat_FromDouble(d);
}

static PyObject *
float16tests_double_to_float16(PyObject *NPY_UNUSED(self), PyObject *args)
{
    double d;
    if (!PyArg_ParseTuple(args, "d", &d)) {
        return NULL;
    }
    npy_half h = npy_double_to_float16(d);
    return PyLong_FromUnsignedLong(h);
}

static PyObject *
float16tests_float_to_float16(PyObject *NPY_UNUSED(self), PyObject *args)
{
    double f;
    if (!PyArg_ParseTuple(args, "d", &f)) {
        return NULL;
    }

    npy_half h = npy_float_to_float16(f);
    return PyLong_FromUnsignedLong(h);
}

/*
 * Comparisons
 */
static PyObject *
float16tests_float16_eq(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int h1, h2;
    if (!PyArg_ParseTuple(args, "II", &h1, &h2)) {
        return NULL;
    }
    int ret = npy_float16_eq(h1,h2);
    return PyBool_FromLong(ret != 0);
}

static PyObject *
float16tests_float16_ne(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int h1, h2;
    if (!PyArg_ParseTuple(args, "II", &h1, &h2)) {
        return NULL;
    }
    int ret = npy_float16_ne(h1,h2);
    return PyBool_FromLong(ret != 0);
}

static PyObject *
float16tests_float16_le(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int h1, h2;
    if (!PyArg_ParseTuple(args, "II", &h1, &h2)) {
        return NULL;
    }
    int ret = npy_float16_le(h1, h2);
    return PyBool_FromLong(ret != 0);
}

static PyObject *
float16tests_float16_lt(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int h1, h2;
    if (!PyArg_ParseTuple(args, "II", &h1, &h2)) {
        return NULL;
    }
    int ret = npy_float16_lt(h1,h2);
    return PyBool_FromLong(ret != 0);
}

static PyObject *
float16tests_float16_ge(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int h1, h2;
    if (!PyArg_ParseTuple(args, "II", &h1, &h2)) {
        return NULL;
    }
    int ret = npy_float16_ge(h1, h2);
    return PyBool_FromLong(ret != 0);
}

static PyObject *
float16tests_float16_gt(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int h1, h2;
    if (!PyArg_ParseTuple(args, "II", &h1, &h2)) {
        return NULL;
    }
    int ret = npy_float16_gt(h1, h2);
    return PyBool_FromLong(ret != 0);
}

/*
 * No Nan Comparisons variants
 */

static PyObject *
float16tests_float16_lt_nonan(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int h1, h2;
    if (!PyArg_ParseTuple(args, "II", &h1, &h2)) {
        return NULL;
    }
    int ret = _npy_float16_lt_nonan(h1, h2);
    return PyBool_FromLong(ret != 0);
}

static PyObject *
float16tests_float16_le_nonan(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int h1, h2;
    if (!PyArg_ParseTuple(args, "II", &h1, &h2)) {
        return NULL;
    }
    int ret = _npy_float16_le_nonan(h1, h2);
    return PyBool_FromLong(ret != 0);
}

/*
 * Miscellaneous Functions
 */
static PyObject *
float16tests_float16_iszero(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int hbits;
    if (!PyArg_ParseTuple(args, "I", &hbits)) {
        return NULL;
    }
    int ret = npy_float16_iszero(hbits);
    return PyBool_FromLong(ret != 0);
}

static PyObject *
float16tests_float16_isnan(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int hbits;
    if (!PyArg_ParseTuple(args, "I", &hbits)) {
        return NULL;
    }
    int ret = npy_float16_isnan(hbits);
    return PyBool_FromLong(ret != 0);
}

static PyObject *
float16tests_float16_isinf(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int hbits;
    if (!PyArg_ParseTuple(args, "I", &hbits)) {
        return NULL;
    }
    int ret = npy_float16_isinf(hbits);
    return PyBool_FromLong(ret != 0);
}

static PyObject *
float16tests_float16_isfinite(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int hbits;
    if (!PyArg_ParseTuple(args, "I", &hbits)) {
        return NULL;
    }
    int ret = npy_float16_isfinite(hbits);
    return PyBool_FromLong(ret != 0);
}

static PyObject *
float16tests_float16_signbit(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int hbits;
    if (!PyArg_ParseTuple(args, "I", &hbits)) {
        return NULL;
    }
    int ret = npy_float16_signbit(hbits);
    return PyBool_FromLong(ret != 0);
}

static PyObject *
float16tests_float16_copysign(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int xbits, ybits;
    if (!PyArg_ParseTuple(args, "II", &xbits, &ybits)) {
        return NULL;
    }

    npy_half x = xbits;
    npy_half y = ybits;

    npy_half r = npy_float16_copysign(x, y);
    return PyLong_FromUnsignedLong(r);
}

static PyObject *
float16tests_float16_nextafter(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int xbits, ybits;
    if (!PyArg_ParseTuple(args, "II", &xbits, &ybits)) {
        return NULL;
    }

    npy_half x = xbits;
    npy_half y = ybits;

    npy_half r = npy_float16_nextafter(x, y);
    return PyLong_FromUnsignedLong(r);
}

static PyMethodDef float16_tests_methods[] = {
    // Conversions
    {"float16_to_float", float16tests_float16_to_float, METH_VARARGS,
    "Convert float16 to float32"},
    {"float16_to_double", float16tests_float16_to_double,METH_VARARGS,
    "Convert float16 to double (Python float, double precision) ."},
    {"double_to_float16", float16tests_double_to_float16, METH_VARARGS,
    "Convert float64 (Python float) to float16 bits (uint16)."},
    {"float_to_float16", float16tests_float_to_float16, METH_VARARGS,
    "Convert float32 (Python float) to float16 bits (uint16)."},
    // Comparisons
    {"float16_eq", float16tests_float16_eq, METH_VARARGS, "h1 == h2 (with NaN handling)."},
    {"float16_ne", float16tests_float16_ne, METH_VARARGS, "h1 != h2 (with NaN handling)."},
    {"float16_le", float16tests_float16_le, METH_VARARGS, "h1 <= h2 (with NaN handling)."},
    {"float16_lt", float16tests_float16_lt, METH_VARARGS, "h1 < h2 (with NaN handling)."},
    {"float16_ge", float16tests_float16_ge, METH_VARARGS, "h1 >= h2 (with NaN handling)."},
    {"float16_gt", float16tests_float16_gt, METH_VARARGS, "h1 > h2 (with NaN handling)."},
    // No Nan Comparsons
    {"float16_lt_nonan", float16tests_float16_lt_nonan, METH_VARARGS,
     "h1 < h2, assuming neither is NaN."},
    {"float16_le_nonan", float16tests_float16_le_nonan, METH_VARARGS,
     "h1 <= h2, assuming neither is NaN."},
    // "Is" Functions
    {"float16_iszero", float16tests_float16_iszero, METH_VARARGS, "Is half zero (+/-0)."},
    {"float16_isnan", float16tests_float16_isnan, METH_VARARGS, "Is half NaN."},
    {"float16_isinf", float16tests_float16_isinf, METH_VARARGS, "Is half infinite."},
    {"float16_isfinite", float16tests_float16_isfinite, METH_VARARGS, "Is half finite."},
    {"float16_signbit", float16tests_float16_signbit, METH_VARARGS, "Signbit of half."},
    // Miscellaneous Functions
    {"float16_copysign", float16tests_float16_copysign, METH_VARARGS,
     "Return half with magnitude of x and sign of y."},
    {"float16_nextafter", float16tests_float16_nextafter, METH_VARARGS,
     "Return next representable half from x toward y (as half bits)."},
    {NULL, NULL, 0, NULL}  /* sentinel */
};

static struct PyModuleDef float16tests_moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_float16_tests",
    .m_doc = "Internal float16 helper functions for testing.",
    .m_size = 0,
    .m_methods = float16_tests_methods,
};

PyMODINIT_FUNC
PyInit__float16_tests(void)
{
    PyObject *m = PyModule_Create(&float16tests_moduledef);
    if (m == NULL) {
        return NULL;
    }

#ifdef Py_GIL_DISABLED
    if (PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED) < 0) {
        Py_DECREF(m);
        return NULL;
    }
#endif

    return m;
}
