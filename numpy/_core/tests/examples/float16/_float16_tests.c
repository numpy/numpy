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
float16tests_float16_eq_nonan(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int h1, h2;
    if (!PyArg_ParseTuple(args, "II", &h1, &h2)) {
        return NULL;
    }
    int ret = npy_float16_eq_nonan(h1, h2);
    return PyBool_FromLong(ret != 0);
}

static PyObject *
float16tests_float16_lt_nonan(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int h1, h2;
    if (!PyArg_ParseTuple(args, "II", &h1, &h2)) {
        return NULL;
    }
    int ret = npy_float16_lt_nonan(h1, h2);
    return PyBool_FromLong(ret != 0);
}

static PyObject *
float16tests_float16_le_nonan(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int h1, h2;
    if (!PyArg_ParseTuple(args, "II", &h1, &h2)) {
        return NULL;
    }
    int ret = npy_float16_le_nonan(h1, h2);
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

static PyObject *
float16tests_float16_divmod(PyObject *NPY_UNUSED(self), PyObject *args)
{
    unsigned int h1bits, h2bits;
    if (!PyArg_ParseTuple(args, "II", &h1bits, &h2bits)) {
        return NULL;
    }

    npy_half h1 = h1bits;
    npy_half h2 = h2bits;

    npy_half mod;
    npy_half quo = npy_float16_divmod(h1, h2, &mod);

    return Py_BuildValue("II", quo, mod);
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
    {"float16_eq_nonan", float16tests_float16_eq_nonan, METH_VARARGS,
     "h1 == h2, assuming neither is NaN."},
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
    {"float16_divmod", float16tests_float16_divmod, METH_VARARGS,
     "Return (quotient, remainder) for float16 divmod, both as half bits."},
    {NULL, NULL, 0, NULL}  /* sentinel */
};

static int
float16tests_exec(PyObject *m)
{
    Py_INCREF(m);
    if (PyModule_AddObject(m, "float16", m) < 0) {
        Py_DECREF(m);
        return -1;
    }

    return 0;
}

static PyModuleDef_Slot float16tests_slots[] = {
#ifdef Py_GIL_DISABLED
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {Py_mod_exec, float16tests_exec},
    {0, NULL}
};

static struct PyModuleDef float16tests_moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_float16_tests",
    .m_doc = "Internal float16 helper functions for testing.",
    .m_size = 0,
    .m_methods = float16_tests_methods,
    .m_slots = float16tests_slots,
};

PyMODINIT_FUNC
PyInit__float16_tests(void)
{
    return PyModuleDef_Init(&float16tests_moduledef);
}
