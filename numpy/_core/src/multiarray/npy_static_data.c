/* numpy static data structs and initialization */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _UMATHMODULE
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include "numpy/arrayobject.h"
#include "npy_import.h"
#include "npy_static_data.h"

// static variables are zero-filled by default, no need to explicitly do so
NPY_VISIBILITY_HIDDEN npy_interned_str_struct npy_interned_str;
NPY_VISIBILITY_HIDDEN npy_static_pydata_struct npy_static_pydata;
NPY_VISIBILITY_HIDDEN npy_static_cdata_struct npy_static_cdata;

NPY_NO_EXPORT int
intern_strings(void)
{
    npy_interned_str.current_allocator = PyUnicode_InternFromString("current_allocator");
    if (npy_interned_str.current_allocator == NULL) {
        return -1;
    }
    npy_interned_str.array = PyUnicode_InternFromString("__array__");
    if (npy_interned_str.array == NULL) {
        return -1;
    }
    npy_interned_str.array_function = PyUnicode_InternFromString("__array_function__");
    if (npy_interned_str.array_function == NULL) {
        return -1;
    }
    npy_interned_str.array_struct = PyUnicode_InternFromString("__array_struct__");
    if (npy_interned_str.array_struct == NULL) {
        return -1;
    }
    npy_interned_str.array_priority = PyUnicode_InternFromString("__array_priority__");
    if (npy_interned_str.array_priority == NULL) {
        return -1;
    }
    npy_interned_str.array_interface = PyUnicode_InternFromString("__array_interface__");
    if (npy_interned_str.array_interface == NULL) {
        return -1;
    }
    npy_interned_str.array_ufunc = PyUnicode_InternFromString("__array_ufunc__");
    if (npy_interned_str.array_ufunc == NULL) {
        return -1;
    }
    npy_interned_str.array_wrap = PyUnicode_InternFromString("__array_wrap__");
    if (npy_interned_str.array_wrap == NULL) {
        return -1;
    }
    npy_interned_str.array_finalize = PyUnicode_InternFromString("__array_finalize__");
    if (npy_interned_str.array_finalize == NULL) {
        return -1;
    }
    npy_interned_str.implementation = PyUnicode_InternFromString("_implementation");
    if (npy_interned_str.implementation == NULL) {
        return -1;
    }
    npy_interned_str.axis1 = PyUnicode_InternFromString("axis1");
    if (npy_interned_str.axis1 == NULL) {
        return -1;
    }
    npy_interned_str.axis2 = PyUnicode_InternFromString("axis2");
    if (npy_interned_str.axis2 == NULL) {
        return -1;
    }
    npy_interned_str.like = PyUnicode_InternFromString("like");
    if (npy_interned_str.like == NULL) {
        return -1;
    }
    npy_interned_str.numpy = PyUnicode_InternFromString("numpy");
    if (npy_interned_str.numpy == NULL) {
        return -1;
    }
    npy_interned_str.where = PyUnicode_InternFromString("where");
    if (npy_interned_str.where == NULL) {
        return -1;
    }
    npy_interned_str.convert = PyUnicode_InternFromString("convert");
    if (npy_interned_str.convert == NULL) {
        return -1;
    }
    npy_interned_str.preserve = PyUnicode_InternFromString("preserve");
    if (npy_interned_str.preserve == NULL) {
        return -1;
    }
    npy_interned_str.convert_if_no_array = PyUnicode_InternFromString("convert_if_no_array");
    if (npy_interned_str.convert_if_no_array == NULL) {
        return -1;
    }
    npy_interned_str.cpu = PyUnicode_InternFromString("cpu");
    if (npy_interned_str.cpu == NULL) {
        return -1;
    }
    npy_interned_str.dtype = PyUnicode_InternFromString("dtype");
    if (npy_interned_str.dtype == NULL) {
        return -1;
    }
    npy_interned_str.array_err_msg_substr = PyUnicode_InternFromString(
            "__array__() got an unexpected keyword argument 'copy'");
    if (npy_interned_str.array_err_msg_substr == NULL) {
        return -1;
    }
    npy_interned_str.out = PyUnicode_InternFromString("out");
    if (npy_interned_str.out == NULL) {
        return -1;
    }
    npy_interned_str.__dlpack__ = PyUnicode_InternFromString("__dlpack__");
    if (npy_interned_str.__dlpack__ == NULL) {
        return -1;
    }
    npy_interned_str.pyvals_name = PyUnicode_InternFromString("UFUNC_PYVALS_NAME");
    if (npy_interned_str.pyvals_name == NULL) {
        return -1;
    }
    return 0;
}

#define IMPORT_GLOBAL(base_path, name, object)  \
    assert(object == NULL);                     \
    npy_cache_import(base_path, name, &object); \
    if (object == NULL) {                       \
        return -1;                              \
    }


/*
 * Initializes global constants.
 *
 * All global constants should live inside the npy_static_pydata
 * struct.
 *
 * Not all entries in the struct are initialized here, some are
 * initialized later but care must be taken in those cases to initialize
 * the constant in a thread-safe manner, ensuring it is initialized
 * exactly once.
 *
 * Anything initialized here is initialized during module import which
 * the python interpreter ensures is done in a single thread.
 *
 * Anything imported here should not need the C-layer at all and will be
 * imported before anything on the C-side is initialized.
 */
NPY_NO_EXPORT int
initialize_static_globals(void)
{
    // cached reference to objects defined in python

    IMPORT_GLOBAL("math", "floor",
                  npy_static_pydata.math_floor_func);

    IMPORT_GLOBAL("math", "ceil",
                  npy_static_pydata.math_ceil_func);

    IMPORT_GLOBAL("math", "trunc",
                  npy_static_pydata.math_trunc_func);

    IMPORT_GLOBAL("math", "gcd",
                  npy_static_pydata.math_gcd_func);

    IMPORT_GLOBAL("numpy.exceptions", "AxisError",
                  npy_static_pydata.AxisError);

    IMPORT_GLOBAL("numpy.exceptions", "ComplexWarning",
                  npy_static_pydata.ComplexWarning);

    IMPORT_GLOBAL("numpy.exceptions", "DTypePromotionError",
                  npy_static_pydata.DTypePromotionError);

    IMPORT_GLOBAL("numpy.exceptions", "TooHardError",
                  npy_static_pydata.TooHardError);

    IMPORT_GLOBAL("numpy.exceptions", "VisibleDeprecationWarning",
                  npy_static_pydata.VisibleDeprecationWarning);

    IMPORT_GLOBAL("numpy._globals", "_CopyMode",
                  npy_static_pydata._CopyMode);

    IMPORT_GLOBAL("numpy._globals", "_NoValue",
                  npy_static_pydata._NoValue);

    IMPORT_GLOBAL("numpy._core._exceptions", "_ArrayMemoryError",
                  npy_static_pydata._ArrayMemoryError);

    IMPORT_GLOBAL("numpy._core._exceptions", "_UFuncBinaryResolutionError",
                  npy_static_pydata._UFuncBinaryResolutionError);

    IMPORT_GLOBAL("numpy._core._exceptions", "_UFuncInputCastingError",
                  npy_static_pydata._UFuncInputCastingError);

    IMPORT_GLOBAL("numpy._core._exceptions", "_UFuncNoLoopError",
                  npy_static_pydata._UFuncNoLoopError);

    IMPORT_GLOBAL("numpy._core._exceptions", "_UFuncOutputCastingError",
                  npy_static_pydata._UFuncOutputCastingError);

    IMPORT_GLOBAL("os", "fspath",
                  npy_static_pydata.os_fspath);

    IMPORT_GLOBAL("os", "PathLike",
                  npy_static_pydata.os_PathLike);

    // default_truediv_type_tupS
    PyArray_Descr *tmp = PyArray_DescrFromType(NPY_DOUBLE);
    if (tmp == NULL) {
        return -1;
    }

    npy_static_pydata.default_truediv_type_tup =
            PyTuple_Pack(3, tmp, tmp, tmp);
    if (npy_static_pydata.default_truediv_type_tup == NULL) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    PyObject *flags = PySys_GetObject("flags");  /* borrowed object */
    if (flags == NULL) {
        PyErr_SetString(PyExc_AttributeError, "cannot get sys.flags");
        return -1;
    }
    PyObject *level = PyObject_GetAttrString(flags, "optimize");
    if (level == NULL) {
        return -1;
    }
    npy_static_cdata.optimize = PyLong_AsLong(level);
    Py_DECREF(level);

    /*
     * see unpack_bits for how this table is used.
     *
     * LUT for bigendian bitorder, littleendian is handled via
     * byteswapping in the loop.
     *
     * 256 8 byte blocks representing 8 bits expanded to 1 or 0 bytes
     */
    npy_intp j;
    for (j=0; j < 256; j++) {
        npy_intp k;
        for (k=0; k < 8; k++) {
            npy_uint8 v = (j & (1 << k)) == (1 << k);
            npy_static_cdata.unpack_lookup_big[j].bytes[7 - k] = v;
        }
    }

    npy_static_pydata.kwnames_is_copy = Py_BuildValue("(s)", "copy");
    if (npy_static_pydata.kwnames_is_copy == NULL) {
        return -1;
    }

    npy_static_pydata.one_obj = PyLong_FromLong((long) 1);
    if (npy_static_pydata.one_obj == NULL) {
        return -1;
    }

    npy_static_pydata.zero_obj = PyLong_FromLong((long) 0);
    if (npy_static_pydata.zero_obj == NULL) {
        return -1;
    }

    return 0;
}

  
