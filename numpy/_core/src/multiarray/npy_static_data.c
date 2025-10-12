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
#include "extobj.h"

// static variables are zero-filled by default, no need to explicitly do so
NPY_VISIBILITY_HIDDEN npy_interned_str_struct npy_interned_str;
NPY_VISIBILITY_HIDDEN npy_static_pydata_struct npy_static_pydata;
NPY_VISIBILITY_HIDDEN npy_static_cdata_struct npy_static_cdata;

#define INTERN_STRING(struct_member, string)                             \
    assert(npy_interned_str.struct_member == NULL);                      \
    npy_interned_str.struct_member = PyUnicode_InternFromString(string); \
    if (npy_interned_str.struct_member == NULL) {                        \
        return -1;                                                       \
    }                                                                    \

NPY_NO_EXPORT int
intern_strings(void)
{
    INTERN_STRING(current_allocator, "current_allocator");
    INTERN_STRING(array, "__array__");
    INTERN_STRING(array_function, "__array_function__");
    INTERN_STRING(array_struct, "__array_struct__");
    INTERN_STRING(array_priority, "__array_priority__");
    INTERN_STRING(array_interface, "__array_interface__");
    INTERN_STRING(array_ufunc, "__array_ufunc__");
    INTERN_STRING(array_wrap, "__array_wrap__");
    INTERN_STRING(array_finalize, "__array_finalize__");
    INTERN_STRING(implementation, "_implementation");
    INTERN_STRING(axis1, "axis1");
    INTERN_STRING(axis2, "axis2");
    INTERN_STRING(item, "item");
    INTERN_STRING(like, "like");
    INTERN_STRING(numpy, "numpy");
    INTERN_STRING(where, "where");
    INTERN_STRING(convert, "convert");
    INTERN_STRING(preserve, "preserve");
    INTERN_STRING(convert_if_no_array, "convert_if_no_array");
    INTERN_STRING(cpu, "cpu");
    INTERN_STRING(dtype, "dtype");
    INTERN_STRING(
            array_err_msg_substr,
            "__array__() got an unexpected keyword argument 'copy'");
    INTERN_STRING(out, "out");
    INTERN_STRING(errmode_strings[0], "ignore");
    INTERN_STRING(errmode_strings[1], "warn");
    INTERN_STRING(errmode_strings[2], "raise");
    INTERN_STRING(errmode_strings[3], "call");
    INTERN_STRING(errmode_strings[4], "print");
    INTERN_STRING(errmode_strings[5], "log");
    INTERN_STRING(__dlpack__, "__dlpack__");
    INTERN_STRING(pyvals_name, "UFUNC_PYVALS_NAME");
    INTERN_STRING(legacy, "legacy");
    INTERN_STRING(__doc__, "__doc__");
    INTERN_STRING(copy, "copy");
    INTERN_STRING(dl_device, "dl_device");
    INTERN_STRING(max_version, "max_version");
    return 0;
}

#define IMPORT_GLOBAL(base_path, name, object)  \
    assert(object == NULL);                     \
    object = npy_import(base_path, name);       \
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
    /*
     * Initialize contents of npy_static_pydata struct
     *
     * This struct holds cached references to python objects
     * that we want to keep alive for the lifetime of the
     * module for performance reasons
     */

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

    IMPORT_GLOBAL("numpy._core.printoptions", "format_options",
                  npy_static_pydata.format_options);

    IMPORT_GLOBAL("os", "fspath",
                  npy_static_pydata.os_fspath);

    IMPORT_GLOBAL("os", "PathLike",
                  npy_static_pydata.os_PathLike);

    // default_truediv_type_tup
    PyArray_Descr *tmp = PyArray_DescrFromType(NPY_DOUBLE);
    npy_static_pydata.default_truediv_type_tup =
            PyTuple_Pack(3, tmp, tmp, tmp);
    Py_DECREF(tmp);
    if (npy_static_pydata.default_truediv_type_tup == NULL) {
        return -1;
    }

    npy_static_pydata.kwnames_is_copy =
            Py_BuildValue("(O)", npy_interned_str.copy);
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

    npy_static_pydata.dl_call_kwnames =
            Py_BuildValue("(OOO)", npy_interned_str.dl_device,
                                   npy_interned_str.copy,
                                   npy_interned_str.max_version);
    if (npy_static_pydata.dl_call_kwnames == NULL) {
        return -1;
    }

    npy_static_pydata.dl_cpu_device_tuple = Py_BuildValue("(i,i)", 1, 0);
    if (npy_static_pydata.dl_cpu_device_tuple == NULL) {
        return -1;
    }

    npy_static_pydata.dl_max_version = Py_BuildValue("(i,i)", 1, 0);
    if (npy_static_pydata.dl_max_version == NULL) {
        return -1;
    }

    /*
     * Initialize contents of npy_static_cdata struct
     *
     * Note that some entries are initialized elsewhere. Care
     * must be taken to ensure all entries are initialized during
     * module initialization and immutable thereafter.
     *
     * This struct holds global static caches. These are set
     * up this way for performance reasons.
     */

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

    return 0;
}


/*
 * Verifies all entries in npy_interned_str and npy_static_pydata are
 * non-NULL.
 *
 * Called at the end of initialization for _multiarray_umath. Some
 * entries are initialized outside of this file because they depend on
 * items that are initialized late in module initialization but they
 * should all be initialized by the time this function is called.
 */
NPY_NO_EXPORT int
verify_static_structs_initialized(void) {
    // verify all entries in npy_interned_str are filled in
    for (int i=0; i < (sizeof(npy_interned_str_struct)/sizeof(PyObject *)); i++) {
        if (*(((PyObject **)&npy_interned_str) + i) == NULL) {
            PyErr_Format(
                    PyExc_SystemError,
                    "NumPy internal error: NULL entry detected in "
                    "npy_interned_str at index %d", i);
            return -1;
        }
    }

    // verify all entries in npy_static_pydata are filled in
    for (int i=0; i < (sizeof(npy_static_pydata_struct)/sizeof(PyObject *)); i++) {
        if (*(((PyObject **)&npy_static_pydata) + i) == NULL) {
            PyErr_Format(
                    PyExc_SystemError,
                    "NumPy internal error: NULL entry detected in "
                    "npy_static_pydata at index %d", i);
            return -1;
        }
    }
    return 0;
}
