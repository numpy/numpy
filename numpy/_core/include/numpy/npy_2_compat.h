/*
 * This header file defines relevant features which:
 * - Require runtime inspection depending on the NumPy version.
 * - May be needed when compiling with an older version of NumPy to allow
 *   a smooth transition.
 *
 * As such, it is shipped with NumPy 2.0, but designed to be vendored in full
 * or parts by downstream projects.
 *
 * It must be included after any other includes.  `import_array()` must have
 * been called in the scope or version dependency will misbehave, even when
 * only `PyUFunc_` API is used.
 *
 * If required complicated defs (with inline functions) should be written as:
 *
 *     #if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
 *         Simple definition when NumPy 2.0 API is guaranteed.
 *     #else
 *         static inline definition of a 1.x compatibility shim
 *         #if NPY_ABI_VERSION < 0x02000000
 *            Make 1.x compatibility shim the public API (1.x only branch)
 *         #else
 *             Runtime dispatched version (1.x or 2.x)
 *         #endif
 *     #endif
 *
 * An internal build always passes NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
 */

#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPAT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPAT_H_

/*
 * Allow users to use `PyArray_RUNTIME_VERSION` when vendoring the file for
 * compilation with NumPy 1.x.
 * Simply do not define when compiling with 2.x.  It must be defined later
 * as it is set during `import_array()`.
 */
#if !defined(PyArray_RUNTIME_VERSION) && NPY_ABI_VERSION < 0x02000000
  /*
   * If we are compiling with NumPy 1.x, PyArray_RUNTIME_VERSION so we
   * pretend the `PyArray_RUNTIME_VERSION` is `NPY_FEATURE_VERSION`.
   */
  #define PyArray_RUNTIME_VERSION NPY_FEATURE_VERSION
#endif

/*
 * New macros for accessing real and complex part of a complex number can be
 * found in "npy_2_complexcompat.h".
 */


/*
 * NPY_DEFAULT_INT
 *
 * The default integer has changed, `NPY_DEFAULT_INT` is available at runtime
 * for use as type number, e.g. `PyArray_DescrFromType(NPY_DEFAULT_INT)`.
 *
 * NPY_RAVEL_AXIS
 *
 * This was introduced in NumPy 2.0 to allow indicating that an axis should be
 * raveled in an operation. Before NumPy 2.0, NPY_MAXDIMS was used for this purpose.
 *
 * NPY_MAXDIMS
 *
 * A constant indicating the maximum number dimensions allowed when creating
 * an ndarray.
 *
 * NPY_NTYPES
 *
 * The number of built-in NumPy dtypes.
 */
#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
    #define NPY_DEFAULT_INT NPY_INTP
    #define NPY_RAVEL_AXIS NPY_MIN_INT
    #define NPY_MAXARGS 64
    #define NPY_NTYPES 24
#elif NPY_ABI_VERSION < 0x02000000
    #define NPY_DEFAULT_INT NPY_LONG
    #define NPY_RAVEL_AXIS 32
    #define NPY_MAXARGS 32
    #define NPY_NTYPES 25
#else
    #define NPY_DEFAULT_INT  \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ? NPY_INTP : NPY_LONG)
    #define NPY_RAVEL_AXIS  \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ? -1 : 32)
    #define NPY_MAXARGS  \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ? 64 : 32)
    #define NPY_NTYPES  \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ? 25 : 24)
#endif

/*
 * The descriptor struct layout was changed in 2.x and for some
 * fields it is necessary to look up the values based on the runtime version.
 * We do not list all possible access macros as most shouldn't be needed.
 */
#if 0
#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
    /* No backcompat necessary, direct access */
    //#define PyDataType_ITEMSIZE(descr) (descr)->itemsize
    #define PyDataType_SET_ITEMSIZE(descr, value) (descr)->itemsize = value
    #define PyDataType_ITEMSIZE(descr) (descr)->alignment
    #define PyDataType_FIELDS(descr)->fields

#elif NPY_ABI_VERSION < 0x02000000
    /* old version of NumPy, only the name changed */
    //#define PyDataType_ITEMSIZE(descr) (descr)->elsize
    #define PyDataType_SET_ITEMSIZE(descr, value) (descr)->elsize = value
    #define PyDataType_ITEMSIZE(descr) (descr)->alignment
    #define PyDataType_FIELDS(descr) (descr)->fields
    #define PyDataType_NAMES(descr) (descr)->names
    #define PyDataType_METADATA(descr) (descr)->metadata

#else
    #define PyDataType_ITEMSIZE(descr)                     \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ?  \
            (((PyArray_DescrNew *)descr)->itemsize) :      \
            (npy_intp)((PyArray_DescrBase *)(descr))->elsize )

    #define PyDataType_SET_ITEMSIZE(descr, value)              \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ?      \
            (((PyArray_DescrNew *)descr)->itemsize = value) :  \
            (npy_intp)((PyArray_DescrBase *)(descr))->elsize = value )

    #define PyDataType_ITEMSIZE(descr)                     \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ?  \
            (((PyArray_DescrNew *)descr)->alignment) :     \
            (npy_intp)((PyArray_DescrBase *)(descr))->alignment )

    #define PyDataType_ITEMSIZE(descr)                     \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ?  \
            _PyDataType_Itemsize(descr) : (npy_intp)(PyArray_DescrBase *)(descr)->elsize )

    #define PyDataType_ITEMSIZE(descr)                     \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ?  \
            _PyDataType_Itemsize(descr) : (npy_intp)(PyArray_DescrBase *)(descr)->elsize )

#endif
#endif

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPAT_H_ */
