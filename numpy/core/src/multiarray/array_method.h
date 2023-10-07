#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAY_METHOD_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAY_METHOD_H_

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>
#include <numpy/ndarraytypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "numpy/_dtype_api.h"


/*
 * It would be nice to just | flags, but in general it seems that 0 bits
 * probably should indicate "default".
 * And that is not necessarily compatible with `|`.
 *
 * NOTE: If made public, should maybe be a function to easier add flags?
 */
#define PyArrayMethod_MINIMAL_FLAGS NPY_METH_NO_FLOATINGPOINT_ERRORS
#define PyArrayMethod_COMBINED_FLAGS(flags1, flags2)  \
        ((NPY_ARRAYMETHOD_FLAGS)(  \
            ((flags1 | flags2) & ~PyArrayMethod_MINIMAL_FLAGS)  \
            | (flags1 & flags2)))

/*
 * Structure of the ArrayMethod. This structure should probably not be made
 * public. If necessary, we can make certain operations on it public
 * (e.g. to allow users indirect access to `get_strided_loop`).
 *
 * NOTE: In some cases, it may not be clear whether information should be
 * stored here or on the bound version. E.g. `nin` and `nout` (and in the
 * future the gufunc `signature`) is already stored on the ufunc so that
 * storing these here duplicates the information.
 */
typedef struct PyArrayMethodObject_tag {
    PyObject_HEAD
    char *name;
    int nin, nout;
    /* Casting is normally "safe" for functions, but is important for casts */
    NPY_CASTING casting;
    /* default flags. The get_strided_loop function can override these */
    NPY_ARRAYMETHOD_FLAGS flags;
    resolve_descriptors_function *resolve_descriptors;
    get_loop_function *get_strided_loop;
    get_reduction_initial_function  *get_reduction_initial;
    /* Typical loop functions (contiguous ones are used in current casts) */
    PyArrayMethod_StridedLoop *strided_loop;
    PyArrayMethod_StridedLoop *contiguous_loop;
    PyArrayMethod_StridedLoop *unaligned_strided_loop;
    PyArrayMethod_StridedLoop *unaligned_contiguous_loop;
    PyArrayMethod_StridedLoop *contiguous_indexed_loop;
    /* Chunk only used for wrapping array method defined in umath */
    struct PyArrayMethodObject_tag *wrapped_meth;
    PyArray_DTypeMeta **wrapped_dtypes;
    translate_given_descrs_func *translate_given_descrs;
    translate_loop_descrs_func *translate_loop_descrs;
    /* Chunk reserved for use by the legacy fallback arraymethod */
    char legacy_initial[sizeof(npy_clongdouble)];  /* initial value storage */
} PyArrayMethodObject;


/*
 * We will sometimes have to create a ArrayMethod and allow passing it around,
 * similar to `instance.method` returning a bound method, e.g. a function like
 * `ufunc.resolve()` can return a bound object.
 * The current main purpose of the BoundArrayMethod is that it holds on to the
 * `dtypes` (the classes), so that the `ArrayMethod` (e.g. for casts) will
 * not create references cycles.  In principle, it could hold any information
 * which is also stored on the ufunc (and thus does not need to be repeated
 * on the `ArrayMethod` itself.
 */
typedef struct {
    PyObject_HEAD
    PyArray_DTypeMeta **dtypes;
    PyArrayMethodObject *method;
} PyBoundArrayMethodObject;


extern NPY_NO_EXPORT PyTypeObject PyArrayMethod_Type;
extern NPY_NO_EXPORT PyTypeObject PyBoundArrayMethod_Type;


/*
 * Used internally (initially) for real to complex loops only
 */
NPY_NO_EXPORT int
npy_default_get_strided_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);


NPY_NO_EXPORT int
PyArrayMethod_GetMaskedStridedLoop(
        PyArrayMethod_Context *context,
        int aligned,
        npy_intp *fixed_strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);



NPY_NO_EXPORT PyObject *
PyArrayMethod_FromSpec(PyArrayMethod_Spec *spec);


/*
 * TODO: This function is the internal version, and its error paths may
 *       need better tests when a public version is exposed.
 */
NPY_NO_EXPORT PyBoundArrayMethodObject *
PyArrayMethod_FromSpec_int(PyArrayMethod_Spec *spec, int priv);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAY_METHOD_H_ */
