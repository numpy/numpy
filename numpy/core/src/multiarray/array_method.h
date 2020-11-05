#ifndef _NPY_ARRAY_METHOD_H
#define _NPY_ARRAY_METHOD_H

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <lowlevel_strided_loops.h>


typedef enum {
    /* Flag for whether the GIL is required */
    NPY_METH_REQUIRES_PYAPI = 1 << 1,
    /*
     * Some functions cannot set floating point error flags, this flag
     * gives us the option (not requirement) to skip floating point error
     * setup/check. No function should set error flags and ignore them
     * since it would interfere with chaining operations (e.g. casting).
     */
    NPY_METH_NO_FLOATINGPOINT_ERRORS = 1 << 2,
    /* Whether the method supports unaligned access (not runtime) */
    NPY_METH_SUPPORTS_UNALIGNED = 1 << 3,

    /* All flags which can change at runtime */
    NPY_METH_RUNTIME_FLAGS = (
            NPY_METH_REQUIRES_PYAPI |
            NPY_METH_NO_FLOATINGPOINT_ERRORS),
} NPY_ARRAYMETHOD_FLAGS;


struct PyArrayMethodObject_tag;

/*
 * This struct is specific to an individual (possibly repeated) call of
 * the DTypeMethods strided operator, and as such is passed into the various
 * methods of the DTypeMethod object (the adjust_descriptors function,
 * the get_loop function and the individual lowlevel strided operator calls).
 * It thus has to be persistent for one end-user call, and then be discarded.
 *
 * We recycle this as a specification for creating new DTypeMethods
 * right now.  (This should probably be reviewed before making it public)
 */
typedef struct {
    PyObject *caller;
    struct PyArrayMethodObject_tag *method;
    int nin, nout;

    PyArray_DTypeMeta **dtypes;
    /* Operand descriptors, filled in by adjust_desciptors */
    PyArray_Descr **descriptors;
} PyArrayMethod_Context;


typedef NPY_CASTING (resolve_descriptors_function)(
        PyArrayMethod_Context *context,
        PyArray_Descr **given_descrs,
        PyArray_Descr **loop_descrs);


typedef int (get_loop_function)(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        npy_intp *strides,
        PyArray_StridedUnaryOp **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);


/*
 * This struct will be public and necessary for creating a new DTypeMethod
 * object (casting and ufuncs).
 * We could version the struct, although since we allow passing arbitrary
 * data using the slots, and have flags, that may be enough?
 * (See also PyBoundArrayMethodObject.)
 */
typedef struct {
    const char *name;
    int nin, nout;
    NPY_CASTING casting;
    NPY_ARRAYMETHOD_FLAGS flags;
    PyArray_DTypeMeta **dtypes;
    PyType_Slot *slots;
} PyArrayMethod_Spec;


/*
 * Structure of the DTypeMethod. This structure should probably not be made
 * public. If necessary, we can make certain operations on it public
 * (e.g. to allow users access to `get_strided_loop`).
 */
typedef struct PyArrayMethodObject_tag {
    PyObject_HEAD
    char *name;
    /* Casting is normally "safe" for functions, but is important for casts */
    NPY_CASTING casting;
    /* default flags. The get_strided_loop function can override these */
    NPY_ARRAYMETHOD_FLAGS flags;
    resolve_descriptors_function *resolve_descriptors;
    get_loop_function *get_strided_loop;
    /* Typical loop functions (contiguous ones are used in current casts) */
    PyArray_StridedUnaryOp *strided_loop;
    PyArray_StridedUnaryOp *contiguous_loop;
    PyArray_StridedUnaryOp *unaligned_strided_loop;
    PyArray_StridedUnaryOp *unaligned_contiguous_loop;
} PyArrayMethodObject;


/*
 * We will sometimes have to create a DTypeMethod and allow passing it around,
 * similar to `instance.method` returning a bound method, e.g. a function like
 * `ufunc.resolve()` can return a bound object.
 * This or the method itself may need further attributes, such as the `owner`
 * (which could be the bound ufunc), the `signature` (of the gufunc), or
 * the identity for reduction support.
 */
typedef struct {
    PyObject_HEAD
    int nin;
    int nout;
    PyArray_DTypeMeta **dtypes;
    PyArrayMethodObject *method;
} PyBoundArrayMethodObject;


extern NPY_NO_EXPORT PyTypeObject PyArrayMethod_Type;
extern NPY_NO_EXPORT PyTypeObject PyBoundArrayMethod_Type;

/*
 * SLOTS IDs For the DTypeMethod creation, one public, the IDs are fixed.
 * TODO: Before making it public, consider adding a large constant to private
 *       slots.
 */
#define NPY_DTMETH_resolve_descriptors 1
#define NPY_DTMETH_get_loop 2
#define NPY_DTMETH_strided_loop 3
#define NPY_DTMETH_contiguous_loop 4
#define NPY_DTMETH_unaligned_strided_loop 5
#define NPY_DTMETH_unaligned_contiguous_loop 6


NPY_NO_EXPORT PyBoundArrayMethodObject *
PyArrayMethod_FromSpec_int(PyArrayMethod_Spec *spec, int private);

#endif  /*_NPY_ARRAY_METHOD_H*/
