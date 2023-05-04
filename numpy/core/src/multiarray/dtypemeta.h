#ifndef NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_

#include "array_method.h"
#include "dtype_traversal.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "numpy/_dtype_api.h"

/* DType flags, currently private, since we may just expose functions 
   Other publicly visible flags are in _dtype_api.h                   */
#define NPY_DT_LEGACY 1 << 0


typedef struct {
    /* DType methods, these could be moved into its own struct */
    discover_descr_from_pyobject_function *discover_descr_from_pyobject;
    is_known_scalar_type_function *is_known_scalar_type;
    default_descr_function *default_descr;
    common_dtype_function *common_dtype;
    common_instance_function *common_instance;
    ensure_canonical_function *ensure_canonical;
    /*
     * Currently only used for experimental user DTypes.
     */
    setitemfunction *setitem;
    getitemfunction *getitem;
    /*
     * Either NULL or fetches a clearing function.  Clearing means deallocating
     * any referenced data and setting it to a safe state.  For Python objects
     * this means using `Py_CLEAR` which is equivalent to `Py_DECREF` and
     * setting the `PyObject *` to NULL.
     * After the clear, the data must be fillable via cast/copy and calling
     * clear a second time must be safe.
     * If the DType class does not implement `get_clear_loop` setting
     * NPY_ITEM_REFCOUNT on its dtype instances is invalid.  Note that it is
     * acceptable for  NPY_ITEM_REFCOUNT to inidicate references that are not
     * Python objects.
     */
    get_traverse_loop_function *get_clear_loop;
    /*
       Either NULL or a function that sets a function pointer to a traversal
       loop that fills an array with zero values appropriate for the dtype. If
       get_fill_zero_loop is undefined or the function pointer set by it is
       NULL, the array buffer is allocated with calloc. If this function is
       defined and it sets a non-NULL function pointer, the array buffer is
       allocated with malloc and the zero-filling loop function pointer is
       called to fill the buffer. For the best performance, avoid using this
       function if a zero-filled array buffer allocated with calloc makes sense
       for the dtype.

       Note that this is currently used only for zero-filling a newly allocated
       array. While it can be used to zero-fill an already-filled buffer, that
       will not work correctly for arrays holding references. If you need to do
       that, clear the array first.
    */
    get_traverse_loop_function *get_fill_zero_loop;
    /*
     * The casting implementation (ArrayMethod) to convert between two
     * instances of this DType, stored explicitly for fast access:
     */
    PyArrayMethodObject *within_dtype_castingimpl;
    /*
     * Dictionary of ArrayMethods representing most possible casts
     * (structured and object are exceptions).
     * This should potentially become a weak mapping in the future.
     */
    PyObject *castingimpls;

    /*
     * Storage for `descr->f`, since we may need to allow some customizatoin
     * here at least in a transition period and we need to set it on every
     * dtype instance for backward compatibility.  (Keep this at end)
     */
    PyArray_ArrFuncs f;
} NPY_DType_Slots;

// This must be updated if new slots before within_dtype_castingimpl
// are added
#define NPY_NUM_DTYPE_SLOTS 10
#define NPY_NUM_DTYPE_PYARRAY_ARRFUNCS_SLOTS 22
#define NPY_DT_MAX_ARRFUNCS_SLOT \
  NPY_NUM_DTYPE_PYARRAY_ARRFUNCS_SLOTS + _NPY_DT_ARRFUNCS_OFFSET


#define NPY_DTYPE(descr) ((PyArray_DTypeMeta *)Py_TYPE(descr))
#define NPY_DT_SLOTS(dtype) ((NPY_DType_Slots *)(dtype)->dt_slots)

#define NPY_DT_is_legacy(dtype) (((dtype)->flags & NPY_DT_LEGACY) != 0)
#define NPY_DT_is_abstract(dtype) (((dtype)->flags & NPY_DT_ABSTRACT) != 0)
#define NPY_DT_is_parametric(dtype) (((dtype)->flags & NPY_DT_PARAMETRIC) != 0)
#define NPY_DT_is_numeric(dtype) (((dtype)->flags & NPY_DT_NUMERIC) != 0)
#define NPY_DT_is_user_defined(dtype) (((dtype)->type_num == -1))

/*
 * Macros for convenient classmethod calls, since these require
 * the DType both for the slot lookup and as first arguments.
 *
 * (Macros may include NULL checks where appropriate)
 */
#define NPY_DT_CALL_discover_descr_from_pyobject(dtype, obj)  \
    NPY_DT_SLOTS(dtype)->discover_descr_from_pyobject(dtype, obj)
#define NPY_DT_CALL_is_known_scalar_type(dtype, obj)  \
    (NPY_DT_SLOTS(dtype)->is_known_scalar_type != NULL  \
        && NPY_DT_SLOTS(dtype)->is_known_scalar_type(dtype, obj))
#define NPY_DT_CALL_default_descr(dtype)  \
    NPY_DT_SLOTS(dtype)->default_descr(dtype)
#define NPY_DT_CALL_common_dtype(dtype, other)  \
    NPY_DT_SLOTS(dtype)->common_dtype(dtype, other)
#define NPY_DT_CALL_ensure_canonical(descr)  \
    NPY_DT_SLOTS(NPY_DTYPE(descr))->ensure_canonical(descr)
#define NPY_DT_CALL_getitem(descr, data_ptr)  \
    NPY_DT_SLOTS(NPY_DTYPE(descr))->getitem(descr, data_ptr)
#define NPY_DT_CALL_setitem(descr, value, data_ptr)  \
    NPY_DT_SLOTS(NPY_DTYPE(descr))->setitem(descr, value, data_ptr)


/*
 * This function will hopefully be phased out or replaced, but was convenient
 * for incremental implementation of new DTypes based on DTypeMeta.
 * (Error checking is not required for DescrFromType, assuming that the
 * type is valid.)
 */
static inline PyArray_DTypeMeta *
PyArray_DTypeFromTypeNum(int typenum)
{
    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    PyArray_DTypeMeta *dtype = NPY_DTYPE(descr);
    Py_INCREF(dtype);
    Py_DECREF(descr);
    return dtype;
}


NPY_NO_EXPORT int
python_builtins_are_known_scalar_types(
        PyArray_DTypeMeta *cls, PyTypeObject *pytype);

NPY_NO_EXPORT int
dtypemeta_wrap_legacy_descriptor(
        PyArray_Descr *dtypem, const char *name, const char *alias);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_ */
