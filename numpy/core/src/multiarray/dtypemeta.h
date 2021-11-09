#ifndef NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_


/* DType flags, currently private, since we may just expose functions */
#define NPY_DT_LEGACY 1 << 0
#define NPY_DT_ABSTRACT 1 << 1
#define NPY_DT_PARAMETRIC 1 << 2


typedef PyArray_Descr *(discover_descr_from_pyobject_function)(
        PyArray_DTypeMeta *cls, PyObject *obj);

/*
 * Before making this public, we should decide whether it should pass
 * the type, or allow looking at the object. A possible use-case:
 * `np.array(np.array([0]), dtype=np.ndarray)`
 * Could consider arrays that are not `dtype=ndarray` "scalars".
 */
typedef int (is_known_scalar_type_function)(
        PyArray_DTypeMeta *cls, PyTypeObject *obj);

typedef PyArray_Descr *(default_descr_function)(PyArray_DTypeMeta *cls);
typedef PyArray_DTypeMeta *(common_dtype_function)(
        PyArray_DTypeMeta *dtype1, PyArray_DTypeMeta *dtype2);
typedef PyArray_Descr *(common_instance_function)(
        PyArray_Descr *dtype1, PyArray_Descr *dtype2);

/*
 * TODO: These two functions are currently only used for experimental DType
 *       API support.  Their relation should be "reversed": NumPy should
 *       always use them internally.
 *       There are open points about "casting safety" though, e.g. setting
 *       elements is currently always unsafe.
 */
typedef int(setitemfunction)(PyArray_Descr *, PyObject *, char *);
typedef PyObject *(getitemfunction)(PyArray_Descr *, char *);


typedef struct {
    /* DType methods, these could be moved into its own struct */
    discover_descr_from_pyobject_function *discover_descr_from_pyobject;
    is_known_scalar_type_function *is_known_scalar_type;
    default_descr_function *default_descr;
    common_dtype_function *common_dtype;
    common_instance_function *common_instance;
    /*
     * Currently only used for experimental user DTypes.
     * Typing as `void *` until NumPy itself uses these (directly).
     */
    setitemfunction *setitem;
    getitemfunction *getitem;
    /*
     * The casting implementation (ArrayMethod) to convert between two
     * instances of this DType, stored explicitly for fast access:
     */
    PyObject *within_dtype_castingimpl;
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


#define NPY_DTYPE(descr) ((PyArray_DTypeMeta *)Py_TYPE(descr))
#define NPY_DT_SLOTS(dtype) ((NPY_DType_Slots *)(dtype)->dt_slots)

#define NPY_DT_is_legacy(dtype) (((dtype)->flags & NPY_DT_LEGACY) != 0)
#define NPY_DT_is_abstract(dtype) (((dtype)->flags & NPY_DT_ABSTRACT) != 0)
#define NPY_DT_is_parametric(dtype) (((dtype)->flags & NPY_DT_PARAMETRIC) != 0)

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
static NPY_INLINE PyArray_DTypeMeta *
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
dtypemeta_wrap_legacy_descriptor(PyArray_Descr *dtypem);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DTYPEMETA_H_ */
