/* Array Descr Object */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <numpy/ndarraytypes.h>
#include <numpy/arrayscalars.h>
#include "npy_pycompat.h"

#include "common.h"
#include "dtypemeta.h"
#include "_datetime.h"
#include "array_coercion.h"
#include "scalartypes.h"
#include "convert_datatype.h"
#include "usertypes.h"

#include <assert.h>

static void
dtypemeta_dealloc(PyArray_DTypeMeta *self) {
    /* Do not accidentally delete a statically defined DType: */
    assert(((PyTypeObject *)self)->tp_flags & Py_TPFLAGS_HEAPTYPE);

    Py_XDECREF(self->scalar_type);
    Py_XDECREF(self->singleton);
    Py_XDECREF(NPY_DT_SLOTS(self)->castingimpls);
    PyMem_Free(self->dt_slots);
    PyType_Type.tp_dealloc((PyObject *) self);
}

static PyObject *
dtypemeta_alloc(PyTypeObject *NPY_UNUSED(type), Py_ssize_t NPY_UNUSED(items))
{
    PyErr_SetString(PyExc_TypeError,
            "DTypes can only be created using the NumPy API.");
    return NULL;
}

static PyObject *
dtypemeta_new(PyTypeObject *NPY_UNUSED(type),
        PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
    PyErr_SetString(PyExc_TypeError,
            "Preliminary-API: Cannot subclass DType.");
    return NULL;
}

static int
dtypemeta_init(PyTypeObject *NPY_UNUSED(type),
        PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
    PyErr_SetString(PyExc_TypeError,
            "Preliminary-API: Cannot __init__ DType class.");
    return -1;
}

/**
 * tp_is_gc slot of Python types. This is implemented only for documentation
 * purposes to indicate and document the subtleties involved.
 *
 * Python Type objects are either statically created (typical C-Extension type)
 * or HeapTypes (typically created in Python).
 * HeapTypes have the Py_TPFLAGS_HEAPTYPE flag and are garbage collected.
 * Our DTypeMeta instances (`np.dtype` and its subclasses) *may* be HeapTypes
 * if the Py_TPFLAGS_HEAPTYPE flag is set (they are created from Python).
 * They are not for legacy DTypes or np.dtype itself.
 *
 * @param self
 * @return nonzero if the object is garbage collected
 */
static NPY_INLINE int
dtypemeta_is_gc(PyObject *dtype_class)
{
    return PyType_Type.tp_is_gc(dtype_class);
}


static int
dtypemeta_traverse(PyArray_DTypeMeta *type, visitproc visit, void *arg)
{
    /*
     * We have to traverse the base class (if it is a HeapType).
     * PyType_Type will handle this logic for us.
     * This function is currently not used, but will probably be necessary
     * in the future when we implement HeapTypes (python/dynamically
     * defined types). It should be revised at that time.
     */
    assert(0);
    assert(!NPY_DT_is_legacy(type) && (PyTypeObject *)type != &PyArrayDescr_Type);
    Py_VISIT(type->singleton);
    Py_VISIT(type->scalar_type);
    return PyType_Type.tp_traverse((PyObject *)type, visit, arg);
}


static PyObject *
legacy_dtype_default_new(PyArray_DTypeMeta *self,
        PyObject *args, PyObject *kwargs)
{
    /* TODO: This should allow endianness and possibly metadata */
    if (NPY_DT_is_parametric(self)) {
        /* reject parametric ones since we would need to get unit, etc. info */
        PyErr_Format(PyExc_TypeError,
                "Preliminary-API: Flexible/Parametric legacy DType '%S' can "
                "only be instantiated using `np.dtype(...)`", self);
        return NULL;
    }

    if (PyTuple_GET_SIZE(args) != 0 ||
                (kwargs != NULL && PyDict_Size(kwargs))) {
        PyErr_Format(PyExc_TypeError,
                "currently only the no-argument instantiation is supported; "
                "use `np.dtype` instead.");
        return NULL;
    }
    Py_INCREF(self->singleton);
    return (PyObject *)self->singleton;
}


static PyArray_Descr *
nonparametric_discover_descr_from_pyobject(
        PyArray_DTypeMeta *cls, PyObject *obj)
{
    /* If the object is of the correct scalar type return our singleton */
    assert(!NPY_DT_is_parametric(cls));
    Py_INCREF(cls->singleton);
    return cls->singleton;
}


static PyArray_Descr *
string_discover_descr_from_pyobject(
        PyArray_DTypeMeta *cls, PyObject *obj)
{
    npy_intp itemsize = -1;
    if (PyBytes_Check(obj)) {
        itemsize = PyBytes_Size(obj);
    }
    else if (PyUnicode_Check(obj)) {
        itemsize = PyUnicode_GetLength(obj);
    }
    if (itemsize != -1) {
        if (cls->type_num == NPY_UNICODE) {
            itemsize *= 4;
        }
        if (itemsize > NPY_MAX_INT) {
            PyErr_SetString(PyExc_TypeError,
                    "string to large to store inside array.");
        }
        PyArray_Descr *res = PyArray_DescrNewFromType(cls->type_num);
        if (res == NULL) {
            return NULL;
        }
        res->elsize = (int)itemsize;
        return res;
    }
    return PyArray_DTypeFromObjectStringDiscovery(obj, NULL, cls->type_num);
}


static PyArray_Descr *
void_discover_descr_from_pyobject(
        PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    if (PyArray_IsScalar(obj, Void)) {
        PyVoidScalarObject *void_obj = (PyVoidScalarObject *)obj;
        Py_INCREF(void_obj->descr);
        return void_obj->descr;
    }
    if (PyBytes_Check(obj)) {
        PyArray_Descr *descr = PyArray_DescrNewFromType(NPY_VOID);
        if (descr == NULL) {
            return NULL;
        }
        Py_ssize_t itemsize = PyBytes_Size(obj);
        if (itemsize > NPY_MAX_INT) {
            PyErr_SetString(PyExc_TypeError,
                    "byte-like to large to store inside array.");
            Py_DECREF(descr);
            return NULL;
        }
        descr->elsize = (int)itemsize;
        return descr;
    }
    PyErr_Format(PyExc_TypeError,
            "A bytes-like object is required, not '%s'", Py_TYPE(obj)->tp_name);
    return NULL;
}


static PyArray_Descr *
discover_datetime_and_timedelta_from_pyobject(
        PyArray_DTypeMeta *cls, PyObject *obj) {
    if (PyArray_IsScalar(obj, Datetime) ||
            PyArray_IsScalar(obj, Timedelta)) {
        PyArray_DatetimeMetaData *meta;
        PyArray_Descr *descr = PyArray_DescrFromScalar(obj);
        meta = get_datetime_metadata_from_dtype(descr);
        if (meta == NULL) {
            return NULL;
        }
        PyArray_Descr *new_descr = create_datetime_dtype(cls->type_num, meta);
        Py_DECREF(descr);
        return new_descr;
    }
    else {
        return find_object_datetime_type(obj, cls->type_num);
    }
}


static PyArray_Descr *
nonparametric_default_descr(PyArray_DTypeMeta *cls)
{
    Py_INCREF(cls->singleton);
    return cls->singleton;
}


/* Ensure a copy of the singleton (just in case we do adapt it somewhere) */
static PyArray_Descr *
datetime_and_timedelta_default_descr(PyArray_DTypeMeta *cls)
{
    return PyArray_DescrNew(cls->singleton);
}


static PyArray_Descr *
void_default_descr(PyArray_DTypeMeta *cls)
{
    PyArray_Descr *res = PyArray_DescrNew(cls->singleton);
    if (res == NULL) {
        return NULL;
    }
    /*
     * The legacy behaviour for `np.array([], dtype="V")` is to use "V8".
     * This is because `[]` uses `float64` as dtype, and then that is used
     * for the size of the requested void.
     */
    res->elsize = 8;
    return res;
}

static PyArray_Descr *
string_and_unicode_default_descr(PyArray_DTypeMeta *cls)
{
    PyArray_Descr *res = PyArray_DescrNewFromType(cls->type_num);
    if (res == NULL) {
        return NULL;
    }
    res->elsize = 1;
    if (cls->type_num == NPY_UNICODE) {
        res->elsize *= 4;
    }
    return res;
}


static PyArray_Descr *
string_unicode_common_instance(PyArray_Descr *descr1, PyArray_Descr *descr2)
{
    if (descr1->elsize >= descr2->elsize) {
        return ensure_dtype_nbo(descr1);
    }
    else {
        return ensure_dtype_nbo(descr2);
    }
}


static PyArray_Descr *
void_common_instance(PyArray_Descr *descr1, PyArray_Descr *descr2)
{
    /*
     * We currently do not support promotion of void types unless they
     * are equivalent.
     */
    if (!PyArray_CanCastTypeTo(descr1, descr2, NPY_EQUIV_CASTING)) {
        if (descr1->subarray == NULL && descr1->names == NULL &&
                descr2->subarray == NULL && descr2->names == NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "Invalid type promotion with void datatypes of different "
                    "lengths. Use the `np.bytes_` datatype instead to pad the "
                    "shorter value with trailing zero bytes.");
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                    "invalid type promotion with structured datatype(s).");
        }
        return NULL;
    }
    Py_INCREF(descr1);
    return descr1;
}

NPY_NO_EXPORT int
python_builtins_are_known_scalar_types(
        PyArray_DTypeMeta *NPY_UNUSED(cls), PyTypeObject *pytype)
{
    /*
     * Always accept the common Python types, this ensures that we do not
     * convert pyfloat->float64->integers. Subclasses are hopefully rejected
     * as being discovered.
     * This is necessary only for python scalar classes which we discover
     * as valid DTypes.
     */
    if (pytype == &PyFloat_Type) {
        return 1;
    }
    if (pytype == &PyLong_Type) {
        return 1;
    }
    if (pytype == &PyBool_Type) {
        return 1;
    }
    if (pytype == &PyComplex_Type) {
        return 1;
    }
    if (pytype == &PyUnicode_Type) {
        return 1;
    }
    if (pytype == &PyBytes_Type) {
        return 1;
    }
    return 0;
}


static int
signed_integers_is_known_scalar_types(
        PyArray_DTypeMeta *cls, PyTypeObject *pytype)
{
    if (python_builtins_are_known_scalar_types(cls, pytype)) {
        return 1;
    }
    /* Convert our scalars (raise on too large unsigned and NaN, etc.) */
    return PyType_IsSubtype(pytype, &PyGenericArrType_Type);
}


static int
datetime_known_scalar_types(
        PyArray_DTypeMeta *cls, PyTypeObject *pytype)
{
    if (python_builtins_are_known_scalar_types(cls, pytype)) {
        return 1;
    }
    /*
     * To be able to identify the descriptor from e.g. any string, datetime
     * must take charge. Otherwise we would attempt casting which does not
     * truly support this. Only object arrays are special cased in this way.
     */
    return (PyType_IsSubtype(pytype, &PyBytes_Type) ||
            PyType_IsSubtype(pytype, &PyUnicode_Type));
}


static int
string_known_scalar_types(
        PyArray_DTypeMeta *cls, PyTypeObject *pytype) {
    if (python_builtins_are_known_scalar_types(cls, pytype)) {
        return 1;
    }
    if (PyType_IsSubtype(pytype, &PyDatetimeArrType_Type)) {
        /*
         * TODO: This should likely be deprecated or otherwise resolved.
         *       Deprecation has to occur in `String->setitem` unfortunately.
         *
         * Datetime currently do not cast to shorter strings, but string
         * coercion for arbitrary values uses `str(obj)[:len]` so it works.
         * This means `np.array(np.datetime64("2020-01-01"), "U9")`
         * and `np.array(np.datetime64("2020-01-01")).astype("U9")` behave
         * differently.
         */
        return 1;
    }
    return 0;
}


/*
 * The following set of functions define the common dtype operator for
 * the builtin types.
 */
static PyArray_DTypeMeta *
default_builtin_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    assert(cls->type_num < NPY_NTYPES);
    if (!NPY_DT_is_legacy(other) || other->type_num > cls->type_num) {
        /*
         * Let the more generic (larger type number) DType handle this
         * (note that half is after all others, which works out here.)
         */
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }

    /*
     * Note: The use of the promotion table should probably be revised at
     *       some point. It may be most useful to remove it entirely and then
     *       consider adding a fast path/cache `PyArray_CommonDType()` itself.
     */
    int common_num = _npy_type_promotion_table[cls->type_num][other->type_num];
    if (common_num < 0) {
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }
    return PyArray_DTypeFromTypeNum(common_num);
}


static PyArray_DTypeMeta *
string_unicode_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    assert(cls->type_num < NPY_NTYPES && cls != other);
    if (!NPY_DT_is_legacy(other) || (!PyTypeNum_ISNUMBER(other->type_num) &&
            /* Not numeric so defer unless cls is unicode and other is string */
            !(cls->type_num == NPY_UNICODE && other->type_num == NPY_STRING))) {
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }
    /*
     * The builtin types are ordered by complexity (aside from object) here.
     * Arguably, we should not consider numbers and strings "common", but
     * we currently do.
     */
    Py_INCREF(cls);
    return cls;
}

static PyArray_DTypeMeta *
datetime_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (cls->type_num == NPY_DATETIME && other->type_num == NPY_TIMEDELTA) {
        /*
         * TODO: We actually currently do allow promotion here. This is
         *       currently relied on within `np.add(datetime, timedelta)`,
         *       while for concatenation the cast step will fail.
         */
        Py_INCREF(cls);
        return cls;
    }
    return default_builtin_common_dtype(cls, other);
}



static PyArray_DTypeMeta *
object_common_dtype(
        PyArray_DTypeMeta *cls, PyArray_DTypeMeta *NPY_UNUSED(other))
{
    /*
     * The object DType is special in that it can represent everything,
     * including all potential user DTypes.
     * One reason to defer (or error) here might be if the other DType
     * does not support scalars so that e.g. `arr1d[0]` returns a 0-D array
     * and `arr.astype(object)` would fail. But object casts are special.
     */
    Py_INCREF(cls);
    return cls;
}


/**
 * This function takes a PyArray_Descr and replaces its base class with
 * a newly created dtype subclass (DTypeMeta instances).
 * There are some subtleties that need to be remembered when doing this,
 * first for the class objects itself it could be either a HeapType or not.
 * Since we are defining the DType from C, we will not make it a HeapType,
 * thus making it identical to a typical *static* type (except that we
 * malloc it). We could do it the other way, but there seems no reason to
 * do so.
 *
 * The DType instances (the actual dtypes or descriptors), are based on
 * prototypes which are passed in. These should not be garbage collected
 * and thus Py_TPFLAGS_HAVE_GC is not set. (We could allow this, but than
 * would have to allocate a new object, since the GC needs information before
 * the actual struct).
 *
 * The above is the reason why we should works exactly like we would for a
 * static type here.
 * Otherwise, we blurry the lines between C-defined extension classes
 * and Python subclasses. e.g. `class MyInt(int): pass` is very different
 * from our `class Float64(np.dtype): pass`, because the latter should not
 * be a HeapType and its instances should be exact PyArray_Descr structs.
 *
 * @param descr The descriptor that should be wrapped.
 * @param name The name for the DType, if NULL the type character is used.
 *
 * @returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
dtypemeta_wrap_legacy_descriptor(PyArray_Descr *descr)
{
    int has_type_set = Py_TYPE(descr) == &PyArrayDescr_Type;

    if (!has_type_set) {
        /* Accept if the type was filled in from an existing builtin dtype */
        for (int i = 0; i < NPY_NTYPES; i++) {
            PyArray_Descr *builtin = PyArray_DescrFromType(i);
            has_type_set = Py_TYPE(descr) == Py_TYPE(builtin);
            Py_DECREF(builtin);
            if (has_type_set) {
                break;
            }
        }
    }
    if (!has_type_set) {
        PyErr_Format(PyExc_RuntimeError,
                "During creation/wrapping of legacy DType, the original class "
                "was not of PyArrayDescr_Type (it is replaced in this step). "
                "The extension creating a custom DType for type %S must be "
                "modified to ensure `Py_TYPE(descr) == &PyArrayDescr_Type` or "
                "that of an existing dtype (with the assumption it is just "
                "copied over and can be replaced).",
                descr->typeobj, Py_TYPE(descr));
        return -1;
    }

    /*
     * Note: we have no intention of freeing the memory again since this
     * behaves identically to static type definition (see comment above).
     * This is seems cleaner for the legacy API, in the new API both static
     * and heap types are possible (some difficulty arises from the fact that
     * these are instances of DTypeMeta and not type).
     * In particular our own DTypes can be true static declarations.
     * However, this function remains necessary for legacy user dtypes.
     */

    const char *scalar_name = descr->typeobj->tp_name;
    /*
     * We have to take only the name, and ignore the module to get
     * a reasonable __name__, since static types are limited in this regard
     * (this is not ideal, but not a big issue in practice).
     * This is what Python does to print __name__ for static types.
     */
    const char *dot = strrchr(scalar_name, '.');
    if (dot) {
        scalar_name = dot + 1;
    }
    Py_ssize_t name_length = strlen(scalar_name) + 14;

    char *tp_name = PyMem_Malloc(name_length);
    if (tp_name == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    snprintf(tp_name, name_length, "numpy.dtype[%s]", scalar_name);

    NPY_DType_Slots *dt_slots = PyMem_Malloc(sizeof(NPY_DType_Slots));
    if (dt_slots == NULL) {
        PyMem_Free(tp_name);
        return -1;
    }
    memset(dt_slots, '\0', sizeof(NPY_DType_Slots));

    PyArray_DTypeMeta *dtype_class = PyMem_Malloc(sizeof(PyArray_DTypeMeta));
    if (dtype_class == NULL) {
        PyMem_Free(tp_name);
        PyMem_Free(dt_slots);
        return -1;
    }

    /*
     * Initialize the struct fields identically to static code by copying
     * a prototype instances for everything except our own fields which
     * vary between the DTypes.
     * In particular any Object initialization must be strictly copied from
     * the untouched prototype to avoid complexities (e.g. with PyPy).
     * Any Type slots need to be fixed before PyType_Ready, although most
     * will be inherited automatically there.
     */
    static PyArray_DTypeMeta prototype = {
        {{
            PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
            .tp_name = NULL,  /* set below */
            .tp_basicsize = sizeof(PyArray_Descr),
            .tp_flags = Py_TPFLAGS_DEFAULT,
            .tp_base = &PyArrayDescr_Type,
            .tp_new = (newfunc)legacy_dtype_default_new,
        },},
        .flags = NPY_DT_LEGACY,
        /* Further fields are not common between DTypes */
    };
    memcpy(dtype_class, &prototype, sizeof(PyArray_DTypeMeta));
    /* Fix name of the Type*/
    ((PyTypeObject *)dtype_class)->tp_name = tp_name;
    dtype_class->dt_slots = dt_slots;

    /* Let python finish the initialization (probably unnecessary) */
    if (PyType_Ready((PyTypeObject *)dtype_class) < 0) {
        Py_DECREF(dtype_class);
        return -1;
    }
    dt_slots->castingimpls = PyDict_New();
    if (dt_slots->castingimpls == NULL) {
        Py_DECREF(dtype_class);
        return -1;
    }

    /*
     * Fill DTypeMeta information that varies between DTypes, any variable
     * type information would need to be set before PyType_Ready().
     */
    dtype_class->singleton = descr;
    Py_INCREF(descr->typeobj);
    dtype_class->scalar_type = descr->typeobj;
    dtype_class->type_num = descr->type_num;
    dt_slots->f = *(descr->f);

    /* Set default functions (correct for most dtypes, override below) */
    dt_slots->default_descr = nonparametric_default_descr;
    dt_slots->discover_descr_from_pyobject = (
            nonparametric_discover_descr_from_pyobject);
    dt_slots->is_known_scalar_type = python_builtins_are_known_scalar_types;
    dt_slots->common_dtype = default_builtin_common_dtype;
    dt_slots->common_instance = NULL;

    if (PyTypeNum_ISSIGNED(dtype_class->type_num)) {
        /* Convert our scalars (raise on too large unsigned and NaN, etc.) */
        dt_slots->is_known_scalar_type = signed_integers_is_known_scalar_types;
    }

    if (PyTypeNum_ISUSERDEF(descr->type_num)) {
        dt_slots->common_dtype = legacy_userdtype_common_dtype_function;
    }
    else if (descr->type_num == NPY_OBJECT) {
        dt_slots->common_dtype = object_common_dtype;
    }
    else if (PyTypeNum_ISDATETIME(descr->type_num)) {
        /* Datetimes are flexible, but were not considered previously */
        dtype_class->flags |= NPY_DT_PARAMETRIC;
        dt_slots->default_descr = datetime_and_timedelta_default_descr;
        dt_slots->discover_descr_from_pyobject = (
                discover_datetime_and_timedelta_from_pyobject);
        dt_slots->common_dtype = datetime_common_dtype;
        dt_slots->common_instance = datetime_type_promotion;
        if (descr->type_num == NPY_DATETIME) {
            dt_slots->is_known_scalar_type = datetime_known_scalar_types;
        }
    }
    else if (PyTypeNum_ISFLEXIBLE(descr->type_num)) {
        dtype_class->flags |= NPY_DT_PARAMETRIC;
        if (descr->type_num == NPY_VOID) {
            dt_slots->default_descr = void_default_descr;
            dt_slots->discover_descr_from_pyobject = (
                    void_discover_descr_from_pyobject);
            dt_slots->common_instance = void_common_instance;
        }
        else {
            dt_slots->default_descr = string_and_unicode_default_descr;
            dt_slots->is_known_scalar_type = string_known_scalar_types;
            dt_slots->discover_descr_from_pyobject = (
                    string_discover_descr_from_pyobject);
            dt_slots->common_dtype = string_unicode_common_dtype;
            dt_slots->common_instance = string_unicode_common_instance;
        }
    }

    if (_PyArray_MapPyTypeToDType(dtype_class, descr->typeobj,
            PyTypeNum_ISUSERDEF(dtype_class->type_num)) < 0) {
        Py_DECREF(dtype_class);
        return -1;
    }

    /* Finally, replace the current class of the descr */
    Py_SET_TYPE(descr, (PyTypeObject *)dtype_class);

    return 0;
}


static PyObject *
dtypemeta_get_abstract(PyArray_DTypeMeta *self) {
    return PyBool_FromLong(NPY_DT_is_abstract(self));
}

static PyObject *
dtypemeta_get_parametric(PyArray_DTypeMeta *self) {
    return PyBool_FromLong(NPY_DT_is_parametric(self));
}

/*
 * Simple exposed information, defined for each DType (class).
 */
static PyGetSetDef dtypemeta_getset[] = {
        {"_abstract", (getter)dtypemeta_get_abstract, NULL, NULL, NULL},
        {"_parametric", (getter)dtypemeta_get_parametric, NULL, NULL, NULL},
        {NULL, NULL, NULL, NULL, NULL}
};

static PyMemberDef dtypemeta_members[] = {
    {"type",
        T_OBJECT, offsetof(PyArray_DTypeMeta, scalar_type), READONLY, NULL},
    {NULL, 0, 0, 0, NULL},
};


NPY_NO_EXPORT PyTypeObject PyArrayDTypeMeta_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy._DTypeMeta",
    .tp_basicsize = sizeof(PyArray_DTypeMeta),
    .tp_dealloc = (destructor)dtypemeta_dealloc,
    /* Types are garbage collected (see dtypemeta_is_gc documentation) */
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_doc = "Preliminary NumPy API: The Type of NumPy DTypes (metaclass)",
    .tp_getset = dtypemeta_getset,
    .tp_members = dtypemeta_members,
    .tp_base = NULL,  /* set to PyType_Type at import time */
    .tp_alloc = dtypemeta_alloc,
    .tp_init = (initproc)dtypemeta_init,
    .tp_new = dtypemeta_new,
    .tp_is_gc = dtypemeta_is_gc,
    .tp_traverse = (traverseproc)dtypemeta_traverse,
};
