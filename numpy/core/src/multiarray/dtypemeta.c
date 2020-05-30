/* Array Descr Object */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include "assert.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/ndarraytypes.h>
#include "npy_pycompat.h"

#include "dtypemeta.h"


static void
dtypemeta_dealloc(PyArray_DTypeMeta *self) {
    /* Do not accidentally delete a statically defined DType: */
    assert(((PyTypeObject *)self)->tp_flags & Py_TPFLAGS_HEAPTYPE);

    Py_XDECREF(self->scalar_type);
    Py_XDECREF(self->singleton);
    PyType_Type.tp_dealloc((PyObject *) self);
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
    assert(!type->legacy && (PyTypeObject *)type != &PyArrayDescr_Type);
    Py_VISIT(type->singleton);
    Py_VISIT(type->scalar_type);
    return PyType_Type.tp_traverse((PyObject *)type, visit, arg);
}


static PyObject *
legacy_dtype_default_new(PyArray_DTypeMeta *self,
        PyObject *args, PyObject *kwargs)
{
    /* TODO: This should allow endianess and possibly metadata */
    if (self->parametric) {
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
    if (Py_TYPE(descr) != &PyArrayDescr_Type) {
        PyErr_Format(PyExc_RuntimeError,
                "During creation/wrapping of legacy DType, the original class "
                "was not PyArrayDescr_Type (it is replaced in this step).");
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
    ssize_t name_length = strlen(scalar_name) + 14;

    char *tp_name = malloc(name_length);
    if (tp_name == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    snprintf(tp_name, name_length, "numpy.dtype[%s]", scalar_name);

    PyArray_DTypeMeta *dtype_class = malloc(sizeof(PyArray_DTypeMeta));
    if (dtype_class == NULL) {
        PyDataMem_FREE(tp_name);
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
        .legacy = 1,
        .abstract = 0, /* this is a concrete DType */
        /* Further fields are not common between DTypes */
    };
    memcpy(dtype_class, &prototype, sizeof(PyArray_DTypeMeta));
    /* Fix name of the Type*/
    ((PyTypeObject *)dtype_class)->tp_name = tp_name;

    /* Let python finish the initialization (probably unnecessary) */
    if (PyType_Ready((PyTypeObject *)dtype_class) < 0) {
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
    dtype_class->type = descr->type;
    dtype_class->f = descr->f;
    dtype_class->kind = descr->kind;

    if (PyTypeNum_ISDATETIME(descr->type_num)) {
        /* Datetimes are flexible, but were not considered previously */
        dtype_class->parametric = NPY_TRUE;
    }
    else if (PyTypeNum_ISFLEXIBLE(descr->type_num)) {
        dtype_class->parametric = NPY_TRUE;
    }

    /* Finally, replace the current class of the descr */
    Py_SET_TYPE(descr, (PyTypeObject *)dtype_class);

    return 0;
}


/*
 * Simple exposed information, defined for each DType (class). This is
 * preliminary (the flags should also return bools).
 */
static PyMemberDef dtypemeta_members[] = {
    {"_abstract",
        T_BYTE, offsetof(PyArray_DTypeMeta, abstract), READONLY, NULL},
    {"type",
        T_OBJECT, offsetof(PyArray_DTypeMeta, scalar_type), READONLY, NULL},
    {"_parametric",
        T_BYTE, offsetof(PyArray_DTypeMeta, parametric), READONLY, NULL},
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
    .tp_members = dtypemeta_members,
    .tp_base = NULL,  /* set to PyType_Type at import time */
    .tp_init = (initproc)dtypemeta_init,
    .tp_new = dtypemeta_new,
    .tp_is_gc = dtypemeta_is_gc,
    .tp_traverse = (traverseproc)dtypemeta_traverse,
};
