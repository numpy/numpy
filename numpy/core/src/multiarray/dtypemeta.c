/* Array Descr Object */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include "structmember.h"


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"
#include "npy_ctypes.h"
#include "npy_pycompat.h"

#include "_datetime.h"
#include "common.h"
#include "alloc.h"
#include "assert.h"

#include "dtypemeta.h"
#include "convert_datatype.h"


static void
dtypemeta_dealloc(PyArray_DTypeMeta *self) {
    /*
     * PyType_Type asserts Py_TPFLAGS_HEAPTYPE as well. Do not rely on
     * a python debug build though.
     */
    assert(((PyTypeObject *)self)->tp_flags & Py_TPFLAGS_HEAPTYPE);
    Py_XDECREF(self->scalar_type);
    PyType_Type.tp_dealloc((PyObject *) self);
}

static PyObject *
dtypemeta_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyErr_SetString(PyExc_TypeError,
            "Preliminary-API: Cannot subclass DType.");
    return NULL;
}

static int
dtypemeta_init(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyErr_SetString(PyExc_TypeError,
            "Preliminary-API: Cannot initialize DType class.");
    return -1;
}

/**
 * tp_is_gc slot of Python types. This is implemented only for documentation
 * purposes to indicate and document the subtleties involved.
 *
 * Python Type objects are either statically created (typical C-Extension type)
 * or HeapTypes (typically created in Python).
 * HeapTypes have the Py_TPFLAGS_HEAPTYPE flag, and are garbage collected, our
 * DTypeMeta instances (`np.dtype` and its subclasses) *may* be HeapTypes
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
     * TODO: In the future, we may have to VISIT some python objects held,
     *       however, only if we are a Py_TPFLAGS_HEAPTYPE.
     */
    assert(!type->is_legacy && (PyTypeObject *)type != &PyArrayDescr_Type);
    Py_VISIT(type->singleton);
    Py_VISIT(type->scalar_type);
    return PyType_Type.tp_traverse((PyObject *)type, visit, arg);
}


static PyObject *
legacy_dtype_default_new(PyArray_DTypeMeta *self,
        PyObject *args, PyObject *kwargs)
{
    /* TODO: This should allow endianess and possibly metadata */
    if (self->is_flexible) {
        /* reject flexible ones since we would need to get unit, etc. info */
        PyErr_Format(PyExc_TypeError,
                "Preliminary-API: Flexible legacy DType '%S' can only be "
                "instantiated using `np.dtype(...)`", self);
        return NULL;
    }

    if (PyTuple_GET_SIZE(args) != 0) {
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
     * Note: we have no intention of cleaning this up, since this behaves
     * identical to static type definition (see comment above).
     * This is much cleaner for the legacy API, in the new API both ways
     * should be possible.
     * In particular our own DTypes can be true static declarations so that
     * this function is only needed for legacy user dtypes.
     */
    char *tp_name = PyDataMem_NEW(100);
    snprintf(tp_name, 100, "numpy.dtype[%s]",
             descr->typeobj->tp_name);

    PyArray_DTypeMeta *dtype_class = PyDataMem_NEW(sizeof(PyArray_DTypeMeta));
    if (dtype_class == NULL) {
        return -1;
    }
    /*
     * Initialize the struct fields similar to static code:
     */
    /* Copy in np.dtype since it shares most things */
    memcpy(dtype_class, &PyArrayDescr_Type, sizeof(PyArray_DTypeMeta));

    /* Fix name, base, and __new__*/
    ((PyTypeObject *)dtype_class)->tp_name = tp_name;
    ((PyTypeObject *)dtype_class)->tp_base = &PyArrayDescr_Type;
    ((PyTypeObject *)dtype_class)->tp_new = (newfunc)legacy_dtype_default_new;

    /* Let python finish the initialization (probably unnecessary) */
    if (PyType_Ready((PyTypeObject *)dtype_class) < 0) {
        return -1;
    }

    /* np.dtype is not a concrete DType, this one is */
    dtype_class->is_abstract = NPY_FALSE;

    Py_INCREF(descr);  /* descr is a singleton that must survive, ensure. */
    dtype_class->singleton = descr;
    Py_INCREF(descr->typeobj);
    dtype_class->scalar_type = descr->typeobj;
    dtype_class->is_legacy = NPY_TRUE;
    dtype_class->type_num = descr->type_num;
    dtype_class->type = descr->type;
    dtype_class->f = descr->f;
    dtype_class->kind = descr->kind;
    dtype_class->itemsize = descr->elsize;

    if (PyTypeNum_ISDATETIME(descr->type_num)) {
        /* Datetimes are flexible, but were not considered previously */
        dtype_class->is_flexible = NPY_TRUE;
    }
    if (PyTypeNum_ISFLEXIBLE(descr->type_num)) {
        dtype_class->is_flexible = NPY_TRUE;
        dtype_class->itemsize = -1;  /* itemsize is not fixed */
    }

    /* Finally, replace the current class of the descr */
    Py_TYPE(descr) = (PyTypeObject *)dtype_class;

    return 0;
}


/*
 * Simple exposed information, defined for each DType (class). This is
 * preliminary (the flags should also return bools).
 */
static PyMemberDef dtypemeta_members[] = {
        {"_abstract",
                T_BYTE, offsetof(PyArray_DTypeMeta, is_abstract), READONLY, NULL},
        {"type",
                T_OBJECT, offsetof(PyArray_DTypeMeta, scalar_type), READONLY, NULL},
        {"_flexible",
                T_BYTE, offsetof(PyArray_DTypeMeta, is_flexible), READONLY, NULL},
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

