/* Array Descr Object */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <numpy/ndarraytypes.h>
#include <numpy/arrayscalars.h>
#include <numpy/npy_math.h>

#include "npy_import.h"

#include "abstractdtypes.h"
#include "arraytypes.h"
#include "common.h"
#include "dtypemeta.h"
#include "descriptor.h"
#include "_datetime.h"
#include "array_coercion.h"
#include "scalartypes.h"
#include "convert_datatype.h"
#include "usertypes.h"
#include "conversion_utils.h"
#include "templ_common.h"
#include "refcount.h"
#include "dtype_traversal.h"
#include "npy_static_data.h"
#include "multiarraymodule.h"

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

static PyArray_DTypeMeta *
dtype_does_not_promote(
        PyArray_DTypeMeta *NPY_UNUSED(self), PyArray_DTypeMeta *NPY_UNUSED(other))
{
    /* `other` is guaranteed not to be `self`, so we don't have to do much... */
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}

NPY_NO_EXPORT PyArray_Descr *
dtypemeta_discover_as_default(
        PyArray_DTypeMeta *cls, PyObject *NPY_UNUSED(obj))
{
    return NPY_DT_CALL_default_descr(cls);
}


static PyArray_Descr *
use_new_as_default(PyArray_DTypeMeta *self)
{
    PyObject *res = PyObject_CallObject((PyObject *)self, NULL);
    if (res == NULL) {
        return NULL;
    }
    /*
     * Let's not trust that the DType is implemented correctly
     * TODO: Should probably do an exact type-check (at least unless this is
     *       an abstract DType).
     */
    if (!PyArray_DescrCheck(res)) {
        PyErr_Format(PyExc_RuntimeError,
                "Instantiating %S did not return a dtype instance, this is "
                "invalid (especially without a custom `default_descr()`).",
                self);
        Py_DECREF(res);
        return NULL;
    }
    PyArray_Descr *descr = (PyArray_Descr *)res;
    /*
     * Should probably do some more sanity checks here on the descriptor
     * to ensure the user is not being naughty. But in the end, we have
     * only limited control anyway.
     */
    return descr;
}


/*
 * By default fill in zero, one, and negative one via the Python casts,
 * users should override this, but this allows us to use it for legacy user dtypes.
 */
static int
default_get_constant(PyArray_Descr *descr, int constant_id, void *data)
{
    return 0;
}


static int
legacy_fallback_setitem(PyArray_Descr *descr, PyObject *value, char *data)
{
    PyArrayObject_fields arr_fields = {
        .flags = NPY_ARRAY_WRITEABLE,  /* assume array is not behaved. */
        .descr = descr,
    };
    Py_SET_TYPE(&arr_fields, &PyArray_Type);
    Py_SET_REFCNT(&arr_fields, 1);

    return PyDataType_GetArrFuncs(descr)->setitem(value, data, &arr_fields);
}


static int
legacy_setitem_using_DType(PyObject *obj, void *data, void *arr)
{
    if (arr == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Using legacy SETITEM with NULL array object is only "
                "supported for basic NumPy DTypes.");
        return -1;
    }
    return NPY_DT_CALL_setitem(PyArray_DESCR(arr), obj, data);
}


static PyObject *
legacy_getitem_using_DType(void *data, void *arr)
{
    if (arr == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Using legacy SETITEM with NULL array object is only "
                "supported for basic NumPy DTypes.");
        return NULL;
    }
    PyArrayDTypeMeta_GetItem *getitem;
    getitem = NPY_DT_SLOTS(NPY_DTYPE(PyArray_DESCR(arr)))->getitem;
    return getitem(PyArray_DESCR(arr), data);
}

/*
 * The descr->f structure used user-DTypes.  Some functions may be filled
 * from the user in the future and more could get defaults for compatibility.
 */
PyArray_ArrFuncs default_funcs = {
        .getitem = &legacy_getitem_using_DType,
        .setitem = &legacy_setitem_using_DType,
};

/*
 * Internal version of PyArrayInitDTypeMeta_FromSpec.
 *
 * See the documentation of that function for more details.
 *
 * Setting priv to a nonzero value indicates that a dtypemeta is being
 * initialized from inside NumPy, otherwise this function is being called by
 * the public implementation.
 */
NPY_NO_EXPORT int
dtypemeta_initialize_struct_from_spec(
        PyArray_DTypeMeta *DType, PyArrayDTypeMeta_Spec *spec, int priv)
{
    if (DType->dt_slots != NULL) {
        PyErr_Format(PyExc_RuntimeError,
                "DType %R appears already registered?", DType);
        return -1;
    }

    DType->flags = spec->flags;
    DType->dt_slots = PyMem_Calloc(1, sizeof(NPY_DType_Slots));
    if (DType->dt_slots == NULL) {
        return -1;
    }

    /* Set default values (where applicable) */
    NPY_DT_SLOTS(DType)->discover_descr_from_pyobject =
            &dtypemeta_discover_as_default;
    NPY_DT_SLOTS(DType)->is_known_scalar_type = (
            &python_builtins_are_known_scalar_types);
    NPY_DT_SLOTS(DType)->default_descr = use_new_as_default;
    NPY_DT_SLOTS(DType)->common_dtype = dtype_does_not_promote;
    /* May need a default for non-parametric? */
    NPY_DT_SLOTS(DType)->common_instance = NULL;
    NPY_DT_SLOTS(DType)->setitem = NULL;
    NPY_DT_SLOTS(DType)->getitem = NULL;
    NPY_DT_SLOTS(DType)->get_clear_loop = NULL;
    NPY_DT_SLOTS(DType)->get_fill_zero_loop = NULL;
    NPY_DT_SLOTS(DType)->finalize_descr = NULL;
    NPY_DT_SLOTS(DType)->get_constant = default_get_constant;
    NPY_DT_SLOTS(DType)->f = default_funcs;

    PyType_Slot *spec_slot = spec->slots;
    while (1) {
        int slot = spec_slot->slot;
        void *pfunc = spec_slot->pfunc;
        spec_slot++;
        if (slot == 0) {
            break;
        }
        if ((slot < 0) ||
            ((slot > NPY_NUM_DTYPE_SLOTS) &&
             (slot <= _NPY_DT_ARRFUNCS_OFFSET)) ||
            (slot > NPY_DT_MAX_ARRFUNCS_SLOT)) {
            PyErr_Format(PyExc_RuntimeError,
                    "Invalid slot with value %d passed in.", slot);
            return -1;
        }
        /*
         * It is up to the user to get this right, the slots in the public API
         * are sorted exactly like they are stored in the NPY_DT_Slots struct
         * right now:
         */
        if (slot <= NPY_NUM_DTYPE_SLOTS) {
            // slot > NPY_NUM_DTYPE_SLOTS are PyArray_ArrFuncs
            void **current = (void **)(&(
                    NPY_DT_SLOTS(DType)->discover_descr_from_pyobject));
            current += slot - 1;
            *current = pfunc;
        }
        else {
            int f_slot = slot - _NPY_DT_ARRFUNCS_OFFSET;
            if (1 <= f_slot && f_slot <= NPY_NUM_DTYPE_PYARRAY_ARRFUNCS_SLOTS) {
                switch (f_slot) {
                    case 1:
                        NPY_DT_SLOTS(DType)->f.getitem = pfunc;
                        break;
                    case 2:
                        NPY_DT_SLOTS(DType)->f.setitem = pfunc;
                        break;
                    case 3:
                        NPY_DT_SLOTS(DType)->f.copyswapn = pfunc;
                        break;
                    case 4:
                        NPY_DT_SLOTS(DType)->f.copyswap = pfunc;
                        break;
                    case 5:
                        NPY_DT_SLOTS(DType)->f.compare = pfunc;
                        break;
                    case 6:
                        NPY_DT_SLOTS(DType)->f.argmax = pfunc;
                        break;
                    case 7:
                        NPY_DT_SLOTS(DType)->f.dotfunc = pfunc;
                        break;
                    case 8:
                        NPY_DT_SLOTS(DType)->f.scanfunc = pfunc;
                        break;
                    case 9:
                        NPY_DT_SLOTS(DType)->f.fromstr = pfunc;
                        break;
                    case 10:
                        NPY_DT_SLOTS(DType)->f.nonzero = pfunc;
                        break;
                    case 11:
                        NPY_DT_SLOTS(DType)->f.fill = pfunc;
                        break;
                    case 12:
                        NPY_DT_SLOTS(DType)->f.fillwithscalar = pfunc;
                        break;
                    case 13:
                        *NPY_DT_SLOTS(DType)->f.sort = pfunc;
                        break;
                    case 14:
                        *NPY_DT_SLOTS(DType)->f.argsort = pfunc;
                        break;
                    case 15:
                    case 16:
                    case 17:
                    case 18:
                    case 19:
                    case 20:
                    case 21:
                        PyErr_Format(
                            PyExc_RuntimeError,
                            "PyArray_ArrFunc casting slot with value %d is disabled.",
                            f_slot
                        );
                        return -1;
                    case 22:
                        NPY_DT_SLOTS(DType)->f.argmin = pfunc;
                        break;
                }
            } else {
                    PyErr_Format(
                        PyExc_RuntimeError,
                        "Invalid PyArray_ArrFunc slot with value %d passed in.",
                        f_slot
                    );
                    return -1;
            }
        }
    }

    /* invalid type num. Ideally, we get away with it! */
    DType->type_num = -1;

    /*
     * Handle the scalar type mapping.
     */
    Py_INCREF(spec->typeobj);
    DType->scalar_type = spec->typeobj;
    if (PyType_GetFlags(spec->typeobj) & Py_TPFLAGS_HEAPTYPE) {
        if (PyObject_SetAttrString((PyObject *)DType->scalar_type,
                "__associated_array_dtype__", (PyObject *)DType) < 0) {
            Py_DECREF(DType);
            return -1;
        }
    }
    if (_PyArray_MapPyTypeToDType(DType, DType->scalar_type, 0) < 0) {
        Py_DECREF(DType);
        return -1;
    }

    /* Ensure cast dict is defined (not sure we have to do it here) */
    NPY_DT_SLOTS(DType)->castingimpls = PyDict_New();
    if (NPY_DT_SLOTS(DType)->castingimpls == NULL) {
        return -1;
    }

    /*
     * And now, register all the casts that are currently defined!
     */
    PyArrayMethod_Spec **next_meth_spec = spec->casts;
    while (1) {
        PyArrayMethod_Spec *meth_spec = *next_meth_spec;
        next_meth_spec++;
        if (meth_spec == NULL) {
            break;
        }
        /*
         * The user doesn't know the name of DType yet, so we have to fill it
         * in for them!
         */
        for (int i=0; i < meth_spec->nin + meth_spec->nout; i++) {
            if (meth_spec->dtypes[i] == NULL) {
                meth_spec->dtypes[i] = DType;
            }
        }
        /* Register the cast! */
        // priv indicates whether or not the is an internal call
        int res = PyArray_AddCastingImplementation_FromSpec(meth_spec, priv);

        /* Also clean up again, so nobody can get bad ideas... */
        for (int i=0; i < meth_spec->nin + meth_spec->nout; i++) {
            if (meth_spec->dtypes[i] == DType) {
                meth_spec->dtypes[i] = NULL;
            }
        }

        if (res < 0) {
            return -1;
        }
    }

    return 0;
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
 * @param dtype_class Pointer to the Python type object
 * @return nonzero if the object is garbage collected
 */
static inline int
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

static PyObject *
string_unicode_new(PyArray_DTypeMeta *self, PyObject *args, PyObject *kwargs)
{
    npy_intp size;

    static char *kwlist[] = {"", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist,
                                     PyArray_IntpFromPyIntConverter, &size)) {
        return NULL;
    }

    if (size < 0) {
        PyErr_Format(PyExc_ValueError,
                     "Strings cannot have a negative size but a size of "
                     "%"NPY_INTP_FMT" was given", size);
        return NULL;
    }

    if (self->type_num == NPY_UNICODE) {
        // unicode strings are 4 bytes per character
        if (npy_mul_sizes_with_overflow(&size, size, 4)) {
            PyErr_SetString(
                PyExc_TypeError,
                "Strings too large to store inside array.");
            return NULL;
        }
    }

    if (size > NPY_MAX_INT) {
        PyErr_SetString(PyExc_TypeError,
                        "Strings too large to store inside array.");
        return NULL;
    }

    PyArray_Descr *res = PyArray_DescrNewFromType(self->type_num);

    if (res == NULL) {
        return NULL;
    }

    res->elsize = (int)size;
    return (PyObject *)res;
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
        if (itemsize > NPY_MAX_INT || (
                cls->type_num == NPY_UNICODE && itemsize > NPY_MAX_INT / 4)) {
            PyErr_SetString(PyExc_TypeError,
                    "string too large to store inside array.");
            return NULL;
        }
        if (cls->type_num == NPY_UNICODE) {
            itemsize *= 4;
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
        return (PyArray_Descr *)void_obj->descr;
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


/*
 * For most builtin (and legacy) dtypes, the canonical property means to
 * ensure native byte-order.  (We do not care about metadata here.)
 */
static PyArray_Descr *
ensure_native_byteorder(PyArray_Descr *descr)
{
    if (PyArray_ISNBO(descr->byteorder)) {
        Py_INCREF(descr);
        return descr;
    }
    else {
        return PyArray_DescrNewByteorder(descr, NPY_NATIVE);
    }
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
        return NPY_DT_CALL_ensure_canonical(descr1);
    }
    else {
        return NPY_DT_CALL_ensure_canonical(descr2);
    }
}


static PyArray_Descr *
void_ensure_canonical(_PyArray_LegacyDescr *self)
{
    if (self->subarray != NULL) {
        PyArray_Descr *new_base = NPY_DT_CALL_ensure_canonical(
                self->subarray->base);
        if (new_base == NULL) {
            return NULL;
        }
        if (new_base == self->subarray->base) {
            /* just return self, no need to modify */
            Py_DECREF(new_base);
            Py_INCREF(self);
            return (PyArray_Descr *)self;
        }
        PyArray_Descr *new = PyArray_DescrNew((PyArray_Descr *)self);
        if (new == NULL) {
            return NULL;
        }
        Py_SETREF(((_PyArray_LegacyDescr *)new)->subarray->base, new_base);
        return new;
    }
    else if (self->names != NULL) {
        /*
         * This branch is fairly complex, since it needs to build a new
         * descriptor that is in canonical form.  This means that the new
         * descriptor should be an aligned struct if the old one was, and
         * otherwise it should be an unaligned struct.
         * Any unnecessary empty space is stripped from the struct.
         *
         * TODO: In principle we could/should try to provide the identity when
         *       no change is necessary. (Simple if we add a flag.)
         */
        Py_ssize_t field_num = PyTuple_GET_SIZE(self->names);

        _PyArray_LegacyDescr *new = (_PyArray_LegacyDescr *)PyArray_DescrNew(
                (PyArray_Descr *)self);
        if (new == NULL) {
            return NULL;
        }
        Py_SETREF(new->fields, PyDict_New());
        if (new->fields == NULL) {
            Py_DECREF(new);
            return NULL;
        }
        int aligned = PyDataType_FLAGCHK((PyArray_Descr *)new, NPY_ALIGNED_STRUCT);
        new->flags = new->flags & ~NPY_FROM_FIELDS;
        new->flags |= NPY_NEEDS_PYAPI;  /* always needed for field access */
        int totalsize = 0;
        int maxalign = 1;
        for (Py_ssize_t i = 0; i < field_num; i++) {
            PyObject *name = PyTuple_GET_ITEM(self->names, i);
            PyObject *tuple = PyDict_GetItem(self->fields, name); // noqa: borrowed-ref OK
            PyObject *new_tuple = PyTuple_New(PyTuple_GET_SIZE(tuple));
            PyArray_Descr *field_descr = NPY_DT_CALL_ensure_canonical(
                    (PyArray_Descr *)PyTuple_GET_ITEM(tuple, 0));
            if (field_descr == NULL) {
                Py_DECREF(new_tuple);
                Py_DECREF(new);
                return NULL;
            }
            new->flags |= field_descr->flags & NPY_FROM_FIELDS;
            PyTuple_SET_ITEM(new_tuple, 0, (PyObject *)field_descr);

            if (aligned) {
                totalsize = NPY_NEXT_ALIGNED_OFFSET(
                        totalsize, field_descr->alignment);
                maxalign = PyArray_MAX(maxalign, field_descr->alignment);
            }
            PyObject *offset_obj = PyLong_FromLong(totalsize);
            if (offset_obj == NULL) {
                Py_DECREF(new_tuple);
                Py_DECREF(new);
                return NULL;
            }
            PyTuple_SET_ITEM(new_tuple, 1, (PyObject *)offset_obj);
            if (PyTuple_GET_SIZE(tuple) == 3) {
                /* Be sure to set all items in the tuple before using it */
                PyObject *title = PyTuple_GET_ITEM(tuple, 2);
                Py_INCREF(title);
                PyTuple_SET_ITEM(new_tuple, 2, title);
                if (PyDict_SetItem(new->fields, title, new_tuple) < 0) {
                    Py_DECREF(new_tuple);
                    Py_DECREF(new);
                    return NULL;
                }
            }
            if (PyDict_SetItem(new->fields, name, new_tuple) < 0) {
                Py_DECREF(new_tuple);
                Py_DECREF(new);
                return NULL;
            }
            Py_DECREF(new_tuple);  /* Reference now owned by PyDataType_FIELDS(new) */
            totalsize += field_descr->elsize;
        }
        totalsize = NPY_NEXT_ALIGNED_OFFSET(totalsize, maxalign);
        new->elsize = totalsize;
        new->alignment = maxalign;
        return (PyArray_Descr *)new;
    }
    else {
        /* unstructured voids are always canonical. */
        Py_INCREF(self);
        return (PyArray_Descr *)self;
    }
}


static PyArray_Descr *
void_common_instance(_PyArray_LegacyDescr *descr1, _PyArray_LegacyDescr *descr2)
{
    if (descr1->subarray == NULL && descr1->names == NULL &&
            descr2->subarray == NULL && descr2->names == NULL) {
        if (descr1->elsize != descr2->elsize) {
            PyErr_SetString(npy_static_pydata.DTypePromotionError,
                    "Invalid type promotion with void datatypes of different "
                    "lengths. Use the `np.bytes_` datatype instead to pad the "
                    "shorter value with trailing zero bytes.");
            return NULL;
        }
        Py_INCREF(descr1);
        return (PyArray_Descr *)descr1;
    }

    if (descr1->names != NULL && descr2->names != NULL) {
        /* If both have fields promoting individual fields may be possible */
        if (npy_cache_import_runtime(
                    "numpy._core._internal", "_promote_fields",
                    &npy_runtime_imports._promote_fields) == -1) {
            return NULL;
        }
        PyObject *result = PyObject_CallFunctionObjArgs(
                npy_runtime_imports._promote_fields,
                descr1, descr2, NULL);
        if (result == NULL) {
            return NULL;
        }
        if (!PyObject_TypeCheck(result, Py_TYPE(descr1))) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Internal NumPy error: `_promote_fields` did not return "
                    "a valid descriptor object.");
            Py_DECREF(result);
            return NULL;
        }
        return (PyArray_Descr *)result;
    }
    else if (descr1->subarray != NULL && descr2->subarray != NULL) {
        int cmp = PyObject_RichCompareBool(
                descr1->subarray->shape, descr2->subarray->shape, Py_EQ);
        if (error_converting(cmp)) {
            return NULL;
        }
        if (!cmp) {
            PyErr_SetString(npy_static_pydata.DTypePromotionError,
                    "invalid type promotion with subarray datatypes "
                    "(shape mismatch).");
            return NULL;
        }
        PyArray_Descr *new_base = PyArray_PromoteTypes(
                descr1->subarray->base, descr2->subarray->base);
        if (new_base == NULL) {
            return NULL;
        }
        /*
         * If it is the same dtype and the container did not change, we might
         * as well preserve identity and metadata.  This could probably be
         * changed.
         */
        if (descr1 == descr2 && new_base == descr1->subarray->base) {
            Py_DECREF(new_base);
            Py_INCREF(descr1);
            return (PyArray_Descr *)descr1;
        }

        PyArray_Descr *new_descr = PyArray_DescrNew((PyArray_Descr *)descr1);
        if (new_descr == NULL) {
            Py_DECREF(new_base);
            return NULL;
        }
        Py_SETREF(((_PyArray_LegacyDescr *)new_descr)->subarray->base, new_base);
        return new_descr;
    }

    PyErr_SetString(npy_static_pydata.DTypePromotionError,
            "invalid type promotion with structured datatype(s).");
    return NULL;
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
    if (pytype == &PyFloat_Type ||
        pytype == &PyLong_Type ||
        pytype == &PyBool_Type ||
        pytype == &PyComplex_Type ||
        pytype == &PyUnicode_Type ||
        pytype == &PyBytes_Type)
    {
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
    assert(cls->type_num < NPY_NTYPES_LEGACY);
    if (NPY_UNLIKELY(!NPY_DT_is_legacy(other))) {
        /*
         * Deal with the non-legacy types we understand: python scalars.
         * These may have lower priority than the concrete inexact types,
         * but can change the type of the result (complex, float, int).
         * If our own DType is not numerical or has lower priority (e.g.
         * integer but abstract one is float), signal not implemented.
         */
        if (other == &PyArray_PyComplexDType) {
            if (PyTypeNum_ISCOMPLEX(cls->type_num)) {
                Py_INCREF(cls);
                return cls;
            }
            else if (cls->type_num == NPY_HALF || cls->type_num == NPY_FLOAT) {
                return NPY_DT_NewRef(&PyArray_CFloatDType);
            }
            else if (cls->type_num == NPY_DOUBLE) {
                return NPY_DT_NewRef(&PyArray_CDoubleDType);
            }
            else if (cls->type_num == NPY_LONGDOUBLE) {
                return NPY_DT_NewRef(&PyArray_CLongDoubleDType);
            }
        }
        else if (other == &PyArray_PyFloatDType) {
            if (PyTypeNum_ISCOMPLEX(cls->type_num)
                    || PyTypeNum_ISFLOAT(cls->type_num)) {
                Py_INCREF(cls);
                return cls;
            }
        }
        else if (other == &PyArray_PyLongDType) {
            if (PyTypeNum_ISCOMPLEX(cls->type_num)
                    || PyTypeNum_ISFLOAT(cls->type_num)
                    || PyTypeNum_ISINTEGER(cls->type_num)
                    || cls->type_num == NPY_TIMEDELTA) {
                Py_INCREF(cls);
                return cls;
            }
        }
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }
    if (other->type_num > cls->type_num) {
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
    assert(cls->type_num < NPY_NTYPES_LEGACY && cls != other);
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
    /*
     * Timedelta/datetime shouldn't actually promote at all.  That they
     * currently do means that we need additional hacks in the comparison
     * type resolver.  For comparisons we have to make sure we reject it
     * nicely in order to return an array of True/False values.
     */
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
 * @param name The name for the DType.
 * @param alias A second name which is also set to the new class for builtins
 *              (i.e. `np.types.LongDType` for `np.types.Int64DType`).
 *              Some may have more aliases, as `intp` is not its own thing,
 *              as of writing this, these are not added here.
 *
 * @returns A borrowed references to the new DType or NULL.
 */
NPY_NO_EXPORT PyArray_DTypeMeta *
dtypemeta_wrap_legacy_descriptor(
    _PyArray_LegacyDescr *descr, PyArray_ArrFuncs *arr_funcs,
    PyTypeObject *dtype_super_class, const char *name, const char *alias)
{
    int has_type_set = Py_TYPE(descr) == &PyArrayDescr_Type;

    if (!has_type_set) {
        /* Accept if the type was filled in from an existing builtin dtype */
        for (int i = 0; i < NPY_NTYPES_LEGACY; i++) {
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
        return NULL;
    }

    NPY_DType_Slots *dt_slots = PyMem_Malloc(sizeof(NPY_DType_Slots));
    if (dt_slots == NULL) {
        return NULL;
    }
    memset(dt_slots, '\0', sizeof(NPY_DType_Slots));
    dt_slots->get_constant = default_get_constant;

    PyArray_DTypeMeta *dtype_class = PyMem_Malloc(sizeof(PyArray_DTypeMeta));
    if (dtype_class == NULL) {
        PyMem_Free(dt_slots);
        return NULL;
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
            .tp_basicsize = sizeof(_PyArray_LegacyDescr),
            .tp_flags = Py_TPFLAGS_DEFAULT,
            .tp_base = NULL,  /* set below */
            .tp_new = (newfunc)legacy_dtype_default_new,
            .tp_doc = (
                "DType class corresponding to the scalar type and dtype of "
                "the same name.\n\n"
                "Please see `numpy.dtype` for the typical way to create\n"
                "dtype instances and :ref:`arrays.dtypes` for additional\n"
                "information."),
        },},
        .flags = NPY_DT_LEGACY,
        /* Further fields are not common between DTypes */
    };
    memcpy(dtype_class, &prototype, sizeof(PyArray_DTypeMeta));
    /* Fix name and superclass of the Type*/
    ((PyTypeObject *)dtype_class)->tp_name = name;
    ((PyTypeObject *)dtype_class)->tp_base = dtype_super_class,
    dtype_class->dt_slots = dt_slots;

    /* Let python finish the initialization */
    if (PyType_Ready((PyTypeObject *)dtype_class) < 0) {
        Py_DECREF(dtype_class);
        return NULL;
    }
    dt_slots->castingimpls = PyDict_New();
    if (dt_slots->castingimpls == NULL) {
        Py_DECREF(dtype_class);
        return NULL;
    }

    /*
     * Fill DTypeMeta information that varies between DTypes, any variable
     * type information would need to be set before PyType_Ready().
     */
    dtype_class->singleton = (PyArray_Descr *)descr;
    Py_INCREF(descr->typeobj);
    dtype_class->scalar_type = descr->typeobj;
    dtype_class->type_num = descr->type_num;
    dt_slots->f = *arr_funcs;

    /* Set default functions (correct for most dtypes, override below) */
    dt_slots->default_descr = nonparametric_default_descr;
    dt_slots->discover_descr_from_pyobject = (
        nonparametric_discover_descr_from_pyobject);
    dt_slots->is_known_scalar_type = python_builtins_are_known_scalar_types;
    dt_slots->common_dtype = default_builtin_common_dtype;
    dt_slots->common_instance = NULL;
    dt_slots->ensure_canonical = ensure_native_byteorder;
    dt_slots->get_fill_zero_loop = NULL;
    dt_slots->finalize_descr = NULL;
    // May be overwritten, but if not provide fallback via array struct hack.
    // `getitem` is a trickier because of structured dtypes returning views.
    if (dt_slots->f.setitem == NULL) {
        dt_slots->f.setitem = legacy_setitem_using_DType;
    }
    dt_slots->setitem = legacy_fallback_setitem;
    dt_slots->getitem = NULL;

    if (PyTypeNum_ISSIGNED(dtype_class->type_num)) {
        /* Convert our scalars (raise on too large unsigned and NaN, etc.) */
        dt_slots->is_known_scalar_type = signed_integers_is_known_scalar_types;
    }

    if (PyTypeNum_ISUSERDEF(descr->type_num)) {
        dt_slots->common_dtype = legacy_userdtype_common_dtype_function;
    }
    else if (descr->type_num == NPY_OBJECT) {
        dt_slots->common_dtype = object_common_dtype;
        dt_slots->get_fill_zero_loop = npy_object_get_fill_zero_loop;
        dt_slots->get_clear_loop = npy_get_clear_object_strided_loop;
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
            dt_slots->common_instance = (PyArrayDTypeMeta_CommonInstance *)void_common_instance;
            dt_slots->ensure_canonical = (PyArrayDTypeMeta_EnsureCanonical *)void_ensure_canonical;
            dt_slots->get_fill_zero_loop =
                    (PyArrayMethod_GetTraverseLoop *)npy_get_zerofill_void_and_legacy_user_dtype_loop;
            dt_slots->get_clear_loop =
                    (PyArrayMethod_GetTraverseLoop *)npy_get_clear_void_and_legacy_user_dtype_loop;
        }
        else {
            dt_slots->default_descr = string_and_unicode_default_descr;
            dt_slots->is_known_scalar_type = string_known_scalar_types;
            dt_slots->discover_descr_from_pyobject = (
                    string_discover_descr_from_pyobject);
            dt_slots->common_dtype = string_unicode_common_dtype;
            dt_slots->common_instance = string_unicode_common_instance;
            ((PyTypeObject*)dtype_class)->tp_new = (newfunc)string_unicode_new;
        }
    }

    if (PyTypeNum_ISNUMBER(descr->type_num)) {
        dtype_class->flags |= NPY_DT_NUMERIC;
    }

    if (_PyArray_MapPyTypeToDType(dtype_class, descr->typeobj,
            PyTypeNum_ISUSERDEF(dtype_class->type_num)) < 0) {
        Py_DECREF(dtype_class);
        return NULL;
    }

    /* Finally, replace the current class of the descr */
    Py_SET_TYPE(descr, (PyTypeObject *)dtype_class);

    /* And it to the types submodule if it is a builtin dtype */
    if (!PyTypeNum_ISUSERDEF(descr->type_num)) {
        if (npy_cache_import_runtime("numpy.dtypes", "_add_dtype_helper",
                                     &npy_runtime_imports._add_dtype_helper) == -1) {
            return NULL;
        }

        if (PyObject_CallFunction(
                npy_runtime_imports._add_dtype_helper,
                "Os", (PyObject *)dtype_class, alias) == NULL) {
            return NULL;
        }
    }
    else {
        // ensure the within dtype cast is populated for legacy user dtypes
        if (PyArray_GetCastingImpl(dtype_class, dtype_class) == NULL) {
            return NULL;
        }
    }

    return dtype_class;
}


static PyObject *
dtypemeta_get_abstract(PyArray_DTypeMeta *self, void *NPY_UNUSED(ignored)) {
    return PyBool_FromLong(NPY_DT_is_abstract(self));
}

static PyObject *
dtypemeta_get_legacy(PyArray_DTypeMeta *self, void *NPY_UNUSED(ignored)) {
    return PyBool_FromLong(NPY_DT_is_legacy(self));
}

static PyObject *
dtypemeta_get_parametric(PyArray_DTypeMeta *self, void *NPY_UNUSED(ignored)) {
    return PyBool_FromLong(NPY_DT_is_parametric(self));
}

static PyObject *
dtypemeta_get_is_numeric(PyArray_DTypeMeta *self, void *NPY_UNUSED(ignored)) {
    return PyBool_FromLong(NPY_DT_is_numeric(self));
}

/*
 * Simple exposed information, defined for each DType (class).
 */
static PyGetSetDef dtypemeta_getset[] = {
        {"_abstract", (getter)dtypemeta_get_abstract, NULL, NULL, NULL},
        {"_legacy", (getter)dtypemeta_get_legacy, NULL, NULL, NULL},
        {"_parametric", (getter)dtypemeta_get_parametric, NULL, NULL, NULL},
        {"_is_numeric", (getter)dtypemeta_get_is_numeric, NULL, NULL, NULL},
        {NULL, NULL, NULL, NULL, NULL}
};

static PyMemberDef dtypemeta_members[] = {
    {"type",
        T_OBJECT, offsetof(PyArray_DTypeMeta, scalar_type), READONLY,
        "scalar type corresponding to the DType."},
    {NULL, 0, 0, 0, NULL},
};

NPY_NO_EXPORT void
initialize_legacy_dtypemeta_aliases(_PyArray_LegacyDescr **_builtin_descrs) {
    _Bool_dtype = NPY_DTYPE(_builtin_descrs[NPY_BOOL]);
    _Byte_dtype = NPY_DTYPE(_builtin_descrs[NPY_BYTE]);
    _UByte_dtype = NPY_DTYPE(_builtin_descrs[NPY_UBYTE]);
    _Short_dtype = NPY_DTYPE(_builtin_descrs[NPY_SHORT]);
    _UShort_dtype = NPY_DTYPE(_builtin_descrs[NPY_USHORT]);
    _Int_dtype = NPY_DTYPE(_builtin_descrs[NPY_INT]);
    _UInt_dtype = NPY_DTYPE(_builtin_descrs[NPY_UINT]);
    _Long_dtype = NPY_DTYPE(_builtin_descrs[NPY_LONG]);
    _ULong_dtype = NPY_DTYPE(_builtin_descrs[NPY_ULONG]);
    _LongLong_dtype = NPY_DTYPE(_builtin_descrs[NPY_LONGLONG]);
    _ULongLong_dtype = NPY_DTYPE(_builtin_descrs[NPY_ULONGLONG]);
    _Int8_dtype = NPY_DTYPE(_builtin_descrs[NPY_INT8]);
    _UInt8_dtype = NPY_DTYPE(_builtin_descrs[NPY_UINT8]);
    _Int16_dtype = NPY_DTYPE(_builtin_descrs[NPY_INT16]);
    _UInt16_dtype = NPY_DTYPE(_builtin_descrs[NPY_UINT16]);
    _Int32_dtype = NPY_DTYPE(_builtin_descrs[NPY_INT32]);
    _UInt32_dtype = NPY_DTYPE(_builtin_descrs[NPY_UINT32]);
    _Int64_dtype = NPY_DTYPE(_builtin_descrs[NPY_INT64]);
    _UInt64_dtype = NPY_DTYPE(_builtin_descrs[NPY_UINT64]);
    _Intp_dtype = NPY_DTYPE(_builtin_descrs[NPY_INTP]);
    _UIntp_dtype = NPY_DTYPE(_builtin_descrs[NPY_UINTP]);
    _DefaultInt_dtype = NPY_DTYPE(_builtin_descrs[NPY_DEFAULT_INT]);
    _Half_dtype = NPY_DTYPE(_builtin_descrs[NPY_HALF]);
    _Float_dtype = NPY_DTYPE(_builtin_descrs[NPY_FLOAT]);
    _Double_dtype = NPY_DTYPE(_builtin_descrs[NPY_DOUBLE]);
    _LongDouble_dtype = NPY_DTYPE(_builtin_descrs[NPY_LONGDOUBLE]);
    _CFloat_dtype = NPY_DTYPE(_builtin_descrs[NPY_CFLOAT]);
    _CDouble_dtype = NPY_DTYPE(_builtin_descrs[NPY_CDOUBLE]);
    _CLongDouble_dtype = NPY_DTYPE(_builtin_descrs[NPY_CLONGDOUBLE]);
    // NPY_STRING is the legacy python2 name
    _Bytes_dtype = NPY_DTYPE(_builtin_descrs[NPY_STRING]);
    _Unicode_dtype = NPY_DTYPE(_builtin_descrs[NPY_UNICODE]);
    _Datetime_dtype = NPY_DTYPE(_builtin_descrs[NPY_DATETIME]);
    _Timedelta_dtype = NPY_DTYPE(_builtin_descrs[NPY_TIMEDELTA]);
    _Object_dtype = NPY_DTYPE(_builtin_descrs[NPY_OBJECT]);
    _Void_dtype = NPY_DTYPE(_builtin_descrs[NPY_VOID]);
}

NPY_NO_EXPORT PyTypeObject PyArrayDTypeMeta_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy._DTypeMeta",
    .tp_basicsize = sizeof(PyArray_DTypeMeta),
    .tp_dealloc = (destructor)dtypemeta_dealloc,
    /* Types are garbage collected (see dtypemeta_is_gc documentation) */
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_doc = "Preliminary NumPy API: The Type of NumPy DTypes (metaclass)",
    .tp_traverse = (traverseproc)dtypemeta_traverse,
    .tp_members = dtypemeta_members,
    .tp_getset = dtypemeta_getset,
    .tp_base = NULL,  /* set to PyType_Type at import time */
    .tp_init = (initproc)dtypemeta_init,
    .tp_alloc = dtypemeta_alloc,
    .tp_new = dtypemeta_new,
    .tp_is_gc = dtypemeta_is_gc,
};

PyArray_DTypeMeta *_Bool_dtype = NULL;
PyArray_DTypeMeta *_Byte_dtype = NULL;
PyArray_DTypeMeta *_UByte_dtype = NULL;
PyArray_DTypeMeta *_Short_dtype = NULL;
PyArray_DTypeMeta *_UShort_dtype = NULL;
PyArray_DTypeMeta *_Int_dtype = NULL;
PyArray_DTypeMeta *_UInt_dtype = NULL;
PyArray_DTypeMeta *_Long_dtype = NULL;
PyArray_DTypeMeta *_ULong_dtype = NULL;
PyArray_DTypeMeta *_LongLong_dtype = NULL;
PyArray_DTypeMeta *_ULongLong_dtype = NULL;
PyArray_DTypeMeta *_Int8_dtype = NULL;
PyArray_DTypeMeta *_UInt8_dtype = NULL;
PyArray_DTypeMeta *_Int16_dtype = NULL;
PyArray_DTypeMeta *_UInt16_dtype = NULL;
PyArray_DTypeMeta *_Int32_dtype = NULL;
PyArray_DTypeMeta *_UInt32_dtype = NULL;
PyArray_DTypeMeta *_Int64_dtype = NULL;
PyArray_DTypeMeta *_UInt64_dtype = NULL;
PyArray_DTypeMeta *_Intp_dtype = NULL;
PyArray_DTypeMeta *_UIntp_dtype = NULL;
PyArray_DTypeMeta *_DefaultInt_dtype = NULL;
PyArray_DTypeMeta *_Half_dtype = NULL;
PyArray_DTypeMeta *_Float_dtype = NULL;
PyArray_DTypeMeta *_Double_dtype = NULL;
PyArray_DTypeMeta *_LongDouble_dtype = NULL;
PyArray_DTypeMeta *_CFloat_dtype = NULL;
PyArray_DTypeMeta *_CDouble_dtype = NULL;
PyArray_DTypeMeta *_CLongDouble_dtype = NULL;
PyArray_DTypeMeta *_Bytes_dtype = NULL;
PyArray_DTypeMeta *_Unicode_dtype = NULL;
PyArray_DTypeMeta *_Datetime_dtype = NULL;
PyArray_DTypeMeta *_Timedelta_dtype = NULL;
PyArray_DTypeMeta *_Object_dtype = NULL;
PyArray_DTypeMeta *_Void_dtype = NULL;



/*NUMPY_API
 * Fetch the ArrFuncs struct which now lives on the DType and not the
 * descriptor.  Use of this struct should be avoided but remains necessary
 * for certain functionality.
 *
 * The use of any slot besides getitem, setitem, copyswap, and copyswapn
 * is only valid after checking for NULL.  Checking for NULL is generally
 * encouraged.
 *
 * This function is exposed with an underscore "privately" because the
 * public version is a static inline function which only calls the function
 * on 2.x but directly accesses the `descr` struct on 1.x.
 * Once 1.x backwards compatibility is gone, it should be exported without
 * the underscore directly.
 * Internally, we define a private inline function `PyDataType_GetArrFuncs`
 * for convenience as we are allowed to access the `DType` slots directly.
 */
NPY_NO_EXPORT PyArray_ArrFuncs *
_PyDataType_GetArrFuncs(const PyArray_Descr *descr)
{
    return PyDataType_GetArrFuncs(descr);
}
