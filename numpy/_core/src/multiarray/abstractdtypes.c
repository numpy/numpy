#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#include "dtypemeta.h"
#include "abstractdtypes.h"
#include "array_coercion.h"
#include "common.h"
#include "npy_static_data.h"


/*
 * Storage for the abstract DType classes.  These are created dynamically in
 * ``initialize_abstract_dtypes`` (called from module init before
 * ``set_typeinfo``) so concrete legacy DType classes can use them as
 * ``tp_base`` directly.  See ``abstractdtypes.h`` for the
 * ``PyArray_*AbstractDType`` macro accessors.
 */
NPY_NO_EXPORT PyArray_DTypeMeta *_NumericAbstract_dtype = NULL;
NPY_NO_EXPORT PyArray_DTypeMeta *_IntegerAbstract_dtype = NULL;
NPY_NO_EXPORT PyArray_DTypeMeta *_SignedIntegerAbstract_dtype = NULL;
NPY_NO_EXPORT PyArray_DTypeMeta *_UnsignedIntegerAbstract_dtype = NULL;
NPY_NO_EXPORT PyArray_DTypeMeta *_InexactAbstract_dtype = NULL;
NPY_NO_EXPORT PyArray_DTypeMeta *_FloatAbstract_dtype = NULL;
NPY_NO_EXPORT PyArray_DTypeMeta *_ComplexAbstract_dtype = NULL;

/*
 * Storage for the implicit DType classes for Python ``int``, ``float``,
 * ``complex`` literals.  Created in ``initialize_abstract_dtypes`` via
 * ``PyType_FromMetaclass`` so they can subclass the abstract DTypes (which
 * are themselves heap types).
 */
NPY_NO_EXPORT PyArray_DTypeMeta *_PyLongDType = NULL;
NPY_NO_EXPORT PyArray_DTypeMeta *_PyFloatDType = NULL;
NPY_NO_EXPORT PyArray_DTypeMeta *_PyComplexDType = NULL;


/*
 * Create a new abstract DType class with the given fully qualified name and
 * abstract base.  The resulting class has ``NPY_DT_ABSTRACT`` set, a freshly
 * allocated ``dt_slots``, and inherits ``NPY_DT_NUMERIC`` from its base.
 *
 * Additional flags on top of ``NPY_DT_ABSTRACT`` (and the inherited
 * ``NPY_DT_NUMERIC``) can be set via ``extra_flags``.
 *
 * These abstract DTypes are "pure" classification DTypes used by
 * ``isinstance`` / ``issubclass`` and the array-API ``isdtype`` /
 * ``issubdtype`` queries; they do not provide descriptor discovery,
 * common DType, etc.
 */
static PyArray_DTypeMeta *
create_abstract_dtype(const char *name, PyArray_DTypeMeta *base,
                      npy_uint64 extra_flags)
{
    if (base != (PyArray_DTypeMeta *)&PyArrayDescr_Type
            && (Py_TYPE((PyObject *)base) != &PyArrayDTypeMeta_Type
                || !NPY_DT_is_abstract(base))) {
        PyErr_SetString(PyExc_TypeError,
                "create_abstract_dtype: base must be np.dtype or an abstract "
                "DType class");
        return NULL;
    }

    NPY_DType_Slots *dt_slots = PyMem_Calloc(1, sizeof(NPY_DType_Slots));
    if (dt_slots == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    PyType_Slot slots[] = {{0, NULL}};
    PyType_Spec spec = {
        .name = name,
        .basicsize = (int)sizeof(PyArray_Descr),
        .itemsize = 0,
        .flags = (Py_TPFLAGS_DEFAULT
                  | Py_TPFLAGS_BASETYPE
                  | Py_TPFLAGS_IMMUTABLETYPE
                  | Py_TPFLAGS_DISALLOW_INSTANTIATION),
        .slots = slots,
    };
    PyObject *bases = PyTuple_Pack(1, (PyObject *)base);
    if (bases == NULL) {
        PyMem_Free(dt_slots);
        return NULL;
    }
    PyArray_DTypeMeta *dt = (PyArray_DTypeMeta *)PyType_FromMetaclass(
            &PyArrayDTypeMeta_Type, NULL, &spec, bases);
    Py_DECREF(bases);
    if (dt == NULL) {
        PyMem_Free(dt_slots);
        return NULL;
    }
    dt->dt_slots = dt_slots;
    dt->type_num = -1;
    dt->scalar_type = NULL;
    dt->singleton = NULL;
    npy_uint64 inherited_flags = base == (PyArray_DTypeMeta *)&PyArrayDescr_Type
            ? 0 : (base->flags & NPY_DT_NUMERIC);
    dt->flags = NPY_DT_ABSTRACT | inherited_flags | extra_flags;
    return dt;
}


/*
 * Specs for the abstract DType classes that mirror the array-API "kind"
 * hierarchy.  Order matters: each entry's ``base`` must come earlier in the
 * list (or be NULL, meaning ``numpy.dtype`` itself).
 */
typedef struct {
    const char *name;
    PyArray_DTypeMeta **out;
    PyArray_DTypeMeta **base;  /* NULL means numpy.dtype (PyArrayDescr_Type) */
    npy_uint64 extra_flags;
} abstract_dtype_spec;

static abstract_dtype_spec abstract_dtype_specs[] = {
    {"numpy.dtypes.NumericAbstractDType",
     &_NumericAbstract_dtype, NULL, NPY_DT_NUMERIC},
    {"numpy.dtypes.IntegerAbstractDType",
     &_IntegerAbstract_dtype, &_NumericAbstract_dtype, 0},
    {"numpy.dtypes.SignedIntegerAbstractDType",
     &_SignedIntegerAbstract_dtype, &_IntegerAbstract_dtype, 0},
    {"numpy.dtypes.UnsignedIntegerAbstractDType",
     &_UnsignedIntegerAbstract_dtype, &_IntegerAbstract_dtype, 0},
    {"numpy.dtypes.InexactAbstractDType",
     &_InexactAbstract_dtype, &_NumericAbstract_dtype, 0},
    {"numpy.dtypes.FloatingAbstractDType",
     &_FloatAbstract_dtype, &_InexactAbstract_dtype, 0},
    {"numpy.dtypes.ComplexFloatingAbstractDType",
     &_ComplexAbstract_dtype, &_InexactAbstract_dtype, 0},
};




static inline PyArray_Descr *
int_default_descriptor(PyArray_DTypeMeta* NPY_UNUSED(cls))
{
    return PyArray_DescrFromType(NPY_INTP);
}

static PyArray_Descr *
discover_descriptor_from_pylong(
        PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyLong_Check(obj));
    /*
     * We check whether long is good enough. If not, check longlong and
     * unsigned long before falling back to `object`.
     */
    long long value = PyLong_AsLongLong(obj);
    if (error_converting(value)) {
        PyErr_Clear();
    }
    else {
        if (NPY_MIN_INTP <= value && value <= NPY_MAX_INTP) {
            return PyArray_DescrFromType(NPY_INTP);
        }
        return PyArray_DescrFromType(NPY_LONGLONG);
    }

    unsigned long long uvalue = PyLong_AsUnsignedLongLong(obj);
    if (uvalue == (unsigned long long)-1 && PyErr_Occurred()){
        PyErr_Clear();
    }
    else {
        return PyArray_DescrFromType(NPY_ULONGLONG);
    }

    return PyArray_DescrFromType(NPY_OBJECT);
}


static inline PyArray_Descr *
float_default_descriptor(PyArray_DTypeMeta* NPY_UNUSED(cls))
{
    return PyArray_DescrFromType(NPY_DOUBLE);
}


static PyArray_Descr*
discover_descriptor_from_pyfloat(
        PyArray_DTypeMeta* NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyFloat_CheckExact(obj));
    return PyArray_DescrFromType(NPY_DOUBLE);
}

static inline PyArray_Descr *
complex_default_descriptor(PyArray_DTypeMeta* NPY_UNUSED(cls))
{
    return PyArray_DescrFromType(NPY_CDOUBLE);
}

static PyArray_Descr*
discover_descriptor_from_pycomplex(
        PyArray_DTypeMeta* NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyComplex_CheckExact(obj));
    return PyArray_DescrFromType(NPY_COMPLEX128);
}


/* Forward declarations of the static slot tables defined further down. */
static NPY_DType_Slots pylongdtype_slots;
static NPY_DType_Slots pyfloatdtype_slots;
static NPY_DType_Slots pycomplexdtype_slots;


NPY_NO_EXPORT int
initialize_abstract_dtypes(void)
{
    /*
     * Create the abstract DType classes that mirror the array-API "kind"
     * hierarchy (NumericAbstractDType -> IntegerAbstractDType ->
     * SignedIntegerAbstractDType / UnsignedIntegerAbstractDType, and
     * NumericAbstractDType -> InexactAbstractDType -> FloatAbstractDType /
     * ComplexAbstractDType) and expose them on ``numpy.dtypes``.  This must
     * run before ``set_typeinfo`` so the legacy concrete DType classes can
     * use these abstracts as ``tp_base``.
     */
    if (npy_cache_import_runtime("numpy.dtypes", "_add_dtype_helper",
                                 &npy_runtime_imports._add_dtype_helper) < 0) {
        return -1;
    }
    for (size_t i = 0; i < Py_ARRAY_LENGTH(abstract_dtype_specs); ++i) {
        abstract_dtype_spec *spec = &abstract_dtype_specs[i];
        PyArray_DTypeMeta *base = (spec->base == NULL)
                ? (PyArray_DTypeMeta *)&PyArrayDescr_Type
                : *spec->base;
        PyArray_DTypeMeta *dt = create_abstract_dtype(
                spec->name, base, spec->extra_flags);
        if (dt == NULL) {
            return -1;
        }
        *spec->out = dt;
        if (PyObject_CallFunctionObjArgs(
                    npy_runtime_imports._add_dtype_helper,
                    (PyObject *)dt,
                    Py_None,
                    NULL) == NULL) {
            return -1;
        }
    }

    /*
     * Set up the implicit DType classes for Python ``int``, ``float``, and
     * ``complex`` values used by value-based promotion.  These are heap
     * types as well (created via ``PyType_FromMetaclass``) so they can
     * directly subclass the corresponding abstract DType class -- this is
     * what makes ``isinstance(int_dtype_instance, IntegerAbstractDType)``
     * work and what lets ufunc loop dispatch find loops registered against
     * the abstract DType when given a Python literal.
     */
    struct {
        const char *name;
        PyArray_DTypeMeta **out;
        PyArray_DTypeMeta *base;
        NPY_DType_Slots *slots;
        PyTypeObject *scalar_type;
    } pyscalar_specs[] = {
        {"numpy.dtypes._PyLongDType", &_PyLongDType,
         _IntegerAbstract_dtype, &pylongdtype_slots, &PyLong_Type},
        {"numpy.dtypes._PyFloatDType", &_PyFloatDType,
         _FloatAbstract_dtype, &pyfloatdtype_slots, &PyFloat_Type},
        {"numpy.dtypes._PyComplexDType", &_PyComplexDType,
         _ComplexAbstract_dtype, &pycomplexdtype_slots, &PyComplex_Type},
    };
    for (size_t i = 0; i < Py_ARRAY_LENGTH(pyscalar_specs); ++i) {
        PyType_Slot slots[] = {{0, NULL}};
        PyType_Spec spec = {
            .name = pyscalar_specs[i].name,
            .basicsize = (int)sizeof(PyArray_Descr),
            .itemsize = 0,
            .flags = (Py_TPFLAGS_DEFAULT
                      | Py_TPFLAGS_IMMUTABLETYPE
                      | Py_TPFLAGS_DISALLOW_INSTANTIATION),
            .slots = slots,
        };
        PyObject *bases = PyTuple_Pack(
                1, (PyObject *)pyscalar_specs[i].base);
        if (bases == NULL) {
            return -1;
        }
        PyArray_DTypeMeta *dt = (PyArray_DTypeMeta *)PyType_FromMetaclass(
                &PyArrayDTypeMeta_Type, NULL, &spec, bases);
        Py_DECREF(bases);
        if (dt == NULL) {
            return -1;
        }
        NPY_DType_Slots *dtype_slots = PyMem_Calloc(1, sizeof(NPY_DType_Slots));
        if (dtype_slots == NULL) {
            Py_DECREF(dt);
            PyErr_NoMemory();
            return -1;
        }
        *dtype_slots = *pyscalar_specs[i].slots;
        dt->dt_slots = dtype_slots;
        dt->type_num = -1;
        Py_INCREF(pyscalar_specs[i].scalar_type);
        dt->scalar_type = pyscalar_specs[i].scalar_type;
        dt->singleton = NULL;
        dt->flags = NPY_DT_NUMERIC;

        if (_PyArray_MapPyTypeToDType(
                dt, pyscalar_specs[i].scalar_type, NPY_FALSE) < 0) {
            Py_DECREF(dt);
            return -1;
        }
        *pyscalar_specs[i].out = dt;
    }
    return 0;
}


/*
 * Map ``str``, ``bytes``, and ``bool`` (for which we do not need abstract
 * versions) to their NumPy DTypes.  Must run after ``set_typeinfo`` because
 * it looks up the legacy DType classes via ``typenum_to_dtypemeta``.
 *
 * TODO: The ``is_known_scalar_type`` function is considered preliminary,
 *       the same could be achieved e.g. with additional abstract DTypes.
 */
NPY_NO_EXPORT int
map_legacy_pytypes_to_dtypes(void)
{
    PyArray_DTypeMeta *dtype;
    dtype = typenum_to_dtypemeta(NPY_UNICODE);
    if (_PyArray_MapPyTypeToDType(dtype, &PyUnicode_Type, NPY_FALSE) < 0) {
        return -1;
    }
    dtype = typenum_to_dtypemeta(NPY_STRING);
    if (_PyArray_MapPyTypeToDType(dtype, &PyBytes_Type, NPY_FALSE) < 0) {
        return -1;
    }
    dtype = typenum_to_dtypemeta(NPY_BOOL);
    if (_PyArray_MapPyTypeToDType(dtype, &PyBool_Type, NPY_FALSE) < 0) {
        return -1;
    }
    return 0;
}


/*
 * The following functions define the "common DType" for the abstract dtypes.
 *
 * Note that the logic with respect to the "higher" dtypes such as floats
 * could likely be more logically defined for them, but since NumPy dtypes
 * largely "know" each other, that is not necessary.
 */
static PyArray_DTypeMeta *
int_common_dtype(PyArray_DTypeMeta *NPY_UNUSED(cls), PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES_LEGACY) {
        if (other->type_num == NPY_BOOL) {
            /* Use the default integer for bools: */
            return NPY_DT_NewRef(&PyArray_IntpDType);
        }
    }
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        PyArray_DTypeMeta *uint8_dt = &PyArray_UInt8DType;
        PyArray_DTypeMeta *res = NPY_DT_CALL_common_dtype(other, uint8_dt);
        if (res == NULL) {
            PyErr_Clear();
        }
        else if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            Py_DECREF(res);
        }
        else {
            return res;
        }
        /* Try again with `int8`, an error may have been set, though */
        PyArray_DTypeMeta *int8_dt = &PyArray_Int8DType;
        res = NPY_DT_CALL_common_dtype(other, int8_dt);
        if (res == NULL) {
            PyErr_Clear();
        }
        else if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            Py_DECREF(res);
        }
        else {
            return res;
        }
        /* And finally, we will try the default integer, just for sports... */
        PyArray_DTypeMeta *default_int = &PyArray_IntpDType;
        res = NPY_DT_CALL_common_dtype(other, default_int);
        if (res == NULL) {
            PyErr_Clear();
        }
        return res;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_DTypeMeta *
float_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES_LEGACY) {
        if (other->type_num == NPY_BOOL || PyTypeNum_ISINTEGER(other->type_num)) {
            /* Use the default integer for bools and ints: */
            return NPY_DT_NewRef(&PyArray_DoubleDType);
        }
    }
    else if (other == &PyArray_PyLongDType) {
        Py_INCREF(cls);
        return cls;
    }
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        PyArray_DTypeMeta *half_dt = &PyArray_HalfDType;
        PyArray_DTypeMeta *res = NPY_DT_CALL_common_dtype(other, half_dt);
        if (res == NULL) {
            PyErr_Clear();
        }
        else if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            Py_DECREF(res);
        }
        else {
            return res;
        }
        /* Retry with double (the default float) */
        PyArray_DTypeMeta *double_dt = &PyArray_DoubleDType;
        res = NPY_DT_CALL_common_dtype(other, double_dt);
        return res;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_DTypeMeta *
complex_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num < NPY_NTYPES_LEGACY) {
        if (other->type_num == NPY_BOOL ||
                PyTypeNum_ISINTEGER(other->type_num)) {
            /* Use the default integer for bools and ints: */
            return NPY_DT_NewRef(&PyArray_CDoubleDType);
        }
    }
    else if (NPY_DT_is_legacy(other)) {
        /* This is a back-compat fallback to usually do the right thing... */
        PyArray_DTypeMeta *cfloat_dt = &PyArray_CFloatDType;
        PyArray_DTypeMeta *res = NPY_DT_CALL_common_dtype(other, cfloat_dt);
        if (res == NULL) {
            PyErr_Clear();
        }
        else if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            Py_DECREF(res);
        }
        else {
            return res;
        }
        /* Retry with cdouble (the default complex) */
        PyArray_DTypeMeta *cdouble_dt = &PyArray_CDoubleDType;
        res = NPY_DT_CALL_common_dtype(other, cdouble_dt);
        return res;

    }
    else if (other == &PyArray_PyLongDType ||
             other == &PyArray_PyFloatDType) {
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


/*
 * Static slot tables for the implicit Python-scalar DTypes.  They are
 * referenced from the heap-allocated DType classes created in
 * ``initialize_abstract_dtypes``.
 */
static NPY_DType_Slots pylongdtype_slots = {
    .discover_descr_from_pyobject = discover_descriptor_from_pylong,
    .default_descr = int_default_descriptor,
    .common_dtype = int_common_dtype,
};

static NPY_DType_Slots pyfloatdtype_slots = {
    .discover_descr_from_pyobject = discover_descriptor_from_pyfloat,
    .default_descr = float_default_descriptor,
    .common_dtype = float_common_dtype,
};

static NPY_DType_Slots pycomplexdtype_slots = {
    .discover_descr_from_pyobject = discover_descriptor_from_pycomplex,
    .default_descr = complex_default_descriptor,
    .common_dtype = complex_common_dtype,
};


/*
 * Additional functions to deal with Python literal int, float, complex
 */
/*
 * This function takes an existing array operand and if the new descr does
 * not match, replaces it with a new array that has the correct descriptor
 * and holds exactly the scalar value.
 */
NPY_NO_EXPORT int
npy_update_operand_for_scalar(
    PyArrayObject **operand, PyObject *scalar, PyArray_Descr *descr,
    NPY_CASTING casting)
{
    if (PyArray_EquivTypes(PyArray_DESCR(*operand), descr)) {
        /*
        * TODO: This is an unfortunate work-around for legacy type resolvers
        *       (see `convert_ufunc_arguments` in `ufunc_object.c`), that
        *       currently forces us to replace the array.
        */
        if (!(PyArray_FLAGS(*operand) & NPY_ARRAY_WAS_PYTHON_INT)) {
            return 0;
        }
    }
    else if (NPY_UNLIKELY(casting == NPY_EQUIV_CASTING) &&
             descr->type_num != NPY_OBJECT) {
        /*
         * incredibly niche, but users could pass equiv casting and we
         * actually need to cast.  Let object pass (technically correct) but
         * in all other cases, we don't technically consider equivalent.
         * NOTE(seberg): I don't think we should be beholden to this logic.
         */
        PyErr_Format(PyExc_TypeError,
            "cannot cast Python %s to %S under the casting rule 'equiv'",
            Py_TYPE(scalar)->tp_name, descr);
        return -1;
    }

    Py_INCREF(descr);
    PyArrayObject *new = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, descr, 0, NULL, NULL, NULL, 0, NULL);
    Py_SETREF(*operand, new);
    if (*operand == NULL) {
        return -1;
    }
    if (scalar == NULL) {
        /* The ufunc.resolve_dtypes paths can go here.  Anything should go. */
        return 0;
    }
    return PyArray_SETITEM(new, PyArray_BYTES(*operand), scalar);
}


/*
 * When a user passed a Python literal (int, float, complex), special promotion
 * rules mean that we don't know the exact descriptor that should be used.
 *
 * Typically, this just doesn't really matter.  Unfortunately, there are two
 * exceptions:
 * 1. The user might have passed `signature=` which may not be compatible.
 *    In that case, we cannot really assume "safe" casting.
 * 2. It is at least fathomable that a DType doesn't deal with this directly.
 *    or that using the original int64/object is wrong in the type resolution.
 *
 * The solution is to assume that we can use the common DType of the signature
 * and the Python scalar DType (`in_DT`) as a safe intermediate.
 */
NPY_NO_EXPORT PyArray_Descr *
npy_find_descr_for_scalar(
    PyObject *scalar, PyArray_Descr *original_descr,
    PyArray_DTypeMeta *in_DT, PyArray_DTypeMeta *op_DT)
{
    PyArray_Descr *res;
    /* There is a good chance, descriptors already match... */
    if (NPY_DTYPE(original_descr) == op_DT) {
        Py_INCREF(original_descr);
        return original_descr;
    }

    PyArray_DTypeMeta *common = PyArray_CommonDType(in_DT, op_DT);
    if (common == NULL) {
        PyErr_Clear();
        /* This is fine.  We simply assume the original descr is viable. */
        Py_INCREF(original_descr);
        return original_descr;
    }
    /* A very likely case is that there is nothing to do: */
    if (NPY_DTYPE(original_descr) == common) {
        Py_DECREF(common);
        Py_INCREF(original_descr);
        return original_descr;
    }
    if (!NPY_DT_is_parametric(common) ||
            /* In some paths we only have a scalar type, can't discover */
            scalar == NULL ||
            /* If the DType doesn't know the scalar type, guess at default. */
            !NPY_DT_CALL_is_known_scalar_type(common, Py_TYPE(scalar))) {
        if (common->singleton != NULL) {
            res = common->singleton;
            Py_INCREF(res);
        }
        else {
            res = NPY_DT_CALL_default_descr(common);
        }
    }
    else {
        res = NPY_DT_CALL_discover_descr_from_pyobject(common, scalar);
    }

    Py_DECREF(common);
    return res;
}
