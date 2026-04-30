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
#include "npy_pycompat.h"


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


NPY_NO_EXPORT PyArray_DTypeMeta *PyArray_IntAbstractDTypePtr = NULL;
NPY_NO_EXPORT PyArray_DTypeMeta *PyArray_FloatAbstractDTypePtr = NULL;
NPY_NO_EXPORT PyArray_DTypeMeta *PyArray_ComplexAbstractDTypePtr = NULL;
NPY_NO_EXPORT PyArray_DTypeMeta *PyArray_PyLongDTypePtr = NULL;
NPY_NO_EXPORT PyArray_DTypeMeta *PyArray_PyFloatDTypePtr = NULL;
NPY_NO_EXPORT PyArray_DTypeMeta *PyArray_PyComplexDTypePtr = NULL;


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
 * Create a heap-type DType class via ``PyType_FromMetaclass`` and fill in
 * the NumPy-specific fields.  If ``slots`` is NULL we allocate an empty
 * ``NPY_DType_Slots`` (abstract DTypes have no functional slots; in
 * principle we should route everything through ``DTypeMetaInitFromSpec``
 * here, but for now we just allocate directly).  When ``scalar_type`` is
 * non-NULL the new DType is also registered for scalar discovery.
 */
static PyArray_DTypeMeta *
make_raw_dtype(const char *name, PyTypeObject *base,
               npy_uint64 flags, NPY_DType_Slots *slots,
               PyTypeObject *scalar_type)
{
    PyType_Slot type_slots[] = {
        {Py_tp_base, base},
        {0, NULL},
    };
    PyType_Spec spec = {
        .name = name,
        .basicsize = sizeof(PyArray_Descr),
        .itemsize = 0,
        .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
        .slots = type_slots,
    };
    if (flags & NPY_DT_ABSTRACT) {
        /* abstract ones can subclass in C but also disallow instant here */
        spec.flags |= Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DISALLOW_INSTANTIATION;
    }
    PyArray_DTypeMeta *dt = (PyArray_DTypeMeta *)PyType_FromMetaclass(
            &PyArrayDTypeMeta_Type, NULL, &spec, NULL);
    if (dt == NULL) {
        return NULL;
    }
    if (slots == NULL) {
        slots = PyMem_Calloc(1, sizeof(NPY_DType_Slots));
        if (slots == NULL) {
            Py_DECREF(dt);
            PyErr_NoMemory();
            return NULL;
        }
    }
    dt->dt_slots = slots;
    dt->type_num = -1;
    Py_XINCREF(scalar_type);
    dt->scalar_type = scalar_type;
    dt->singleton = NULL;
    dt->flags = flags;
    NpyUnstable_SetImmortal((PyObject *)dt);

    if (scalar_type != NULL) {
        if (_PyArray_MapPyTypeToDType(dt, scalar_type, NPY_FALSE) < 0) {
            Py_DECREF(dt);
            return NULL;
        }
    }
    return dt;
}


/*
 * Create the abstract integer/float/complex DType classes (which the
 * legacy concrete DTypes inherit from in ``arraytypes.c.src``) and the
 * implicit DType classes for Python ``int``/``float``/``complex``
 * literals, and register the latter for scalar discovery.
 *
 * Must be called before ``set_typeinfo``: ``dtypemeta_wrap_legacy_descriptor``
 * inherits from the abstract DTypes created here.
 */
NPY_NO_EXPORT int
initialize_abstract_dtypes(void)
{
    struct dtype_spec {
        const char *name;
        PyArray_DTypeMeta **out;
        /* Indirected so Py-scalar entries below can reference an abstract
         * DType created earlier in the same loop iteration. */
        PyTypeObject **base_ptr;
        npy_uint64 flags;
        NPY_DType_Slots *slots;
        PyTypeObject *scalar_type;
    };
    PyTypeObject *descr_base = (PyTypeObject *)&PyArrayDescr_Type;

    struct dtype_spec specs[] = {
        /* Abstract DTypes; concrete legacy DTypes may inherit from these. */
        {"numpy.dtypes._IntegerAbstractDType", &PyArray_IntAbstractDTypePtr,
            &descr_base, NPY_DT_ABSTRACT, NULL, NULL},
        {"numpy.dtypes._FloatAbstractDType", &PyArray_FloatAbstractDTypePtr,
            &descr_base, NPY_DT_ABSTRACT, NULL, NULL},
        {"numpy.dtypes._ComplexAbstractDType", &PyArray_ComplexAbstractDTypePtr,
            &descr_base, NPY_DT_ABSTRACT, NULL, NULL},
        /* Py-scalar DTypes; bases are the abstract DTypes created above. */
        {"numpy.dtypes._PyLongDType", &PyArray_PyLongDTypePtr,
            (PyTypeObject **)&PyArray_IntAbstractDTypePtr,
            0, &pylongdtype_slots, &PyLong_Type},
        {"numpy.dtypes._PyFloatDType", &PyArray_PyFloatDTypePtr,
            (PyTypeObject **)&PyArray_FloatAbstractDTypePtr,
            0, &pyfloatdtype_slots, &PyFloat_Type},
        {"numpy.dtypes._PyComplexDType", &PyArray_PyComplexDTypePtr,
            (PyTypeObject **)&PyArray_ComplexAbstractDTypePtr,
            0, &pycomplexdtype_slots, &PyComplex_Type},
    };
    for (size_t i = 0; i < sizeof(specs) / sizeof(specs[0]); i++) {
        *specs[i].out = make_raw_dtype(
                specs[i].name, *specs[i].base_ptr, specs[i].flags,
                specs[i].slots, specs[i].scalar_type);
        if (*specs[i].out == NULL) {
            return -1;
        }
    }
    return 0;
}


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
