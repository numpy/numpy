/*
 * This file implements a basic scaled float64 DType.  The reason is to have
 * a simple parametric DType for testing.  It is not meant to be a useful
 * DType by itself, but due to the scaling factor has similar properties as
 * a Unit DType.
 *
 * The code here should be seen as a work in progress.  Some choices are made
 * to test certain code paths, but that does not mean that they must not
 * be modified.
 *
 * NOTE: The tests were initially written using private API and ABI, ideally
 *       they should be replaced/modified with versions using public API.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"

#include "array_method.h"
#include "common.h"
#include "numpy/npy_math.h"
#include "convert_datatype.h"
#include "dtypemeta.h"
#include "dispatching.h"


/* TODO: from wrapping_array_method.c, use proper public header eventually  */
NPY_NO_EXPORT int
PyUFunc_AddWrappingLoop(PyObject *ufunc_obj,
        PyArray_DTypeMeta *new_dtypes[], PyArray_DTypeMeta *wrapped_dtypes[],
        translate_given_descrs_func *translate_given_descrs,
        translate_loop_descrs_func *translate_loop_descrs);


typedef struct {
    PyArray_Descr base;
    double scaling;
} PyArray_SFloatDescr;

static PyArray_DTypeMeta PyArray_SFloatDType;
static PyArray_SFloatDescr SFloatSingleton;


static int
sfloat_is_known_scalar_type(PyArray_DTypeMeta *NPY_UNUSED(cls), PyTypeObject *type)
{
    /* Accept only floats (some others may work due to normal casting) */
    if (type == &PyFloat_Type) {
        return 1;
    }
    return 0;
}


static PyArray_Descr *
sfloat_default_descr(PyArray_DTypeMeta *NPY_UNUSED(cls))
{
    Py_INCREF(&SFloatSingleton);
    return (PyArray_Descr *)&SFloatSingleton;
}


static PyArray_Descr *
sfloat_discover_from_pyobject(PyArray_DTypeMeta *cls, PyObject *NPY_UNUSED(obj))
{
    return sfloat_default_descr(cls);
}


static PyArray_DTypeMeta *
sfloat_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (NPY_DT_is_legacy(other) && other->type_num == NPY_DOUBLE) {
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_Descr *
sfloat_common_instance(PyArray_Descr *descr1, PyArray_Descr *descr2)
{
    PyArray_SFloatDescr *sf1 = (PyArray_SFloatDescr *)descr1;
    PyArray_SFloatDescr *sf2 = (PyArray_SFloatDescr *)descr2;
    /* We make the choice of using the larger scaling */
    if (sf1->scaling >= sf2->scaling) {
        Py_INCREF(descr1);
        return descr1;
    }
    Py_INCREF(descr2);
    return descr2;
}


/*
 * Implement minimal getitem and setitem to make this DType mostly(?) safe to
 * expose in Python.
 * TODO: This should not use the old-style API, but the new-style is missing!
*/

static PyObject *
sfloat_getitem(char *data, PyArrayObject *arr)
{
    PyArray_SFloatDescr *descr = (PyArray_SFloatDescr *)PyArray_DESCR(arr);
    double value;

    memcpy(&value, data, sizeof(double));
    return PyFloat_FromDouble(value * descr->scaling);
}


static int
sfloat_setitem(PyObject *obj, char *data, PyArrayObject *arr)
{
    if (!PyFloat_CheckExact(obj)) {
        PyErr_SetString(PyExc_NotImplementedError,
                "Currently only accepts floats");
        return -1;
    }

    PyArray_SFloatDescr *descr = (PyArray_SFloatDescr *)PyArray_DESCR(arr);
    double value = PyFloat_AsDouble(obj);
    value /= descr->scaling;

    memcpy(data, &value, sizeof(double));
    return 0;
}


/* Special DType methods and the descr->f slot storage */
NPY_DType_Slots sfloat_slots = {
    .discover_descr_from_pyobject = &sfloat_discover_from_pyobject,
    .is_known_scalar_type = &sfloat_is_known_scalar_type,
    .default_descr = &sfloat_default_descr,
    .common_dtype = &sfloat_common_dtype,
    .common_instance = &sfloat_common_instance,
    .f = {
        .getitem = (PyArray_GetItemFunc *)&sfloat_getitem,
        .setitem = (PyArray_SetItemFunc *)&sfloat_setitem,
    }
};

static PyArray_SFloatDescr SFloatSingleton = {{
        .byteorder = '|',  /* do not bother with byte-swapping... */
        .flags = NPY_USE_GETITEM|NPY_USE_SETITEM,
        .type_num = -1,
        .elsize = sizeof(double),
        .alignment = NPY_ALIGNOF(double),
        .f = &sfloat_slots.f,
    },
    .scaling = 1,
};


static PyArray_Descr *
sfloat_scaled_copy(PyArray_SFloatDescr *self, double factor) {
    PyArray_SFloatDescr *new = PyObject_New(
            PyArray_SFloatDescr, (PyTypeObject *)&PyArray_SFloatDType);
    if (new == NULL) {
        return NULL;
    }
    /* Don't copy PyObject_HEAD part */
    memcpy((char *)new + sizeof(PyObject),
            (char *)self + sizeof(PyObject),
            sizeof(PyArray_SFloatDescr) - sizeof(PyObject));

    new->scaling = new->scaling * factor;
    return (PyArray_Descr *)new;
}


PyObject *
python_sfloat_scaled_copy(PyArray_SFloatDescr *self, PyObject *arg)
{
    if (!PyFloat_Check(arg)) {
        PyErr_SetString(PyExc_TypeError,
                "Scaling factor must be a python float.");
        return NULL;
    }
    double factor = PyFloat_AsDouble(arg);

    return (PyObject *)sfloat_scaled_copy(self, factor);
}


static PyObject *
sfloat_get_scaling(PyArray_SFloatDescr *self, PyObject *NPY_UNUSED(args))
{
    return PyFloat_FromDouble(self->scaling);
}


PyMethodDef sfloat_methods[] = {
    {"scaled_by",
         (PyCFunction)python_sfloat_scaled_copy, METH_O,
        "Method to get a dtype copy with different scaling, mainly to "
        "avoid having to implement many ways to create new instances."},
    {"get_scaling",
        (PyCFunction)sfloat_get_scaling, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};


static PyObject *
sfloat_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    double scaling = 1.;
    static char *kwargs_strs[] = {"scaling", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|d:_ScaledFloatTestDType", kwargs_strs, &scaling)) {
        return NULL;
    }
    if (scaling == 1.) {
        Py_INCREF(&SFloatSingleton);
        return (PyObject *)&SFloatSingleton;
    }
    return (PyObject *)sfloat_scaled_copy(&SFloatSingleton, scaling);
}


static PyObject *
sfloat_repr(PyArray_SFloatDescr *self)
{
    PyObject *scaling = PyFloat_FromDouble(self->scaling);
    if (scaling == NULL) {
        return NULL;
    }
    PyObject *res = PyUnicode_FromFormat(
            "_ScaledFloatTestDType(scaling=%R)", scaling);
    Py_DECREF(scaling);
    return res;
}


static PyArray_DTypeMeta PyArray_SFloatDType = {{{
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "numpy._ScaledFloatTestDType",
        .tp_basicsize = sizeof(PyArray_SFloatDescr),
        .tp_repr = (reprfunc)sfloat_repr,
        .tp_str = (reprfunc)sfloat_repr,
        .tp_methods = sfloat_methods,
        .tp_new = sfloat_new,
    }},
    .type_num = -1,
    .scalar_type = NULL,
    .flags = NPY_DT_PARAMETRIC | NPY_DT_NUMERIC,
    .dt_slots = &sfloat_slots,
};


/*
 * Implement some casts.
 */

/*
 * It would make more sense to test this early on, but this allows testing
 * error returns.
 */
static int
check_factor(double factor) {
    if (npy_isfinite(factor) && factor != 0.) {
        return 0;
    }
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    PyErr_SetString(PyExc_TypeError,
            "error raised inside the core-loop: non-finite factor!");
    NPY_DISABLE_C_API;
    return -1;
}


static int
cast_sfloat_to_sfloat_unaligned(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    /* could also be moved into auxdata: */
    double factor = ((PyArray_SFloatDescr *)context->descriptors[0])->scaling;
    factor /= ((PyArray_SFloatDescr *)context->descriptors[1])->scaling;
    if (check_factor(factor) < 0) {
        return -1;
    }

    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    for (npy_intp i = 0; i < N; i++) {
        double tmp;
        memcpy(&tmp, in, sizeof(double));
        tmp *= factor;
        memcpy(out, &tmp, sizeof(double));

        in += strides[0];
        out += strides[1];
    }
    return 0;
}


static int
cast_sfloat_to_sfloat_aligned(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    /* could also be moved into auxdata: */
    double factor = ((PyArray_SFloatDescr *)context->descriptors[0])->scaling;
    factor /= ((PyArray_SFloatDescr *)context->descriptors[1])->scaling;
    if (check_factor(factor) < 0) {
        return -1;
    }

    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    for (npy_intp i = 0; i < N; i++) {
        *(double *)out = *(double *)in * factor;
        in += strides[0];
        out += strides[1];
    }
    return 0;
}


static NPY_CASTING
sfloat_to_sfloat_resolve_descriptors(
            PyArrayMethodObject *NPY_UNUSED(self),
            PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
            PyArray_Descr *given_descrs[2],
            PyArray_Descr *loop_descrs[2],
            npy_intp *view_offset)
{
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(loop_descrs[0]);

    if (given_descrs[1] == NULL) {
        loop_descrs[1] = given_descrs[0];
    }
    else {
        loop_descrs[1] = given_descrs[1];
    }
    Py_INCREF(loop_descrs[1]);

    if (((PyArray_SFloatDescr *)loop_descrs[0])->scaling
            == ((PyArray_SFloatDescr *)loop_descrs[1])->scaling) {
        /* same scaling is just a view */
        *view_offset = 0;
        return NPY_NO_CASTING;
    }
    else if (-((PyArray_SFloatDescr *)loop_descrs[0])->scaling
             == ((PyArray_SFloatDescr *)loop_descrs[1])->scaling) {
        /* changing the sign does not lose precision */
        return NPY_EQUIV_CASTING;
    }
    /* Technically, this is not a safe cast, since over/underflows can occur */
    return NPY_SAME_KIND_CASTING;
}


/*
 * Casting to and from doubles.
 *
 * To keep things interesting, we ONLY define the trivial cast with a factor
 * of 1.  All other casts have to be handled by the sfloat to sfloat cast.
 *
 * The casting machinery should optimize this step away normally, since we
 * flag the this is a view.
 */
static int
cast_float_to_from_sfloat(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    for (npy_intp i = 0; i < N; i++) {
        *(double *)out = *(double *)in;
        in += strides[0];
        out += strides[1];
    }
    return 0;
}


static NPY_CASTING
float_to_from_sfloat_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *dtypes[2],
        PyArray_Descr *NPY_UNUSED(given_descrs[2]),
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    loop_descrs[0] = NPY_DT_CALL_default_descr(dtypes[0]);
    if (loop_descrs[0] == NULL) {
        return -1;
    }
    loop_descrs[1] = NPY_DT_CALL_default_descr(dtypes[1]);
    if (loop_descrs[1] == NULL) {
        return -1;
    }
    *view_offset = 0;
    return NPY_NO_CASTING;
}


/*
 * Cast to boolean (for testing the logical functions a bit better).
 */
static int
cast_sfloat_to_bool(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    for (npy_intp i = 0; i < N; i++) {
        *(npy_bool *)out = *(double *)in != 0;
        in += strides[0];
        out += strides[1];
    }
    return 0;
}

static NPY_CASTING
sfloat_to_bool_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    if (loop_descrs[0] == NULL) {
        return -1;
    }
    loop_descrs[1] = PyArray_DescrFromType(NPY_BOOL);  /* cannot fail */
    return NPY_UNSAFE_CASTING;
}


static int
sfloat_init_casts(void)
{
    PyArray_DTypeMeta *dtypes[2] = {&PyArray_SFloatDType, &PyArray_SFloatDType};
    PyType_Slot slots[4] = {{0, NULL}};
    PyArrayMethod_Spec spec = {
        .name = "sfloat_to_sfloat_cast",
        .nin = 1,
        .nout = 1,
        /* minimal guaranteed casting */
        .casting = NPY_SAME_KIND_CASTING,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,
        .dtypes = dtypes,
        .slots = slots,
    };

    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &sfloat_to_sfloat_resolve_descriptors;

    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &cast_sfloat_to_sfloat_aligned;

    slots[2].slot = NPY_METH_unaligned_strided_loop;
    slots[2].pfunc = &cast_sfloat_to_sfloat_unaligned;

    if (PyArray_AddCastingImplementation_FromSpec(&spec, 0)) {
        return -1;
    }

    spec.name = "float_to_sfloat_cast";
    /* Technically, it is just a copy currently so this is fine: */
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    PyArray_DTypeMeta *double_DType = PyArray_DTypeFromTypeNum(NPY_DOUBLE);
    Py_DECREF(double_DType);  /* immortal anyway */
    dtypes[0] = double_DType;

    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &float_to_from_sfloat_resolve_descriptors;
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &cast_float_to_from_sfloat;
    slots[2].slot = 0;
    slots[2].pfunc = NULL;

    if (PyArray_AddCastingImplementation_FromSpec(&spec, 0)) {
        return -1;
    }

    spec.name = "sfloat_to_float_cast";
    dtypes[0] = &PyArray_SFloatDType;
    dtypes[1] = double_DType;

    if (PyArray_AddCastingImplementation_FromSpec(&spec, 0)) {
        return -1;
    }

    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &sfloat_to_bool_resolve_descriptors;
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &cast_sfloat_to_bool;
    slots[2].slot = 0;
    slots[2].pfunc = NULL;

    spec.name = "sfloat_to_bool_cast";
    dtypes[0] = &PyArray_SFloatDType;
    dtypes[1] = PyArray_DTypeFromTypeNum(NPY_BOOL);
    Py_DECREF(dtypes[1]);  /* immortal anyway */

    if (PyArray_AddCastingImplementation_FromSpec(&spec, 0)) {
        return -1;
    }

    return 0;
}


/*
 * We also wish to test very simple ufunc functionality.  So create two
 * ufunc loops:
 * 1. Multiplication, which can multiply the factors and work with that.
 * 2. Addition, which needs to use the common instance, and runs into
 *    cast safety subtleties since we will implement it without an additional
 *    cast.
 */
static int
multiply_sfloats(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    for (npy_intp i = 0; i < N; i++) {
        *(double *)out = *(double *)in1 * *(double *)in2;
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }
    return 0;
}


static NPY_CASTING
multiply_sfloats_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[3]),
        PyArray_Descr *given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    /*
     * Multiply the scaling for the result.  If the result was passed in we
     * simply ignore it and let the casting machinery fix it up here.
     */
    double factor = ((PyArray_SFloatDescr *)given_descrs[1])->scaling;
    loop_descrs[2] = sfloat_scaled_copy(
            (PyArray_SFloatDescr *)given_descrs[0], factor);
    if (loop_descrs[2] == 0) {
        return -1;
    }
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    return NPY_NO_CASTING;
}


/*
 * Unlike the multiplication implementation above, this loops deals with
 * scaling (casting) internally.  This allows to test some different paths.
 */
static int
add_sfloats(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    double fin1 = ((PyArray_SFloatDescr *)context->descriptors[0])->scaling;
    double fin2 = ((PyArray_SFloatDescr *)context->descriptors[1])->scaling;
    double fout = ((PyArray_SFloatDescr *)context->descriptors[2])->scaling;

    double fact1 = fin1 / fout;
    double fact2 = fin2 / fout;
    if (check_factor(fact1) < 0) {
        return -1;
    }
    if (check_factor(fact2) < 0) {
        return -1;
    }

    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    for (npy_intp i = 0; i < N; i++) {
        *(double *)out = (*(double *)in1 * fact1) + (*(double *)in2 * fact2);
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }
    return 0;
}


static NPY_CASTING
add_sfloats_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[3]),
        PyArray_Descr *given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    /*
     * Here we accept an output descriptor (the inner loop can deal with it),
     * if none is given, we use the "common instance":
     */
    if (given_descrs[2] == NULL) {
        loop_descrs[2] = sfloat_common_instance(
                given_descrs[0], given_descrs[1]);
        if (loop_descrs[2] == 0) {
            return -1;
        }
    }
    else {
        Py_INCREF(given_descrs[2]);
        loop_descrs[2] = given_descrs[2];
    }
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    /* If the factors mismatch, we do implicit casting inside the ufunc! */
    double fin1 = ((PyArray_SFloatDescr *)loop_descrs[0])->scaling;
    double fin2 = ((PyArray_SFloatDescr *)loop_descrs[1])->scaling;
    double fout = ((PyArray_SFloatDescr *)loop_descrs[2])->scaling;

    if (fin1 == fout && fin2 == fout) {
        return NPY_NO_CASTING;
    }
    if (npy_fabs(fin1) == npy_fabs(fout) && npy_fabs(fin2) == npy_fabs(fout)) {
        return NPY_EQUIV_CASTING;
    }
    return NPY_SAME_KIND_CASTING;
}


/*
 * We define the hypot loop using the "PyUFunc_AddWrappingLoop" API.
 * We use this very narrowly for mapping to the double hypot loop currently.
 */
static int
translate_given_descrs_to_double(
        int nin, int nout, PyArray_DTypeMeta *wrapped_dtypes[],
        PyArray_Descr *given_descrs[], PyArray_Descr *new_descrs[])
{
    assert(nin == 2 && nout == 1);
    for (int i = 0; i < 3; i++) {
        if (given_descrs[i] == NULL) {
            new_descrs[i] = NULL;
        }
        else {
            new_descrs[i] = PyArray_DescrFromType(NPY_DOUBLE);
        }
    }
    return 0;
}


static int
translate_loop_descrs(
        int nin, int nout, PyArray_DTypeMeta *new_dtypes[],
        PyArray_Descr *given_descrs[],
        PyArray_Descr *NPY_UNUSED(original_descrs[]),
        PyArray_Descr *loop_descrs[])
{
    assert(nin == 2 && nout == 1);
    loop_descrs[0] = sfloat_common_instance(
            given_descrs[0], given_descrs[1]);
    if (loop_descrs[0] == 0) {
        return -1;
    }
    Py_INCREF(loop_descrs[0]);
    loop_descrs[1] = loop_descrs[0];
    Py_INCREF(loop_descrs[0]);
    loop_descrs[2] = loop_descrs[0];
    return 0;
}


static PyObject *
sfloat_get_ufunc(const char *ufunc_name)
{
    PyObject *mod = PyImport_ImportModule("numpy");
    if (mod == NULL) {
        return NULL;
    }
    PyObject *ufunc = PyObject_GetAttrString(mod, ufunc_name);
    Py_DECREF(mod);
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type)) {
        Py_DECREF(ufunc);
        PyErr_Format(PyExc_TypeError,
                "numpy.%s was not a ufunc!", ufunc_name);
        return NULL;
    }
    return ufunc;
}


static int
sfloat_add_loop(const char *ufunc_name,
        PyArray_DTypeMeta *dtypes[3], PyObject *meth_or_promoter)
{
    PyObject *ufunc = sfloat_get_ufunc(ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }
    PyObject *dtype_tup = PyArray_TupleFromItems(3, (PyObject **)dtypes, 1);
    if (dtype_tup == NULL) {
        Py_DECREF(ufunc);
        return -1;
    }
    PyObject *info = PyTuple_Pack(2, dtype_tup, meth_or_promoter);
    Py_DECREF(dtype_tup);
    if (info == NULL) {
        Py_DECREF(ufunc);
        return -1;
    }
    int res = PyUFunc_AddLoop((PyUFuncObject *)ufunc, info, 0);
    Py_DECREF(ufunc);
    Py_DECREF(info);
    return res;
}


static int
sfloat_add_wrapping_loop(const char *ufunc_name, PyArray_DTypeMeta *dtypes[3])
{
    PyObject *ufunc = sfloat_get_ufunc(ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }
    PyArray_DTypeMeta *double_dt = PyArray_DTypeFromTypeNum(NPY_DOUBLE);
    PyArray_DTypeMeta *wrapped_dtypes[3] = {double_dt, double_dt, double_dt};
    int res = PyUFunc_AddWrappingLoop(
        ufunc, dtypes, wrapped_dtypes, &translate_given_descrs_to_double,
        &translate_loop_descrs);
    Py_DECREF(ufunc);
    Py_DECREF(double_dt);

    return res;
}


/*
 * We add some very basic promoters to allow multiplying normal and scaled
 */
static int
promote_to_sfloat(PyUFuncObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_DTypeMeta *const signature[3],
        PyArray_DTypeMeta *new_dtypes[3])
{
    for (int i = 0; i < 3; i++) {
        PyArray_DTypeMeta *new = &PyArray_SFloatDType;
        if (signature[i] != NULL) {
            new = signature[i];
        }
        Py_INCREF(new);
        new_dtypes[i] = new;
    }
    return 0;
}


/*
 * Add new ufunc loops (this is somewhat clumsy as of writing it, but should
 * get less so with the introduction of public API).
 */
static int
sfloat_init_ufuncs(void) {
    PyArray_DTypeMeta *dtypes[3] = {
            &PyArray_SFloatDType, &PyArray_SFloatDType, &PyArray_SFloatDType};
    PyType_Slot slots[3] = {{0, NULL}};
    PyArrayMethod_Spec spec = {
        .nin = 2,
        .nout =1,
        .dtypes = dtypes,
        .slots = slots,
    };
    spec.name = "sfloat_multiply";
    spec.casting = NPY_NO_CASTING;

    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &multiply_sfloats_resolve_descriptors;
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &multiply_sfloats;
    PyBoundArrayMethodObject *bmeth = PyArrayMethod_FromSpec_int(&spec, 0);
    if (bmeth == NULL) {
        return -1;
    }
    int res = sfloat_add_loop("multiply",
            bmeth->dtypes, (PyObject *)bmeth->method);
    Py_DECREF(bmeth);
    if (res < 0) {
        return -1;
    }

    spec.name = "sfloat_add";
    spec.casting = NPY_SAME_KIND_CASTING;

    slots[0].slot = NPY_METH_resolve_descriptors;
    slots[0].pfunc = &add_sfloats_resolve_descriptors;
    slots[1].slot = NPY_METH_strided_loop;
    slots[1].pfunc = &add_sfloats;
    bmeth = PyArrayMethod_FromSpec_int(&spec, 0);
    if (bmeth == NULL) {
        return -1;
    }
    res = sfloat_add_loop("add",
            bmeth->dtypes, (PyObject *)bmeth->method);
    Py_DECREF(bmeth);
    if (res < 0) {
        return -1;
    }

    /* N.B.: Wrapping isn't actually correct if scaling can be negative */
    if (sfloat_add_wrapping_loop("hypot", dtypes) < 0) {
        return -1;
    }

    /*
     * Add a promoter for both directions of multiply with double.
     */
    PyArray_DTypeMeta *double_DType = PyArray_DTypeFromTypeNum(NPY_DOUBLE);
    Py_DECREF(double_DType);  /* immortal anyway */

    PyArray_DTypeMeta *promoter_dtypes[3] = {
            &PyArray_SFloatDType, double_DType, NULL};

    PyObject *promoter = PyCapsule_New(
            &promote_to_sfloat, "numpy._ufunc_promoter", NULL);
    if (promoter == NULL) {
        return -1;
    }
    res = sfloat_add_loop("multiply", promoter_dtypes, promoter);
    if (res < 0) {
        Py_DECREF(promoter);
        return -1;
    }
    promoter_dtypes[0] = double_DType;
    promoter_dtypes[1] = &PyArray_SFloatDType;
    res = sfloat_add_loop("multiply", promoter_dtypes, promoter);
    Py_DECREF(promoter);
    if (res < 0) {
        return -1;
    }

    return 0;
}


/*
 * Python entry point, exported via `umathmodule.h` and `multiarraymodule.c`.
 * TODO: Should be moved when the necessary API is not internal anymore.
 */
NPY_NO_EXPORT PyObject *
get_sfloat_dtype(PyObject *NPY_UNUSED(mod), PyObject *NPY_UNUSED(args))
{
    /* Allow calling the function multiple times. */
    static npy_bool initialized = NPY_FALSE;

    if (initialized) {
        Py_INCREF(&PyArray_SFloatDType);
        return (PyObject *)&PyArray_SFloatDType;
    }

    PyArray_SFloatDType.super.ht_type.tp_base = &PyArrayDescr_Type;

    if (PyType_Ready((PyTypeObject *)&PyArray_SFloatDType) < 0) {
        return NULL;
    }
    NPY_DT_SLOTS(&PyArray_SFloatDType)->castingimpls = PyDict_New();
    if (NPY_DT_SLOTS(&PyArray_SFloatDType)->castingimpls == NULL) {
        return NULL;
    }

    PyObject *o = PyObject_Init(
            (PyObject *)&SFloatSingleton, (PyTypeObject *)&PyArray_SFloatDType);
    if (o == NULL) {
        return NULL;
    }

    if (sfloat_init_casts() < 0) {
        return NULL;
    }

    if (sfloat_init_ufuncs() < 0) {
        return NULL;
    }

    initialized = NPY_TRUE;
    return (PyObject *)&PyArray_SFloatDType;
}
