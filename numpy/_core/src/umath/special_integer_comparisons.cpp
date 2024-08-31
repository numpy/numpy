#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"

#include "abstractdtypes.h"
#include "dispatching.h"
#include "dtypemeta.h"
#include "convert_datatype.h"

#include "legacy_array_method.h"  /* For `get_wrapped_legacy_ufunc_loop`. */
#include "special_integer_comparisons.h"


/*
 * Helper for templating, avoids warnings about uncovered switch paths.
 */
enum class COMP {
    EQ, NE, LT, LE, GT, GE,
};

static char const *
comp_name(COMP comp) {
    switch(comp) {
        case COMP::EQ: return "equal";
        case COMP::NE: return "not_equal";
        case COMP::LT: return "less";
        case COMP::LE: return "less_equal";
        case COMP::GT: return "greater";
        case COMP::GE: return "greater_equal";
        default:
            assert(0);
            return nullptr;
    }
}


template <bool result>
static int
fixed_result_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *out = data[2];
    npy_intp stride = strides[2];

    while (N--) {
        *reinterpret_cast<npy_bool *>(out) = result;
        out += stride;
    }
    return 0;
}

static inline void
get_min_max(int typenum, long long *min, unsigned long long *max)
{
    *min = 0;
    switch (typenum) {
        case NPY_BYTE:
            *min = NPY_MIN_BYTE;
            *max = NPY_MAX_BYTE;
            break;
        case NPY_UBYTE:
            *max = NPY_MAX_UBYTE;
            break;
        case NPY_SHORT:
            *min = NPY_MIN_SHORT;
            *max = NPY_MAX_SHORT;
            break;
        case NPY_USHORT:
            *max = NPY_MAX_USHORT;
            break;
        case NPY_INT:
            *min = NPY_MIN_INT;
            *max = NPY_MAX_INT;
            break;
        case NPY_UINT:
            *max = NPY_MAX_UINT;
            break;
        case NPY_LONG:
            *min = NPY_MIN_LONG;
            *max = NPY_MAX_LONG;
            break;
        case NPY_ULONG:
            *max = NPY_MAX_ULONG;
            break;
        case NPY_LONGLONG:
            *min = NPY_MIN_LONGLONG;
            *max = NPY_MAX_LONGLONG;
            break;
        case NPY_ULONGLONG:
            *max = NPY_MAX_ULONGLONG;
            break;
        default:
            *max = 0;
            assert(0);
    }
}


/*
 * Determine if a Python long is within the typenums range, smaller, or larger.
 * 
 * Function returns -1 for errors.
 */
static inline int
get_value_range(PyObject *value, int type_num, int *range)
{
    long long min;
    unsigned long long max;
    get_min_max(type_num, &min, &max);

    int overflow;
    long long val = PyLong_AsLongLongAndOverflow(value, &overflow);
    if (val == -1 && overflow == 0 && PyErr_Occurred()) {
        return -1;
    }

    if (overflow == 0) {
        if (val < min) {
            *range = -1;
        }
        else if (val > 0 && (unsigned long long)val > max) {
            *range = 1;
        }
        else {
            *range = 0;
        }
    }
    else if (overflow < 0) {
        *range = -1;
    }
    else if (max <= NPY_MAX_LONGLONG) {
        *range = 1;
    }
    else {
        /*
        * If we are checking for unisgned long long, the value may be larger
        * then long long, but within range of unsigned long long.  Check this
        * by doing the normal Python integer comparison.
        */
        PyObject *obj = PyLong_FromUnsignedLongLong(max);
        if (obj == NULL) {
            return -1;
        }
        int cmp = PyObject_RichCompareBool(value, obj, Py_GT);
        Py_DECREF(obj);
        if (cmp < 0) {
            return -1;
        }
        if (cmp) {
            *range = 1;
        }
        else {
            *range = 0;
        }
    }
    return 0;
}


/*
 * Find the type resolution for any numpy_int with pyint comparison.  This
 * function supports *both* directions for all types.
 */
static NPY_CASTING
resolve_descriptors_with_scalars(
    PyArrayMethodObject *self, PyArray_DTypeMeta **dtypes,
    PyArray_Descr **given_descrs, PyObject *const *input_scalars,
    PyArray_Descr **loop_descrs, npy_intp *view_offset)
{
    int value_range = 0;

    npy_bool first_is_pyint = dtypes[0] == &PyArray_PyLongDType;
    int arr_idx = first_is_pyint? 1 : 0;
    int scalar_idx = first_is_pyint? 0 : 1;
    PyObject *scalar = input_scalars[scalar_idx];
    assert(PyTypeNum_ISINTEGER(dtypes[arr_idx]->type_num));
    PyArray_DTypeMeta *arr_dtype = dtypes[arr_idx];

    /*
     * Three way decision (with hack) on value range:
     *  0: The value fits within the range of the dtype.
     *  1: The value came second and is larger or came first and is smaller.
     * -1: The value came second and is smaller or came first and is larger
     */
    if (scalar != NULL && PyLong_CheckExact(scalar)) {
        if (get_value_range(scalar, arr_dtype->type_num, &value_range) < 0) {
            return _NPY_ERROR_OCCURRED_IN_CAST;
        }
        if (first_is_pyint == 1) {
            value_range *= -1;
        }
    }

    /*
     * Very small/large values always need to be encoded as `object` dtype
     * in order to never fail casting (NumPy will store the Python integer
     * in a 0-D object array this way -- even if we never inspect it).
     *
     * TRICK: We encode the value range by whether or not we use the object
     *        singleton!  This information is then available in `get_loop()`
     *        to pick a loop that returns always True or False.
     */
    if (value_range == 0) {
        Py_INCREF(arr_dtype->singleton);
        loop_descrs[scalar_idx] = arr_dtype->singleton;
    }
    else if (value_range < 0) {
        loop_descrs[scalar_idx] = PyArray_DescrFromType(NPY_OBJECT);
    }
    else {
        loop_descrs[scalar_idx] = PyArray_DescrNewFromType(NPY_OBJECT);
        if (loop_descrs[scalar_idx] == NULL) {
            return _NPY_ERROR_OCCURRED_IN_CAST;
        }
    }
    Py_INCREF(arr_dtype->singleton);
    loop_descrs[arr_idx] = arr_dtype->singleton;
    loop_descrs[2] = PyArray_DescrFromType(NPY_BOOL);

    return NPY_NO_CASTING;
}


template<COMP comp>
static int
get_loop(PyArrayMethod_Context *context,
        int aligned, int move_references, const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    if (context->descriptors[1]->type_num == context->descriptors[0]->type_num) {
        /*
         * Fall back to the current implementation, which wraps legacy loops.
         */
        return get_wrapped_legacy_ufunc_loop(
                context, aligned, move_references, strides,
                out_loop, out_transferdata, flags);
    }
    else {
        PyArray_Descr *other_descr;
        if (context->descriptors[1]->type_num == NPY_OBJECT) {
            other_descr = context->descriptors[1];
        }
        else {
            assert(context->descriptors[0]->type_num == NPY_OBJECT);
            other_descr = context->descriptors[0];
        }
        /* HACK: If the descr is the singleton the result is smaller */
        PyArray_Descr *obj_singleton = PyArray_DescrFromType(NPY_OBJECT);
        if (other_descr == obj_singleton) {
            /* Singleton came second and is smaller, or first and is larger */
            switch (comp) {
                case COMP::EQ:
                case COMP::LT:
                case COMP::LE:
                    *out_loop = &fixed_result_loop<false>;
                    break;
                case COMP::NE:
                case COMP::GT:
                case COMP::GE:
                    *out_loop = &fixed_result_loop<true>;
                    break;
            }
        }
        else {
            /* Singleton came second and is larger, or first and is smaller */
            switch (comp) {
                case COMP::EQ:
                case COMP::GT:
                case COMP::GE:
                    *out_loop = &fixed_result_loop<false>;
                    break;
                case COMP::NE:
                case COMP::LT:
                case COMP::LE:
                    *out_loop = &fixed_result_loop<true>;
                    break;
            }
        }
        Py_DECREF(obj_singleton);
    }
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    return 0;
}


/*
 * Machinery to add the python integer to NumPy integer comparsisons as well
 * as a special promotion to special case Python int with Python int
 * comparisons.
 */

/*
 * Simple promoter that ensures we use the object loop when the input
 * is python integers only.
 * Note that if a user would pass the Python `int` abstract DType explicitly
 * they promise to actually pass a Python int and we accept that we never
 * check for that.
 */
static int
pyint_comparison_promoter(PyUFuncObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    new_op_dtypes[0] = NPY_DT_NewRef(&PyArray_ObjectDType);
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_ObjectDType);
    new_op_dtypes[2] = NPY_DT_NewRef(&PyArray_BoolDType);
    return 0;
}


/*
 * This function replaces the strided loop with the passed in one,
 * and registers it with the given ufunc.
 * It additionally adds a promoter for (pyint, pyint, bool) to use the
 * (object, object, bool) implementation.
 */
template<COMP comp>
static int
add_dtype_loops(PyObject *umath, PyArrayMethod_Spec *spec, PyObject *info)
{
    PyArray_DTypeMeta *PyInt = &PyArray_PyLongDType;

    PyObject *name = PyUnicode_FromString(comp_name(comp));
    if (name == nullptr) {
        return -1;
    }
    PyUFuncObject *ufunc = (PyUFuncObject *)PyObject_GetItem(umath, name);
    Py_DECREF(name);
    if (ufunc == nullptr) {
        return -1;
    }
    if (Py_TYPE(ufunc) != &PyUFunc_Type) {
        PyErr_SetString(PyExc_RuntimeError,
                "internal NumPy error: comparison not a ufunc");
        goto fail;
    }

    /* 
     * NOTE: Iterates all type numbers, it would be nice to reduce this.
     *       (that would be easier if we consolidate int DTypes in general.)
     */
    for (int typenum = NPY_BYTE; typenum <= NPY_ULONGLONG; typenum++) {
        spec->slots[0].pfunc = (void *)get_loop<comp>;

        PyArray_DTypeMeta *Int = PyArray_DTypeFromTypeNum(typenum);

        /* Register the spec/loop for both forward and backward direction */
        spec->dtypes[0] = Int;
        spec->dtypes[1] = PyInt;
        int res = PyUFunc_AddLoopFromSpec_int((PyObject *)ufunc, spec, 1);
        if (res < 0) {
            Py_DECREF(Int);
            goto fail;
        }
        spec->dtypes[0] = PyInt;
        spec->dtypes[1] = Int;
        res = PyUFunc_AddLoopFromSpec_int((PyObject *)ufunc, spec, 1);
        Py_DECREF(Int);
        if (res < 0) {
            goto fail;
        }
    }

    /*
     * Install the promoter info to allow two Python integers to work.
     */
    return PyUFunc_AddLoop((PyUFuncObject *)ufunc, info, 0);

    Py_DECREF(ufunc);
    return 0;

  fail:
    Py_DECREF(ufunc);
    return -1;
}


template<COMP...>
struct add_loops;

template<>
struct add_loops<> {
    int operator()(PyObject*, PyArrayMethod_Spec*, PyObject *) {
        return 0;
    }
};


template<COMP comp, COMP... comps>
struct add_loops<comp, comps...> {
    int operator()(PyObject* umath, PyArrayMethod_Spec* spec, PyObject *info) {
        if (add_dtype_loops<comp>(umath, spec, info) < 0) {
            return -1;
        }
        else {
            return add_loops<comps...>()(umath, spec, info);
        }
    }
};


NPY_NO_EXPORT int
init_special_int_comparisons(PyObject *umath)
{
    int res = -1;
    PyObject *info = NULL, *promoter = NULL;
    PyArray_DTypeMeta *Bool = &PyArray_BoolDType;

    /* All loops have a boolean out DType (others filled in later) */
    PyArray_DTypeMeta *dtypes[] = {NULL, NULL, Bool};
    /*
     * We only have one loop right now, the strided one.  The default type
     * resolver ensures native byte order/canonical representation.
     */
    PyType_Slot slots[] = {
        {NPY_METH_get_loop, nullptr},
        {_NPY_METH_resolve_descriptors_with_scalars,
             (void *)&resolve_descriptors_with_scalars},
        {0, NULL},
    };

    PyArrayMethod_Spec spec = {};
    spec.name = "templated_pyint_to_integers_comparisons";
    spec.nin = 2;
    spec.nout = 1;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    /*
     * The following sets up the correct promoter to make comparisons like
     * `np.equal(2, 4)` (with two python integers) use an object loop.
     */
    PyObject *dtype_tuple = PyTuple_Pack(3,
            &PyArray_PyLongDType, &PyArray_PyLongDType, Bool);
    if (dtype_tuple == NULL) {
        goto finish;
    }
    promoter = PyCapsule_New(
            (void *)&pyint_comparison_promoter, "numpy._ufunc_promoter", NULL);
    if (promoter == NULL) {
        Py_DECREF(dtype_tuple);
        goto finish;
    }
    info = PyTuple_Pack(2, dtype_tuple, promoter);
    Py_DECREF(dtype_tuple);
    Py_DECREF(promoter);
    if (info == NULL) {
        goto finish;
    }

    /* Add all combinations of PyInt and NumPy integer comparisons */
    using comp_looper = add_loops<COMP::EQ, COMP::NE, COMP::LT, COMP::LE, COMP::GT, COMP::GE>;
    if (comp_looper()(umath, &spec, info) < 0) {
        goto finish;
    }

    res = 0;
  finish:

    Py_XDECREF(info);
    return res;
}
