/*
 * This file implements an abstraction layer for "Array methods", which
 * work with a specific DType class input and provide low-level C function
 * pointers to do fast operations on the given input functions.
 * It thus adds an abstraction layer around individual ufunc loops.
 *
 * Unlike methods, a ArrayMethod can have multiple inputs and outputs.
 * This has some serious implication for garbage collection, and as far
 * as I (@seberg) understands, it is not possible to always guarantee correct
 * cyclic garbage collection of dynamically created DTypes with methods.
 * The keyword (or rather the solution) for this seems to be an "ephemeron"
 * which I believe should allow correct garbage collection but seems
 * not implemented in Python at this time.
 * The vast majority of use-cases will not require correct garbage collection.
 * Some use cases may require the user to be careful.
 *
 * Generally there are two main ways to solve this issue:
 *
 * 1. A method with a single input (or inputs of all the same DTypes) can
 *    be "owned" by that DType (it becomes unusable when the DType is deleted).
 *    This holds especially for all casts, which must have a defined output
 *    DType and must hold on to it strongly.
 * 2. A method which can infer the output DType(s) from the input types does
 *    not need to keep the output type alive. (It can use NULL for the type,
 *    or an abstract base class which is known to be persistent.)
 *    It is then sufficient for a ufunc (or other owner) to only hold a
 *    weak reference to the input DTypes.
 */


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <npy_pycompat.h>
#include "arrayobject.h"
#include "array_method.h"
#include "dtypemeta.h"
#include "convert_datatype.h"


/*
 * The default descriptor resolution function.  The logic is as follows:
 *
 * 1. The output is ensured to be canonical (currently native byte order),
 *    if it is of the correct DType.
 * 2. If any DType is was not defined, it is replaced by the common DType
 *    of all inputs. (If that common DType is parametric, this is an error.)
 *
 * We could allow setting the output descriptors specifically to simplify
 * this step.
 */
static NPY_CASTING
default_resolve_descriptors(
        PyArrayMethod_Context *context,
        PyArray_Descr **input_descrs,
        PyArray_Descr **output_descrs)
{
    int nin = context->nin;
    int nout = context->nout;
    int all_defined = 1;

    for (int i = 0; i < nin + nout; i++) {
        PyArray_DTypeMeta *dtype = context->dtypes[i];
        if (dtype == NULL) {
            output_descrs[i] = NULL;
            all_defined = 0;
            continue;
        }
        if (NPY_DTYPE(input_descrs[i]) == dtype) {
            output_descrs[i] = ensure_dtype_nbo(input_descrs[i]);
        }
        else {
            output_descrs[i] = dtype->default_descr(dtype);
        }
        if (NPY_UNLIKELY(output_descrs[i] == NULL)) {
            goto fail;
        }
    }
    if (all_defined) {
        return context->method->casting;
    }

    if (NPY_UNLIKELY(nin == 0 || context->dtypes[0] == NULL)) {
        /* Registration should reject this, so this would be indicates a bug */
        PyErr_SetString(PyExc_RuntimeError,
                "Invalid use of default resolver without inputs or with "
                "input or output DType incorrectly missing.");
        goto fail;
    }
    /* We find the common dtype of all inputs, and use it for the unknowns */
    PyArray_DTypeMeta *common_dtype = context->dtypes[0];
    assert(common_dtype != NULL);
    for (int i = 1; i < nin; i++) {
        Py_SETREF(common_dtype, PyArray_CommonDType(common_dtype, context->dtypes[i]));
        if (common_dtype == NULL) {
            goto fail;
        }
    }
    for (int i = nin; i < nin + nout; i++) {
        if (output_descrs[i] != NULL) {
            continue;
        }
        if (NPY_DTYPE(input_descrs[i]) == common_dtype) {
            output_descrs[i] = ensure_dtype_nbo(input_descrs[i]);
        }
        else {
            output_descrs[i] = common_dtype->default_descr(common_dtype);
        }
        if (NPY_UNLIKELY(output_descrs[i] == NULL)) {
            goto fail;
        }
    }

    return context->method->casting;

  fail:
    for (int i = 0; i < nin + nout; i++) {
        Py_XDECREF(output_descrs[i]);
    }
    return -1;
}


/**
 * The default method to fetch the correct loop for a cast or ufunc
 * (at the time of writing only casts).
 * The default version can return loops explicitly registered during method
 * creation. It does specialize contiguous loops, although has to check
 * all descriptors itemsizes for this.
 *
 * @param context
 * @param aligned
 * @param move_references UNUSED.
 * @param strides
 * @param descriptors
 * @param out_loop
 * @param out_transferdata
 * @param flags
 * @return 0 on success -1 on failure.
 */
static int
default_get_strided_loop(
        PyArrayMethod_Context *NPY_UNUSED(context),
        int NPY_UNUSED(aligned), int NPY_UNUSED(move_references),
        npy_intp *NPY_UNUSED(strides),
        PyArray_StridedUnaryOp **NPY_UNUSED(out_loop),
        NpyAuxData **NPY_UNUSED(out_transferdata),
        NPY_ARRAYMETHOD_FLAGS *NPY_UNUSED(flags))
{
    PyErr_SetString(PyExc_NotImplementedError,
            "default loop getter is not implemented");
    return -1;
}


/**
 * Validate that the input is usable to create a new ArrayMethod.
 *
 * @param spec
 * @return 0 on success -1 on error.
 */
static int
validate_spec(PyArrayMethod_Spec *spec)
{
    int nargs = spec->nin + spec->nout;
    /* Check the passed spec for invalid fields/values */
    if (spec->nin < 0 || spec->nout < 0 || nargs > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                "ArrayMethod inputs and outputs must be greater zero and"
                "not exceed %d. (method: %s)", NPY_MAXARGS, spec->name);
        return -1;
    }
    switch (spec->casting & ~_NPY_CAST_IS_VIEW) {
        case NPY_NO_CASTING:
        case NPY_EQUIV_CASTING:
        case NPY_SAFE_CASTING:
        case NPY_SAME_KIND_CASTING:
        case NPY_UNSAFE_CASTING:
            break;
        default:
            PyErr_Format(PyExc_TypeError,
                    "ArrayMethod has invalid casting `%d`. (method: %s)",
                    spec->casting, spec->name);
            return -1;
    }

    for (int i = 0; i < nargs; i++) {
        if (spec->dtypes[i] == NULL && i < spec->nin) {
            PyErr_Format(PyExc_TypeError,
                    "ArrayMethod must have well defined input DTypes. "
                    "(method: %s)", spec->name);
            return -1;
        }
        if (!PyObject_TypeCheck(spec->dtypes[i], &PyArrayDTypeMeta_Type)) {
            PyErr_Format(PyExc_TypeError,
                    "ArrayMethod provided object %R is not a DType."
                    "(method: %s)", spec->dtypes[i], spec->name);
            return -1;
        }
        if (spec->dtypes[i]->abstract && i < spec->nin) {
            PyErr_Format(PyExc_TypeError,
                    "abstract DType %S are currently not allowed for inputs."
                    "(method: %s defined at %s)", spec->dtypes[i], spec->name);
            return -1;
        }
    }
    return 0;
}


static int
fill_arraymethod_from_slots(
        PyBoundArrayMethodObject *res, PyArrayMethod_Spec *spec,
        int private)
{
    PyArrayMethodObject *meth = res->method;

    /* Set the defaults */
    meth->get_strided_loop = &default_get_strided_loop;
    meth->resolve_descriptors = &default_resolve_descriptors;

    /* Fill in the slots passed by the user */
    /*
     * TODO: This is reasonable for now, but it would be nice to find a
     *       shorter solution, and add some additional error checking (e.g.
     *       the same slot used twice). Python uses an array of slot offsets.
     */
    for (PyType_Slot *slot = &spec->slots[0]; slot->slot != 0; slot++) {
        switch (slot->slot) {
            case NPY_DTMETH_resolve_descriptors:
                meth->resolve_descriptors = slot->pfunc;
                continue;
            case NPY_DTMETH_get_loop:
                if (private) {
                    /* Only allow override for private functions initially */
                    meth->get_strided_loop = slot->pfunc;
                    continue;
                }
                break;
            case NPY_DTMETH_strided_loop:
                meth->strided_loop = slot->pfunc;
                continue;
            case NPY_DTMETH_contiguous_loop:
                meth->contiguous_loop = slot->pfunc;
                continue;
            case NPY_DTMETH_unaligned_strided_loop:
                meth->unaligned_strided_loop = slot->pfunc;
                continue;
            case NPY_DTMETH_unaligned_contiguous_loop:
                meth->unaligned_contiguous_loop = slot->pfunc;
                continue;
            default:
                break;
        }
        PyErr_Format(PyExc_RuntimeError,
                "invalid slot number %d to ArrayMethod: %s",
                slot->slot, spec->name);
        return -1;
    }

    /* Check whether the slots are valid: */
    if (meth->resolve_descriptors == &default_resolve_descriptors) {
        for (int i = 0; i < res->nin + res->nout; i++) {
            if (res->dtypes[i] == NULL) {
                if (i < res->nin) {
                    PyErr_Format(PyExc_TypeError,
                            "All input DTypes must be specified when using "
                            "the default `resolve_descriptors` function. "
                            "(method: %s)", spec->name);
                    return -1;
                }
                else if (res->nin == 0) {
                    PyErr_Format(PyExc_TypeError,
                            "Must specify output DTypes or use custom "
                            "`resolve_descriptors` when there are no inputs. "
                            "(method: %s defined at %s)", spec->name);
                    return -1;
                }
            }
            if (i >= res->nin && res->dtypes[i]->parametric) {
                PyErr_Format(PyExc_TypeError,
                        "must provide a `resolve_descriptors` function if any "
                        "output DType is parametric. (method: %s)",
                        spec->name);
                return -1;
            }
        }
    }
    if (meth->get_strided_loop != &default_get_strided_loop) {
        /* Do not check the actual loop fields. */
        return 0;
    }

    /* Check whether the provided loops make sense. */
    if (meth->strided_loop == NULL) {
        PyErr_Format(PyExc_TypeError,
                "Must provide a strided inner loop function. (method: %s)",
                spec->name);
        return -1;
    }
    if (meth->contiguous_loop == NULL) {
        meth->contiguous_loop = meth->strided_loop;
    }
    if (meth->unaligned_contiguous_loop != NULL &&
            meth->unaligned_strided_loop == NULL) {
        PyErr_Format(PyExc_TypeError,
                "Must provide unaligned strided inner loop when providing "
                "a contiguous version. (method: %s)", spec->name);
        return -1;
    }
    if ((meth->unaligned_strided_loop == NULL) !=
            !(meth->flags & NPY_METH_SUPPORTS_UNALIGNED)) {
        PyErr_Format(PyExc_TypeError,
                "Must provide unaligned strided inner loop when providing "
                "a contiguous version. (method: %s)", spec->name);
        return -1;
    }

    return 0;
}


/**
 * Create a new ArrayMethod (internal version).
 *
 * @param name A name for the individual method, may be NULL.
 * @param spec A filled context object to pass generic information about
 *        the method (such as usually needing the API, and the DTypes).
 *        Unused fields must be NULL.
 * @param slots Slots with the correct pair of IDs and (function) pointers.
 * @param private Some slots are currently considered private, if not true,
 *        these will be rejected.
 *
 * @returns A new (bound) ArrayMethod object.
 */
NPY_NO_EXPORT PyBoundArrayMethodObject *
PyArrayMethod_FromSpec_int(PyArrayMethod_Spec *spec, int private)
{
    int nargs = spec->nin + spec->nout;

    if (spec->name == NULL) {
        spec->name = "<unknown>";
    }

    if (validate_spec(spec) < 0) {
        return NULL;
    }

    PyBoundArrayMethodObject *res;
    res = PyObject_New(PyBoundArrayMethodObject, &PyBoundArrayMethod_Type);
    if (res == NULL) {
        return NULL;
    }
    res->nin = spec->nin;
    res->nout = spec->nout;
    res->method = NULL;

    res->dtypes = PyMem_Malloc(sizeof(PyArray_DTypeMeta *) * nargs);
    if (res->dtypes == NULL) {
        Py_DECREF(res);
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i < nargs ; i++) {
        Py_XINCREF(spec->dtypes[i]);
        res->dtypes[i] = spec->dtypes[i];
    }

    res->method = PyObject_New(PyArrayMethodObject, &PyArrayMethod_Type);
    if (res->method == NULL) {
        Py_DECREF(res);
        PyErr_NoMemory();
        return NULL;
    }
    memset((char *)(res->method) + sizeof(PyObject), 0,
           sizeof(PyArrayMethodObject) - sizeof(PyObject));

    res->method->flags = spec->flags;
    res->method->casting = spec->casting;
    if (fill_arraymethod_from_slots(res, spec, private) < 0) {
        Py_DECREF(res);
        return NULL;
    }

    ssize_t length = strlen(spec->name);
    res->method->name = PyMem_Malloc(length + 1);
    if (res->method->name == NULL) {
        Py_DECREF(res);
        PyErr_NoMemory();
        return NULL;
    }
    strcpy(res->method->name, spec->name);

    return res;
}


static void
arraymethod_dealloc(PyObject *self)
{
    PyArrayMethodObject *meth;
    meth = ((PyArrayMethodObject *)self);

    PyMem_Free(meth->name);

    Py_TYPE(self)->tp_free(self);
}


NPY_NO_EXPORT PyTypeObject PyArrayMethod_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy._ArrayMethod",
    .tp_basicsize = sizeof(PyArrayMethodObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = arraymethod_dealloc,
};



static PyObject *
boundarraymethod_repr(PyBoundArrayMethodObject *self)
{
    PyObject *dtypes = PyTuple_New(self->nin + self->nout);
    if (dtypes == NULL) {
        return NULL;
    }
    for (int i = 0; i < self->nin + self->nout; i++) {
        Py_INCREF(self->dtypes[i]);
        PyTuple_SET_ITEM(dtypes, i, (PyObject *)self->dtypes[i]);
    }
    return PyUnicode_FromFormat(
            "<np._BoundArrayMethod `%s` for dtypes %S>",
            self->method->name, dtypes);
}


static void
boundarraymethod_dealloc(PyObject *self)
{
    PyBoundArrayMethodObject *meth;
    meth = ((PyBoundArrayMethodObject *)self);

    for (int i = 0; i < meth->nin + meth->nout; i++) {
        Py_XDECREF(meth->dtypes[i]);
    }
    PyMem_Free(meth->dtypes);

    Py_XDECREF(meth->method);

    Py_TYPE(self)->tp_free(self);
}


/*
 * Calls resolve_descriptors() and returns the casting level and the resolved
 * descriptors as a tuple. If the operation is impossible returns (-1, None).
 * May raise an error, but usually should not.
 * The function validates the casting attribute compared to the returned
 * casting level.
 */
static PyObject *
boundarraymethod__resolve_descripors(
        PyBoundArrayMethodObject *self, PyObject *descr_tuple)
{
    PyArray_Descr *given_descrs[NPY_MAXARGS];
    PyArray_Descr *loop_descrs[NPY_MAXARGS];

    if (!PyTuple_CheckExact(descr_tuple) ||
            PyTuple_Size(descr_tuple) != self->nin + self->nout) {
        PyErr_Format(PyExc_ValueError,
                "_resolve_descriptors() takes exactly one tuple with as many "
                "elements as the method takes arguments (%d+%d).",
                self->nin, self->nout);
        return NULL;
    }

    for (int i = 0; i < self->nin + self->nout; i++) {
        PyObject *tmp = PyTuple_GetItem(descr_tuple, i);
        if (tmp == NULL) {
            return NULL;
        }
        else if (tmp == Py_None) {
            if (i < self->nin) {
                PyErr_SetString(PyExc_ValueError,
                        "only output dtypes may be omitted (set to None).");
                return NULL;
            }
            given_descrs[i] = NULL;
        }
        else if (PyArray_DescrCheck(tmp)) {
            if (Py_TYPE(tmp) != (PyTypeObject *)self->dtypes[i]) {
                PyErr_Format(PyExc_ValueError,
                        "input dtype %S was not an exact instance of the bound "
                        "DType class %S.", tmp, self->dtypes[i]);
                return NULL;
            }
            given_descrs[i] = (PyArray_Descr *)tmp;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                    "dtype tuple can only contain dtype instances or None.");
            return NULL;
        }
    }

    PyArrayMethod_Context context = {
        .caller = NULL,
        .method = self->method,
        .dtypes = self->dtypes,
        .nin = self->nin,
        .nout = self->nout,
        .descriptors = NULL,  /* Used after resolve_descriptors */
    };
    NPY_CASTING casting = self->method->resolve_descriptors(
            &context, given_descrs, loop_descrs);

    if (casting < 0 && PyErr_Occurred()) {
        return NULL;
    }
    else if (casting < 0) {
        return Py_BuildValue("iO", casting, Py_None);
    }

    PyObject *result_tuple = PyTuple_New(self->nin + self->nout);
    if (result_tuple == NULL) {
        return NULL;
    }
    for (int i = 0; i < self->nin + self->nout; i++) {
        /* transfer ownership to the tuple. */
        PyTuple_SET_ITEM(result_tuple, i, (PyObject *)loop_descrs[i]);
    }

    /*
     * The casting flags should be the most generic casting level (except the
     * cast-is-view flag.  If no input is parametric, it must match exactly.
     */
    int parametric = 0;
    for (int i = 0; i < self->nin + self->nout; i++) {
        if (self->dtypes[i]->parametric) {
            parametric = 1;
            break;
        }
    }
    if (!parametric) {
        /*
         * Non-parametric can only mismatch if it switches from no to equiv
         * (e.g. due to byteorder changes).
         */
        if (self->method->casting != (casting & ~_NPY_CAST_IS_VIEW) &&
                !(self->method->casting == NPY_NO_CASTING &&
                  casting == NPY_EQUIV_CASTING)) {
            PyErr_Format(PyExc_RuntimeError,
                    "resolve_descriptors cast level did not match stored one "
                    "(expected %d, got %d) for method %s",
                    self->method->casting, (casting & ~_NPY_CAST_IS_VIEW),
                    self->method->name);
            Py_DECREF(result_tuple);
            return NULL;
        }
    }
    else {
        NPY_CASTING cast = casting & ~_NPY_CAST_IS_VIEW;
        if (cast != PyArray_MinCastSafety(cast, self->method->casting)) {
            PyErr_Format(PyExc_RuntimeError,
                    "resolve_descriptors cast level did not match stored one "
                    "(expected %d, got %d) for method %s",
                    self->method->casting, (casting & ~_NPY_CAST_IS_VIEW),
                    self->method->name);
            Py_DECREF(result_tuple);
            return NULL;
        }
    }

    return Py_BuildValue("iN", casting, result_tuple);
}


PyMethodDef boundarraymethod_methods[] = {
    {"_resolve_descriptors", (PyCFunction)boundarraymethod__resolve_descripors,
     METH_O, "Resolve the given dtypes."},
    {NULL, 0, 0, NULL},
};


static PyObject *
boundarraymethod__supports_unaligned(PyBoundArrayMethodObject *self)
{
    return PyBool_FromLong(self->method->flags & NPY_METH_SUPPORTS_UNALIGNED);
}


PyGetSetDef boundarraymethods_getters[] = {
    {"_supports_unaligned",
     (getter)boundarraymethod__supports_unaligned, NULL,
     "whether the method supports unaligned inputs/outputs.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};


NPY_NO_EXPORT PyTypeObject PyBoundArrayMethod_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy._BoundArrayMethod",
    .tp_basicsize = sizeof(PyBoundArrayMethodObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_repr = (reprfunc)boundarraymethod_repr,
    .tp_dealloc = boundarraymethod_dealloc,
    .tp_methods = boundarraymethod_methods,
    .tp_getset = boundarraymethods_getters,
};
