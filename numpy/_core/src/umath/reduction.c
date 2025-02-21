/*
 * This file implements generic methods for computing reductions on arrays.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "npy_config.h"
#include "numpy/arrayobject.h"


#include "array_assign.h"
#include "array_coercion.h"
#include "array_method.h"
#include "ctors.h"
#include "refcount.h"

#include "numpy/ufuncobject.h"
#include "lowlevel_strided_loops.h"
#include "reduction.h"
#include "extobj.h"  /* for _check_ufunc_fperr */


/*
 * Count the number of dimensions selected in 'axis_flags'
 */
static int
count_axes(int ndim, const npy_bool *axis_flags)
{
    int idim;
    int naxes = 0;

    for (idim = 0; idim < ndim; ++idim) {
        if (axis_flags[idim]) {
            naxes++;
        }
    }
    return naxes;
}

/*
 * This function initializes a result array for a reduction operation
 * which has no identity. This means it needs to copy the first element
 * it sees along the reduction axes to result.
 *
 * If a reduction has an identity, such as 0 or 1, the result should be
 * fully initialized to the identity, because this function raises an
 * exception when there are no elements to reduce (which is appropriate if,
 * and only if, the reduction operation has no identity).
 *
 * This means it copies the subarray indexed at zero along each reduction axis
 * into 'result'.
 *
 * result  : The array into which the result is computed. This must have
 *           the same number of dimensions as 'operand', but for each
 *           axis i where 'axis_flags[i]' is True, it has a single element.
 * operand : The array being reduced.
 * axis_flags : An array of boolean flags, one for each axis of 'operand'.
 *              When a flag is True, it indicates to reduce along that axis.
 * funcname : The name of the reduction operation, for the purpose of
 *            better quality error messages. For example, "numpy.max"
 *            would be a good name for NumPy's max function.
 *
 * Returns -1 if an error occurred, and otherwise the reduce arrays size,
 * which is the number of elements already initialized.
 */
static npy_intp
PyArray_CopyInitialReduceValues(
                    PyArrayObject *result, PyArrayObject *operand,
                    const npy_bool *axis_flags, const char *funcname,
                    int keepdims)
{
    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];
    npy_intp *shape_orig = PyArray_SHAPE(operand);
    npy_intp *strides_orig = PyArray_STRIDES(operand);
    PyArrayObject *op_view = NULL;

    int ndim = PyArray_NDIM(operand);

    /*
     * Copy the subarray of the first element along each reduction axis.
     *
     * Adjust the shape to only look at the first element along
     * any of the reduction axes. If keepdims is False remove the axes
     * entirely.
     */
    int idim_out = 0;
    npy_intp size = 1;
    for (int idim = 0; idim < ndim; idim++) {
        if (axis_flags[idim]) {
            if (NPY_UNLIKELY(shape_orig[idim] == 0)) {
                PyErr_Format(PyExc_ValueError,
                        "zero-size array to reduction operation %s "
                        "which has no identity", funcname);
                return -1;
            }
            if (keepdims) {
                shape[idim_out] = 1;
                strides[idim_out] = 0;
                idim_out++;
            }
        }
        else {
            size *= shape_orig[idim];
            shape[idim_out] = shape_orig[idim];
            strides[idim_out] = strides_orig[idim];
            idim_out++;
        }
    }

    PyArray_Descr *descr = PyArray_DESCR(operand);
    Py_INCREF(descr);
    op_view = (PyArrayObject *)PyArray_NewFromDescr(
            &PyArray_Type, descr, idim_out, shape, strides,
            PyArray_DATA(operand), 0, NULL);
    if (op_view == NULL) {
        return -1;
    }

    /*
     * Copy the elements into the result to start.
     */
    int res = PyArray_CopyInto(result, op_view);
    Py_DECREF(op_view);
    if (res < 0) {
        return -1;
    }

    /*
     * If there were no reduction axes, we would already be done here.
     * Note that if there is only a single reduction axis, in principle the
     * iteration could be set up more efficiently here by removing that
     * axis before setting up the iterator (simplifying the iteration since
     * `skip_first_count` (the returned size) can be set to 0).
     */
    return size;
}

/*
 * This function executes all the standard NumPy reduction function
 * boilerplate code, just calling the appropriate inner loop function where
 * necessary.
 *
 * context     : The ArrayMethod context (with ufunc, method, and descriptors).
 * operand     : The array to be reduced.
 * out         : NULL, or the array into which to place the result.
 * wheremask   : Reduction mask of valid values used for `where=`.
 * axis_flags  : Flags indicating the reduction axes of 'operand'.
 * keepdims    : If true, leaves the reduction dimensions in the result
 *               with size one.
 * subok       : If true, the result uses the subclass of operand, otherwise
 *               it is always a base class ndarray.
 * initial     : Initial value, if NULL the default is fetched from the
 *               ArrayMethod (typically as the default from the ufunc).
 * loop        : `reduce_loop` from `ufunc_object.c`.  TODO: Refactor
 * buffersize  : Buffer size for the iterator. For the default, pass in 0.
 * funcname    : The name of the reduction function, for error messages.
 * errormask   : forwarded from _get_bufsize_errmask
 *
 * TODO FIXME: if you squint, this is essentially an second independent
 * implementation of generalized ufuncs with signature (i)->(), plus a few
 * extra bells and whistles. (Indeed, as far as I can tell, it was originally
 * split out to support a fancy version of count_nonzero... which is not
 * actually a reduction function at all, it's just a (i)->() function!) So
 * probably these two implementation should be merged into one. (In fact it
 * would be quite nice to support axis= and keepdims etc. for arbitrary
 * generalized ufuncs!)
 */
NPY_NO_EXPORT PyArrayObject *
PyUFunc_ReduceWrapper(PyArrayMethod_Context *context,
        PyArrayObject *operand, PyArrayObject *out, PyArrayObject *wheremask,
        npy_bool *axis_flags, int keepdims,
        PyObject *initial, PyArray_ReduceLoopFunc *loop,
        npy_intp buffersize, const char *funcname, int errormask)
{
    assert(loop != NULL);
    PyArrayObject *result = NULL;
    npy_intp skip_first_count = 0;

    /* Iterator parameters */
    NpyIter *iter = NULL;
    PyArrayObject *op[3];
    PyArray_Descr *op_dtypes[3];
    npy_uint32 it_flags, op_flags[3];
    /* Loop auxdata (must be freed on error) */
    NpyAuxData *auxdata = NULL;

    /* Set up the iterator */
    op[0] = out;
    op[1] = operand;
    op_dtypes[0] = context->descriptors[0];
    op_dtypes[1] = context->descriptors[1];

    /* Buffer to use when we need an initial value */
    char *initial_buf = NULL;

    /* More than one axis means multiple orders are possible */
    if (!(context->method->flags & NPY_METH_IS_REORDERABLE)
            && count_axes(PyArray_NDIM(operand), axis_flags) > 1) {
        PyErr_Format(PyExc_ValueError,
                "reduction operation '%s' is not reorderable, "
                "so at most one axis may be specified",
                funcname);
        goto fail;
    }

    it_flags = NPY_ITER_BUFFERED |
            NPY_ITER_EXTERNAL_LOOP |
            NPY_ITER_GROWINNER |
            NPY_ITER_ZEROSIZE_OK |
            NPY_ITER_REFS_OK |
            NPY_ITER_DELAY_BUFALLOC |
            /*
             * stride negation (if reorderable) could currently misalign the
             * first-visit and initial value copy logic.
             */
            NPY_ITER_DONT_NEGATE_STRIDES |
            NPY_ITER_COPY_IF_OVERLAP;

    op_flags[0] = NPY_ITER_READWRITE |
                  NPY_ITER_ALIGNED |
                  NPY_ITER_ALLOCATE |
                  NPY_ITER_NO_SUBTYPE;
    op_flags[1] = NPY_ITER_READONLY |
                  NPY_ITER_ALIGNED |
                  NPY_ITER_NO_BROADCAST;

    if (wheremask != NULL) {
        op[2] = wheremask;
        /* wheremask is guaranteed to be NPY_BOOL, so borrow its reference */
        op_dtypes[2] = PyArray_DESCR(wheremask);
        assert(op_dtypes[2]->type_num == NPY_BOOL);
        if (op_dtypes[2] == NULL) {
            goto fail;
        }
        op_flags[2] = NPY_ITER_READONLY;
    }
    /* Set up result array axes mapping, operand and wheremask use default */
    int result_axes[NPY_MAXDIMS];
    int *op_axes[3] = {result_axes, NULL, NULL};

    int curr_axis = 0;
    for (int i = 0; i < PyArray_NDIM(operand); i++) {
        if (axis_flags[i]) {
            if (keepdims) {
                result_axes[i] = NPY_ITER_REDUCTION_AXIS(curr_axis);
                curr_axis++;
            }
            else {
                result_axes[i] = NPY_ITER_REDUCTION_AXIS(-1);
            }
        }
        else {
            result_axes[i] = curr_axis;
            curr_axis++;
        }
    }
    if (out != NULL) {
        /* NpyIter does not raise a good error message in this common case. */
        if (NPY_UNLIKELY(curr_axis != PyArray_NDIM(out))) {
            if (keepdims) {
                PyErr_Format(PyExc_ValueError,
                        "output parameter for reduction operation %s has the "
                        "wrong number of dimensions: Found %d but expected %d "
                        "(must match the operand's when keepdims=True)",
                        funcname, PyArray_NDIM(out), curr_axis);
            }
            else {
                PyErr_Format(PyExc_ValueError,
                        "output parameter for reduction operation %s has the "
                        "wrong number of dimensions: Found %d but expected %d",
                        funcname, PyArray_NDIM(out), curr_axis);
            }
            goto fail;
        }
    }

    iter = NpyIter_AdvancedNew(wheremask == NULL ? 2 : 3, op, it_flags,
                               NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                               op_flags,
                               op_dtypes,
                               PyArray_NDIM(operand), op_axes, NULL, buffersize);
    if (iter == NULL) {
        goto fail;
    }

    npy_bool empty_iteration = NpyIter_GetIterSize(iter) == 0;
    result = NpyIter_GetOperandArray(iter)[0];

    /*
     * Get the initial value (if it exists).  If the iteration is empty
     * then we assume the reduction is also empty.  The reason is that when
     * the outer iteration is empty we just won't use the initial value
     * in any case.  (`np.sum(np.zeros((0, 3)), axis=0)` is a length 3
     * reduction but has an empty result.)
     */
    if ((initial == NULL && context->method->get_reduction_initial == NULL)
            || initial == Py_None) {
        /* There is no initial value, or initial value was explicitly unset */
    }
    else {
        /* Not all functions will need initialization, but init always: */
        initial_buf = PyMem_Calloc(1, op_dtypes[0]->elsize);
        if (initial_buf == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        if (initial != NULL) {
            /* must use user provided initial value */
            if (PyArray_Pack(op_dtypes[0], initial_buf, initial) < 0) {
                goto fail;
            }
        }
        else {
            /*
             * Fetch initial from ArrayMethod, we pretend the reduction is
             * empty when the iteration is.  This may be wrong, but when it is,
             * we will not need the identity as the result is also empty.
             */
            int has_initial = context->method->get_reduction_initial(
                    context, empty_iteration, initial_buf);
            if (has_initial < 0) {
                goto fail;
            }
            if (!has_initial) {
                /* We have no initial value available, free buffer to indicate */
                PyMem_FREE(initial_buf);
                initial_buf = NULL;
            }
        }
    }

    PyArrayMethod_StridedLoop *strided_loop;
    NPY_ARRAYMETHOD_FLAGS flags;

    npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
    if (wheremask != NULL) {
        if (PyArrayMethod_GetMaskedStridedLoop(context,
                1, strideptr, &strided_loop, &auxdata, &flags) < 0) {
            goto fail;
        }
    }
    else {
        if (context->method->get_strided_loop(context,
                1, 0, strideptr, &strided_loop, &auxdata, &flags) < 0) {
            goto fail;
        }
    }
    flags = PyArrayMethod_COMBINED_FLAGS(flags, NpyIter_GetTransferFlags(iter));

    int needs_api = (flags & NPY_METH_REQUIRES_PYAPI) != 0;
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* Start with the floating-point exception flags cleared */
        npy_clear_floatstatus_barrier((char*)&iter);
    }

    /*
     * Initialize the result to the reduction unit if possible,
     * otherwise copy the initial values and get a view to the rest.
     */
    if (initial_buf != NULL) {
        /* Loop provided an identity or default value, assign to result. */
        int ret = raw_array_assign_scalar(
                PyArray_NDIM(result), PyArray_DIMS(result),
                PyArray_DESCR(result),
                PyArray_BYTES(result), PyArray_STRIDES(result),
                op_dtypes[0], initial_buf);
        if (ret < 0) {
            goto fail;
        }
    }
    else {
        /* Can only use where with an initial (from identity or argument) */
        if (wheremask != NULL) {
            PyErr_Format(PyExc_ValueError,
                    "reduction operation '%s' does not have an identity, "
                    "so to use a where mask one has to specify 'initial'",
                    funcname);
            goto fail;
        }

        /*
         * For 1-D skip_first_count could be optimized to 0, but no-identity
         * reductions are not super common.
         * (see also comment in CopyInitialReduceValues)
         */
        skip_first_count = PyArray_CopyInitialReduceValues(
                result, operand, axis_flags, funcname, keepdims);
        if (skip_first_count < 0) {
            goto fail;
        }
    }

    if (!NpyIter_Reset(iter, NULL)) {
        goto fail;
    }

    if (!empty_iteration) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *countptr;

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);

        if (loop(context, strided_loop, auxdata,
                iter, dataptr, strideptr, countptr, iternext,
                needs_api, skip_first_count) < 0) {
            goto fail;
        }
    }

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* NOTE: We could check float errors even on error */
        if (_check_ufunc_fperr(errormask, "reduce") < 0) {
            goto fail;
        }
    }

    if (out != NULL) {
        result = out;
    }
    Py_INCREF(result);

    if (initial_buf != NULL && PyDataType_REFCHK(PyArray_DESCR(result))) {
        PyArray_ClearBuffer(PyArray_DESCR(result), initial_buf, 0, 1, 1);
    }
    PyMem_FREE(initial_buf);
    NPY_AUXDATA_FREE(auxdata);
    if (!NpyIter_Deallocate(iter)) {
        Py_DECREF(result);
        return NULL;
    }
    return result;

fail:
    if (initial_buf != NULL && PyDataType_REFCHK(op_dtypes[0])) {
        PyArray_ClearBuffer(op_dtypes[0], initial_buf, 0, 1, 1);
    }
    PyMem_FREE(initial_buf);
    NPY_AUXDATA_FREE(auxdata);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    return NULL;
}
