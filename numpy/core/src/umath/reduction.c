/*
 * This file implements generic methods for computing reductions on arrays.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */
#define _UMATHMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "npy_config.h"
#define PY_ARRAY_UNIQUE_SYMBOL _npy_umathmodule_ARRAY_API
#define NO_IMPORT_ARRAY

#include <numpy/arrayobject.h>

#include "npy_config.h"
#include "npy_pycompat.h"

#include "numpy/ufuncobject.h"
#include "lowlevel_strided_loops.h"
#include "reduction.h"
#include "extobj.h"  /* for _check_ufunc_fperr */

/*
 * Allocates a result array for a reduction operation, with
 * dimensions matching 'arr' except set to 1 with 0 stride
 * wherever axis_flags is True. Dropping the reduction axes
 * from the result must be done later by the caller once the
 * computation is complete.
 *
 * This function always allocates a base class ndarray.
 *
 * If 'dtype' isn't NULL, this function steals its reference.
 */
static PyArrayObject *
allocate_reduce_result(PyArrayObject *arr, npy_bool *axis_flags,
                        PyArray_Descr *dtype, int subok)
{
    npy_intp strides[NPY_MAXDIMS], stride;
    npy_intp shape[NPY_MAXDIMS], *arr_shape = PyArray_DIMS(arr);
    npy_stride_sort_item strideperm[NPY_MAXDIMS];
    int idim, ndim = PyArray_NDIM(arr);

    if (dtype == NULL) {
        dtype = PyArray_DTYPE(arr);
        Py_INCREF(dtype);
    }

    PyArray_CreateSortedStridePerm(PyArray_NDIM(arr),
                                    PyArray_STRIDES(arr), strideperm);

    /* Build the new strides and shape */
    stride = dtype->elsize;
    memcpy(shape, arr_shape, ndim * sizeof(shape[0]));
    for (idim = ndim-1; idim >= 0; --idim) {
        npy_intp i_perm = strideperm[idim].perm;
        if (axis_flags[i_perm]) {
            strides[i_perm] = 0;
            shape[i_perm] = 1;
        }
        else {
            strides[i_perm] = stride;
            stride *= shape[i_perm];
        }
    }

    /* Finally, allocate the array */
    return (PyArrayObject *)PyArray_NewFromDescr(
                                    subok ? Py_TYPE(arr) : &PyArray_Type,
                                    dtype, ndim, shape, strides,
                                    NULL, 0, subok ? (PyObject *)arr : NULL);
}

/*
 * Conforms an output parameter 'out' to have 'ndim' dimensions
 * with dimensions of size one added in the appropriate places
 * indicated by 'axis_flags'.
 *
 * The return value is a view into 'out'.
 */
static PyArrayObject *
conform_reduce_result(int ndim, npy_bool *axis_flags,
                      PyArrayObject *out, int keepdims, const char *funcname,
                      int need_copy)
{
    npy_intp strides[NPY_MAXDIMS], shape[NPY_MAXDIMS];
    npy_intp *strides_out = PyArray_STRIDES(out);
    npy_intp *shape_out = PyArray_DIMS(out);
    int idim, idim_out, ndim_out = PyArray_NDIM(out);
    PyArray_Descr *dtype;
    PyArrayObject_fields *ret;

    /*
     * If the 'keepdims' parameter is true, do a simpler validation and
     * return a new reference to 'out'.
     */
    if (keepdims) {
        if (PyArray_NDIM(out) != ndim) {
            PyErr_Format(PyExc_ValueError,
                    "output parameter for reduction operation %s "
                    "has the wrong number of dimensions (must match "
                    "the operand's when keepdims=True)", funcname);
            return NULL;
        }

        for (idim = 0; idim < ndim; ++idim) {
            if (axis_flags[idim]) {
                if (shape_out[idim] != 1) {
                    PyErr_Format(PyExc_ValueError,
                            "output parameter for reduction operation %s "
                            "has a reduction dimension not equal to one "
                            "(required when keepdims=True)", funcname);
                    return NULL;
                }
            }
        }

        Py_INCREF(out);
        return out;
    }

    /* Construct the strides and shape */
    idim_out = 0;
    for (idim = 0; idim < ndim; ++idim) {
        if (axis_flags[idim]) {
            strides[idim] = 0;
            shape[idim] = 1;
        }
        else {
            if (idim_out >= ndim_out) {
                PyErr_Format(PyExc_ValueError,
                        "output parameter for reduction operation %s "
                        "does not have enough dimensions", funcname);
                return NULL;
            }
            strides[idim] = strides_out[idim_out];
            shape[idim] = shape_out[idim_out];
            ++idim_out;
        }
    }

    if (idim_out != ndim_out) {
        PyErr_Format(PyExc_ValueError,
                "output parameter for reduction operation %s "
                "has too many dimensions", funcname);
        return NULL;
    }

    /* Allocate the view */
    dtype = PyArray_DESCR(out);
    Py_INCREF(dtype);

    ret = (PyArrayObject_fields *)PyArray_NewFromDescr(&PyArray_Type,
                               dtype,
                               ndim, shape,
                               strides,
                               PyArray_DATA(out),
                               PyArray_FLAGS(out),
                               NULL);
    if (ret == NULL) {
        return NULL;
    }

    Py_INCREF(out);
    if (PyArray_SetBaseObject((PyArrayObject *)ret, (PyObject *)out) < 0) {
        Py_DECREF(ret);
        return NULL;
    }

    if (need_copy) {
        PyArrayObject *ret_copy;

        ret_copy = (PyArrayObject *)PyArray_NewLikeArray(
            (PyArrayObject *)ret, NPY_ANYORDER, NULL, 0);
        if (ret_copy == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        if (PyArray_CopyInto(ret_copy, (PyArrayObject *)ret) != 0) {
            Py_DECREF(ret);
            Py_DECREF(ret_copy);
            return NULL;
        }

        Py_INCREF(ret);
        if (PyArray_SetWritebackIfCopyBase(ret_copy, (PyArrayObject *)ret) < 0) {
            Py_DECREF(ret);
            Py_DECREF(ret_copy);
            return NULL;
        }

        return ret_copy;
    }
    else {
        return (PyArrayObject *)ret;
    }
}

/*
 * Creates a result for reducing 'operand' along the axes specified
 * in 'axis_flags'. If 'dtype' isn't NULL, this function steals a
 * reference to 'dtype'.
 *
 * If 'out' isn't NULL, this function creates a view conforming
 * to the number of dimensions of 'operand', adding a singleton dimension
 * for each reduction axis specified. In this case, 'dtype' is ignored
 * (but its reference is still stolen), and the caller must handle any
 * type conversion/validity check for 'out'
 *
 * If 'subok' is true, creates a result with the subtype of 'operand',
 * otherwise creates on with the base ndarray class.
 *
 * If 'out' is NULL, it allocates a new array whose shape matches that of
 * 'operand', except for at the reduction axes. If 'dtype' is NULL, the dtype
 * of 'operand' is used for the result.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_CreateReduceResult(PyArrayObject *operand, PyArrayObject *out,
                           PyArray_Descr *dtype, npy_bool *axis_flags,
                           int keepdims, int subok,
                           const char *funcname)
{
    PyArrayObject *result;

    if (out == NULL) {
        /* This function steals the reference to 'dtype' */
        result = allocate_reduce_result(operand, axis_flags, dtype, subok);
    }
    else {
        int need_copy = 0;

        if (solve_may_share_memory(operand, out, 1) != 0) {
            need_copy = 1;
        }

        /* Steal the dtype reference */
        Py_XDECREF(dtype);
        result = conform_reduce_result(PyArray_NDIM(operand), axis_flags,
                                       out, keepdims, funcname, need_copy);
    }

    return result;
}

/*
 * Checks that there are only zero or one dimensions selected in 'axis_flags',
 * and raises an error about a non-reorderable reduction if not.
 */
static int
check_nonreorderable_axes(int ndim, npy_bool *axis_flags, const char *funcname)
{
    int idim, single_axis = 0;
    for (idim = 0; idim < ndim; ++idim) {
        if (axis_flags[idim]) {
            if (single_axis) {
                PyErr_Format(PyExc_ValueError,
                        "reduction operation '%s' is not reorderable, "
                        "so only one axis may be specified",
                        funcname);
                return -1;
            }
            else {
                single_axis = 1;
            }
        }
    }

    return 0;
}

/*
 * This function initializes a result array for a reduction operation
 * which has no identity. This means it needs to copy the first element
 * it sees along the reduction axes to result, then return a view of
 * the operand which excludes that element.
 *
 * If a reduction has an identity, such as 0 or 1, the result should be
 * initialized by calling PyArray_AssignZero(result, NULL, NULL) or
 * PyArray_AssignOne(result, NULL, NULL), because this function raises an
 * exception when there are no elements to reduce (which appropriate iff the
 * reduction operation has no identity).
 *
 * This means it copies the subarray indexed at zero along each reduction axis
 * into 'result', then returns a view into 'operand' excluding those copied
 * elements.
 *
 * result  : The array into which the result is computed. This must have
 *           the same number of dimensions as 'operand', but for each
 *           axis i where 'axis_flags[i]' is True, it has a single element.
 * operand : The array being reduced.
 * axis_flags : An array of boolean flags, one for each axis of 'operand'.
 *              When a flag is True, it indicates to reduce along that axis.
 * reorderable : If True, the reduction being done is reorderable, which
 *               means specifying multiple axes of reduction at once is ok,
 *               and the reduction code may calculate the reduction in an
 *               arbitrary order. The calculation may be reordered because
 *               of cache behavior or multithreading requirements.
 * out_skip_first_count : This gets populated with the number of first-visit
 *                        elements that should be skipped during the
 *                        iteration loop.
 * funcname : The name of the reduction operation, for the purpose of
 *            better quality error messages. For example, "numpy.max"
 *            would be a good name for NumPy's max function.
 *
 * Returns a view which contains the remaining elements on which to do
 * the reduction.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_InitializeReduceResult(
                    PyArrayObject *result, PyArrayObject *operand,
                    npy_bool *axis_flags, int reorderable,
                    npy_intp *out_skip_first_count, const char *funcname)
{
    npy_intp *strides, *shape, shape_orig[NPY_MAXDIMS];
    PyArrayObject *op_view = NULL;
    int idim, ndim, nreduce_axes;

    ndim = PyArray_NDIM(operand);

    /* Default to no skipping first-visit elements in the iteration */
    *out_skip_first_count = 0;

    /*
     * If this reduction is non-reorderable, make sure there are
     * only 0 or 1 axes in axis_flags.
     */
    if (!reorderable && check_nonreorderable_axes(ndim,
                                    axis_flags, funcname) < 0) {
        return NULL;
    }

    /* Take a view into 'operand' which we can modify. */
    op_view = (PyArrayObject *)PyArray_View(operand, NULL, &PyArray_Type);
    if (op_view == NULL) {
        return NULL;
    }

    /*
     * Now copy the subarray of the first element along each reduction axis,
     * then return a view to the rest.
     *
     * Adjust the shape to only look at the first element along
     * any of the reduction axes. We count the number of reduction axes
     * at the same time.
     */
    shape = PyArray_SHAPE(op_view);
    nreduce_axes = 0;
    memcpy(shape_orig, shape, ndim * sizeof(npy_intp));
    for (idim = 0; idim < ndim; ++idim) {
        if (axis_flags[idim]) {
            if (shape[idim] == 0) {
                PyErr_Format(PyExc_ValueError,
                             "zero-size array to reduction operation %s "
                             "which has no identity",
                             funcname);
                Py_DECREF(op_view);
                return NULL;
            }
            shape[idim] = 1;
            ++nreduce_axes;
        }
    }

    /*
     * Copy the elements into the result to start.
     */
    if (PyArray_CopyInto(result, op_view) < 0) {
        Py_DECREF(op_view);
        return NULL;
    }

    /*
     * If there is one reduction axis, adjust the view's
     * shape to only look at the remaining elements
     */
    if (nreduce_axes == 1) {
        strides = PyArray_STRIDES(op_view);
        for (idim = 0; idim < ndim; ++idim) {
            if (axis_flags[idim]) {
                shape[idim] = shape_orig[idim] - 1;
                ((PyArrayObject_fields *)op_view)->data += strides[idim];
            }
        }
    }
    /* If there are zero reduction axes, make the view empty */
    else if (nreduce_axes == 0) {
        for (idim = 0; idim < ndim; ++idim) {
            shape[idim] = 0;
        }
    }
    /*
     * Otherwise iterate over the whole operand, but tell the inner loop
     * to skip the elements we already copied by setting the skip_first_count.
     */
    else {
        *out_skip_first_count = PyArray_SIZE(result);

        Py_DECREF(op_view);
        Py_INCREF(operand);
        op_view = operand;
    }

    return op_view;
}

/*
 * This function executes all the standard NumPy reduction function
 * boilerplate code, just calling assign_identity and the appropriate
 * inner loop function where necessary.
 *
 * operand     : The array to be reduced.
 * out         : NULL, or the array into which to place the result.
 * wheremask   : NOT YET SUPPORTED, but this parameter is placed here
 *               so that support can be added in the future without breaking
 *               API compatibility. Pass in NULL.
 * operand_dtype : The dtype the inner loop expects for the operand.
 * result_dtype : The dtype the inner loop expects for the result.
 * casting     : The casting rule to apply to the operands.
 * axis_flags  : Flags indicating the reduction axes of 'operand'.
 * reorderable : If True, the reduction being done is reorderable, which
 *               means specifying multiple axes of reduction at once is ok,
 *               and the reduction code may calculate the reduction in an
 *               arbitrary order. The calculation may be reordered because
 *               of cache behavior or multithreading requirements.
 * keepdims    : If true, leaves the reduction dimensions in the result
 *               with size one.
 * subok       : If true, the result uses the subclass of operand, otherwise
 *               it is always a base class ndarray.
 * assign_identity : If NULL, PyArray_InitializeReduceResult is used, otherwise
 *               this function is called to initialize the result to
 *               the reduction's unit.
 * loop        : The loop which does the reduction.
 * data        : Data which is passed to assign_identity and the inner loop.
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
PyUFunc_ReduceWrapper(PyArrayObject *operand, PyArrayObject *out,
                      PyArrayObject *wheremask,
                      PyArray_Descr *operand_dtype,
                      PyArray_Descr *result_dtype,
                      NPY_CASTING casting,
                      npy_bool *axis_flags, int reorderable,
                      int keepdims,
                      int subok,
                      PyArray_AssignReduceIdentityFunc *assign_identity,
                      PyArray_ReduceLoopFunc *loop,
                      void *data, npy_intp buffersize, const char *funcname,
                      int errormask)
{
    PyArrayObject *result = NULL, *op_view = NULL;
    npy_intp skip_first_count = 0;

    /* Iterator parameters */
    NpyIter *iter = NULL;
    PyArrayObject *op[2];
    PyArray_Descr *op_dtypes[2];
    npy_uint32 flags, op_flags[2];

    /* Validate that the parameters for future expansion are NULL */
    if (wheremask != NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Reduce operations in NumPy do not yet support "
                "a where mask");
        return NULL;
    }

    /*
     * This either conforms 'out' to the ndim of 'operand', or allocates
     * a new array appropriate for this reduction.
     *
     * A new array with WRITEBACKIFCOPY is allocated if operand and out have memory
     * overlap.
     */
    Py_INCREF(result_dtype);
    result = PyArray_CreateReduceResult(operand, out,
                            result_dtype, axis_flags,
                            keepdims, subok, funcname);
    if (result == NULL) {
        goto fail;
    }

    /*
     * Initialize the result to the reduction unit if possible,
     * otherwise copy the initial values and get a view to the rest.
     */
    if (assign_identity != NULL) {
        /*
         * If this reduction is non-reorderable, make sure there are
         * only 0 or 1 axes in axis_flags.
         */
        if (!reorderable && check_nonreorderable_axes(PyArray_NDIM(operand),
                                        axis_flags, funcname) < 0) {
            goto fail;
        }

        if (assign_identity(result, data) < 0) {
            goto fail;
        }
        op_view = operand;
        Py_INCREF(op_view);
    }
    else {
        op_view = PyArray_InitializeReduceResult(result, operand,
                            axis_flags, reorderable,
                            &skip_first_count, funcname);
        if (op_view == NULL) {
            goto fail;
        }
        /* empty op_view signals no reduction; but 0-d arrays cannot be empty */
        if ((PyArray_SIZE(op_view) == 0) || (PyArray_NDIM(operand) == 0)) {
            Py_DECREF(op_view);
            op_view = NULL;
            goto finish;
        }
    }

    /* Set up the iterator */
    op[0] = result;
    op[1] = op_view;
    op_dtypes[0] = result_dtype;
    op_dtypes[1] = operand_dtype;

    flags = NPY_ITER_BUFFERED |
            NPY_ITER_EXTERNAL_LOOP |
            NPY_ITER_GROWINNER |
            NPY_ITER_DONT_NEGATE_STRIDES |
            NPY_ITER_ZEROSIZE_OK |
            NPY_ITER_REDUCE_OK |
            NPY_ITER_REFS_OK;
    op_flags[0] = NPY_ITER_READWRITE |
                  NPY_ITER_ALIGNED |
                  NPY_ITER_NO_SUBTYPE;
    op_flags[1] = NPY_ITER_READONLY |
                  NPY_ITER_ALIGNED;

    iter = NpyIter_AdvancedNew(2, op, flags,
                               NPY_KEEPORDER, casting,
                               op_flags,
                               op_dtypes,
                               -1, NULL, NULL, buffersize);
    if (iter == NULL) {
        goto fail;
    }

    /* Start with the floating-point exception flags cleared */
    PyUFunc_clearfperr();

    if (NpyIter_GetIterSize(iter) != 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strideptr;
        npy_intp *countptr;
        int needs_api;

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        strideptr = NpyIter_GetInnerStrideArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);

        needs_api = NpyIter_IterationNeedsAPI(iter);

        /* Straightforward reduction */
        if (loop == NULL) {
            PyErr_Format(PyExc_RuntimeError,
                    "reduction operation %s did not supply an "
                    "inner loop function", funcname);
            goto fail;
        }

        if (loop(iter, dataptr, strideptr, countptr,
                        iternext, needs_api, skip_first_count, data) < 0) {

            goto fail;
        }
    }
    
    /* Check whether any errors occurred during the loop */
    if (PyErr_Occurred() ||
            _check_ufunc_fperr(errormask, NULL, "reduce") < 0) {
        goto fail;
    }

    NpyIter_Deallocate(iter);
    Py_DECREF(op_view);

finish:
    /* Strip out the extra 'one' dimensions in the result */
    if (out == NULL) {
        if (!keepdims) {
            PyArray_RemoveAxesInPlace(result, axis_flags);
        }
    }
    else {
        PyArray_ResolveWritebackIfCopy(result); /* prevent spurious warnings */
        Py_DECREF(result);
        result = out;
        Py_INCREF(result);
    }

    return result;

fail:
    PyArray_ResolveWritebackIfCopy(result); /* prevent spurious warnings */
    Py_XDECREF(result);
    Py_XDECREF(op_view);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    return NULL;
}
