#ifndef _NPY_PRIVATE__REDUCTION_H_
#define _NPY_PRIVATE__REDUCTION_H_

/************************************************************
 * Typedefs used by PyArray_ReduceWrapper, new in 1.7.
 ************************************************************/

/*
 * This is a function for assigning a reduction identity to the result,
 * before doing the reduction computation. The
 * value in 'data' is passed through from PyArray_ReduceWrapper.
 *
 * This function could, for example, simply be a call like
 *      return PyArray_AssignZero(result, NULL, NULL);
 *
 * It should return -1 on failure, or 0 on success.
 */
typedef int (PyArray_AssignReduceIdentityFunc)(PyArrayObject *result,
                                               void *data);

/*
 * Inner definition of the reduce loop, only used for a static function.
 * At some point around NumPy 1.6, there was probably an intention to make
 * the reduce loop customizable at this level (per ufunc?).
 *
 * TODO: This should be refactored/removed.
 */
typedef int (PyArray_ReduceLoopFunc)(PyArrayMethod_Context *context,
        PyArrayMethod_StridedLoop *strided_loop, NpyAuxData *auxdata,
        NpyIter *iter, char **dataptrs, npy_intp const *strides,
        npy_intp const *countptr, NpyIter_IterNextFunc *iternext,
        int needs_api, npy_intp skip_first_count);

/*
 * This function executes all the standard NumPy reduction function
 * boilerplate code, just calling the appropriate inner loop function where
 * necessary.
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
 * identity    : If Py_None, PyArray_CopyInitialReduceValues is used, otherwise
 *               this value is used to initialize the result to
 *               the reduction's unit.
 * loop        : The loop which does the reduction.
 * data        : Data which is passed to the inner loop.
 * buffersize  : Buffer size for the iterator. For the default, pass in 0.
 * funcname    : The name of the reduction function, for error messages.
 * errormask   : forwarded from _get_bufsize_errmask
 */
NPY_NO_EXPORT PyArrayObject *
PyUFunc_ReduceWrapper(PyArrayMethod_Context *context,
        PyArrayObject *operand, PyArrayObject *out, PyArrayObject *wheremask,
        npy_bool *axis_flags, int keepdims,
        PyObject *initial, PyArray_ReduceLoopFunc *loop,
        npy_intp buffersize, const char *funcname, int errormask);

#endif
