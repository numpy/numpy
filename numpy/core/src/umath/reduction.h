#ifndef _NPY_PRIVATE__REDUCTION_H_
#define _NPY_PRIVATE__REDUCTION_H_

/*
 * This function initializes a result array for a reduction operation
 * which has no identity. This means it needs to copy the first element
 * it sees along the reduction axes to result, then return a view of
 * the operand which excludes that element.
 *
 * If a reduction has an identity, such as 0 or 1, the result should
 * be initialized by calling PyArray_AssignZero(result, NULL, NULL)
 * or PyArray_AssignOne(result, NULL, NULL), because this
 * function raises an exception when there are no elements to reduce.
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
                    npy_intp *out_skip_first_count, const char *funcname);

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
                    const char *funcname);

#endif
