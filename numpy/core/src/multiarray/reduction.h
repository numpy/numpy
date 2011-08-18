#ifndef _NPY_PRIVATE__REDUCTION_H_
#define _NPY_PRIVATE__REDUCTION_H_

/*
 * This is a function for assigning a reduction unit to the result,
 * before doing the reduction computation. If 'preservena' is True,
 * any masked NA values in 'result' should not be overwritten. The
 * value in 'data' is passed through from PyArray_ReduceWrapper.
 *
 * This function could, for example, simply be a call like
 *      return PyArray_AssignZero(result, NULL, preservena, NULL);
 *
 * It should return -1 on failure, or 0 on success.
 */
typedef int (PyArray_AssignReduceUnitFunc)(PyArrayObject *result,
                                            int preservena, void *data);

/*
 * This is a function for the inner reduce loop. Both the unmasked and
 * masked variants have the same prototype, but should behave differently.
 *
 * The needs_api parameter indicates whether it's ok to release the GIL during
 * the inner loop, such as when the iternext() function never calls
 * a function which could raise a Python exception.
 *
 * The unmasked inner loop gets two data pointers and two strides, and should
 * look roughly like this:
 *      NPY_BEGIN_THREADS_DEF;
 *      if (!needs_api) {
 *          NPY_BEGIN_THREADS;
 *      }
 *      do {
 *          char *data0 = dataptr[0], *data1 = dataptr[1];
 *          npy_intp stride0 = strideptr[0], stride1 = strideptr[1];
 *          npy_intp count = *countptr;
 *
 *          while (count--) {
 *              *(result_t *)data0 = my_reduce_op(*(result_t *)data0,
 *                                                *(operand_t *)data1);
 *              data0 += stride0;
 *              data1 += stride1;
 *          }
 *      } while (iternext(iter));
 *      if (!needs_api) {
 *          NPY_END_THREADS;
 *      }
 *
 * The masked inner loop gets three data pointers and three strides, and
 * should look roughly like this:
 *      NPY_BEGIN_THREADS_DEF;
 *      if (!needs_api) {
 *          NPY_BEGIN_THREADS;
 *      }
 *      do {
 *          char *data0 = dataptr[0], *data1 = dataptr[1], *data2 = dataptr[2];
 *          npy_intp stride0 = strideptr[0], stride1 = strideptr[1],
 *                      stride2 = strideptr[2];
 *          npy_intp count = *countptr;
 *
 *          while (count--) {
 *              if (NpyMaskValue_IsExposed((npy_mask)*data2)) {
 *                  *(result_t *)data0 = my_reduce_op(*(result_t *)data0,
 *                                                    *(operand_t *)data1);
 *              }
 *              data0 += stride0;
 *              data1 += stride1;
 *              data2 += stride2;
 *          }
 *      } while (iternext(iter));
 *      if (!needs_api) {
 *          NPY_END_THREADS;
 *      }
 *
 * Once the inner loop is finished, PyArray_ReduceWrapper calls
 * PyErr_Occurred to check if any exception was raised during the
 * computation.
 */
typedef void (PyArray_ReduceInnerLoopFunc)(NpyIter *iter,
                                            char **dataptr,
                                            npy_intp *strideptr,
                                            npy_intp *countptr,
                                            NpyIter_IterNextFunc *iternext,
                                            int needs_api,
                                            void *data);

/*
 * This function executes all the standard NumPy reduction function
 * boilerplate code, just calling assign_unit and the appropriate
 * inner loop function where necessary.
 *
 * operand     : The array to be reduced.
 * out         : NULL, or the array into which to place the result.
 * operand_dtype : The dtype the inner loop expects for the operand.
 * result_dtype : The dtype the inner loop expects for the result.
 * axis_flags  : Flags indicating the reduction axes of 'operand'.
 * skipna      : If true, NAs are skipped instead of propagating.
 * keepdims    : If true, leaves the reduction dimensions in the result
 *               with size one.
 * assign_unit : If NULL, PyArray_InitializeReduceResult is used, otherwise
 *               this function is called to initialize the result to
 *               the reduction's unit.
 * inner_loop  : The inner loop which does the reduction.
 * masked_inner_loop: The inner loop which does the reduction with a mask.
 * data        : Data which is passed to assign_unit and the inner loop.
 * buffersize  : Buffer size for the iterator. For the default, pass in 0.
 * funcname    : The name of the reduction function, for error messages.
 */
NPY_NO_EXPORT PyArrayObject *
PyArray_ReduceWrapper(PyArrayObject *operand, PyArrayObject *out,
                        PyArray_Descr *operand_dtype,
                        PyArray_Descr *result_dtype,
                        npy_bool *axis_flags, int skipna, int keepdims,
                        PyArray_AssignReduceUnitFunc *assign_unit,
                        PyArray_ReduceInnerLoopFunc *inner_loop,
                        PyArray_ReduceInnerLoopFunc *masked_inner_loop,
                        void *data, npy_intp buffersize, const char *funcname);

/*
 * This function counts the number of elements that a reduction
 * will see along the reduction directions, given the provided options.
 *
 * If the reduction operand has no NA mask or 'skipna' is false, this
 * is simply the prod`uct of all the reduction axis sizes. A NumPy
 * scalar is returned in this case.
 *
 * If the reduction operand has an NA mask and 'skipna' is true, this
 * counts the number of elements which are not NA along the reduction
 * dimensions, and returns an array with the counts.
 */
NPY_NO_EXPORT PyObject *
PyArray_CountReduceItems(PyArrayObject *operand,
                            npy_bool *axis_flags, int skipna, int keepdims);


#endif
