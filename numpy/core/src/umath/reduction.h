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
 * This is a function for the reduce loop.
 *
 * The needs_api parameter indicates whether it's ok to release the GIL during
 * the loop, such as when the iternext() function never calls
 * a function which could raise a Python exception.
 *
 * The skip_first_count parameter indicates how many elements need to be
 * skipped based on NpyIter_IsFirstVisit checks. This can only be positive
 * when the 'assign_identity' parameter was NULL when calling
 * PyArray_ReduceWrapper.
 *
 * The loop gets two data pointers and two strides, and should
 * look roughly like this:
 *  {
 *      NPY_BEGIN_THREADS_DEF;
 *      if (!needs_api) {
 *          NPY_BEGIN_THREADS;
 *      }
 *      // This first-visit loop can be skipped if 'assign_identity' was non-NULL
 *      if (skip_first_count > 0) {
 *          do {
 *              char *data0 = dataptr[0], *data1 = dataptr[1];
 *              npy_intp stride0 = strideptr[0], stride1 = strideptr[1];
 *              npy_intp count = *countptr;
 *
 *              // Skip any first-visit elements
 *              if (NpyIter_IsFirstVisit(iter, 0)) {
 *                  if (stride0 == 0) {
 *                      --count;
 *                      --skip_first_count;
 *                      data1 += stride1;
 *                  }
 *                  else {
 *                      skip_first_count -= count;
 *                      count = 0;
 *                  }
 *              }
 *
 *              while (count--) {
 *                  *(result_t *)data0 = my_reduce_op(*(result_t *)data0,
 *                                                    *(operand_t *)data1);
 *                  data0 += stride0;
 *                  data1 += stride1;
 *              }
 *
 *              // Jump to the faster loop when skipping is done
 *              if (skip_first_count == 0) {
 *                  if (iternext(iter)) {
 *                      break;
 *                  }
 *                  else {
 *                      goto finish_loop;
 *                  }
 *              }
 *          } while (iternext(iter));
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
 *  finish_loop:
 *      if (!needs_api) {
 *          NPY_END_THREADS;
 *      }
 *      return (needs_api && PyErr_Occurred()) ? -1 : 0;
 *  }
 *
 * If needs_api is True, this function should call PyErr_Occurred()
 * to check if an error occurred during processing, and return -1 for
 * error, 0 for success.
 */
typedef int (PyArray_ReduceLoopFunc)(NpyIter *iter,
                                            char **dataptr,
                                            npy_intp *strideptr,
                                            npy_intp *countptr,
                                            NpyIter_IterNextFunc *iternext,
                                            int needs_api,
                                            npy_intp skip_first_count,
                                            void *data);

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
 * subok       : If true, the result uses the subclass of operand, otherwise
 *               it is always a base class ndarray.
 * identity    : If Py_None, PyArray_InitializeReduceResult is used, otherwise
 *               this value is used to initialize the result to
 *               the reduction's unit.
 * loop        : The loop which does the reduction.
 * data        : Data which is passed to the inner loop.
 * buffersize  : Buffer size for the iterator. For the default, pass in 0.
 * funcname    : The name of the reduction function, for error messages.
 * errormask   : forwarded from _get_bufsize_errmask
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
                      PyObject *identity,
                      PyArray_ReduceLoopFunc *loop,
                      void *data, npy_intp buffersize, const char *funcname,
                      int errormask);

#endif
