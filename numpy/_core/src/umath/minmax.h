#ifndef _NPY_CORE_SRC_UMATH_MINMAX_H_
#define _NPY_CORE_SRC_UMATH_MINMAX_H_

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT int
init_minimummaximum(PyObject *umath);

/*
 * Returns the loop that folds the minimum and the maximum of integers of
 * type `type_num` into two values, or NULL if `type_num` is not a 32 or 64
 * bit integer. To fold `count` integers into `*min` and `*max`, which hold
 * the values found so far, call it as
 *
 *     char *const args[5] = {min, max, data, min, max};
 *     const npy_intp strides[5] = {0, 0, stride, 0, 0};
 *     loop(NULL, args, &count, strides, NULL);
 *
 * The loop needs aligned data and no GIL.
 */
NPY_NO_EXPORT PyArrayMethod_StridedLoop *
npy_get_integer_minimummaximum_loop(int type_num);

#ifdef __cplusplus
}
#endif

#endif  /* _NPY_CORE_SRC_UMATH_MINMAX_H_ */
