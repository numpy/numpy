#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_GROWTH_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_GROWTH_H_

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT npy_intp
grow_size_and_multiply(npy_intp *size, npy_intp min_grow, npy_intp itemsize);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_GROWTH_H_ */
