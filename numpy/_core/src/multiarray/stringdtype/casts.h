#ifndef _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_
#define _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_


#ifdef __cplusplus
extern "C" {
#endif

PyArrayMethod_Spec **get_casts();

// Whether obj is a floating point NaN missing value: a real float NaN, or a
// complex value with a NaN real part (any imaginary part). Returns 1 if it is,
// 0 if not, and -1 if coercing a complex value raised a ComplexWarning as an
// error (its imaginary part is discarded). Shared by the float-to-string cast
// and stringdtype_setitem so both treat NaN as the missing value identically.
NPY_NO_EXPORT int
pyobj_is_nan_na(PyObject *obj);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_ */
