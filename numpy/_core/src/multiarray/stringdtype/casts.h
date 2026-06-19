#ifndef _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_
#define _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_


#ifdef __cplusplus
extern "C" {
#endif

PyArrayMethod_Spec **get_casts();

// Returns 1 if obj is a real (non-complex) floating point NaN, else 0. Shared
// by the float-to-string cast and stringdtype_setitem so both treat NaN as the
// missing value identically.
NPY_NO_EXPORT int
pyobj_is_nan_na(PyObject *obj);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_ */
