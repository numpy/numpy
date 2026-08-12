#ifndef _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_
#define _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_


#ifdef __cplusplus
extern "C" {
#endif

PyArrayMethod_Spec **get_casts();

// Whether obj is a NaN missing value: a float NaN or a complex value with a
// NaN component (np.isnan semantics). Returns 1 if it is, 0 if not, and -1 if
// the ComplexWarning raised for complex values was turned into an error. Used
// by stringdtype_setitem; kept consistent with float_is_nan_na in the
// float-to-string casts.
NPY_NO_EXPORT int
pyobj_is_nan_na(PyObject *obj);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_ */
