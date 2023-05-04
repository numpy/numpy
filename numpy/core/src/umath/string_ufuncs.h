#ifndef _NPY_CORE_SRC_UMATH_STRING_UFUNCS_H_
#define _NPY_CORE_SRC_UMATH_STRING_UFUNCS_H_

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT int
init_string_ufuncs(PyObject *umath);

NPY_NO_EXPORT PyObject *
_umath_strings_richcompare(
        PyArrayObject *self, PyArrayObject *other, int cmp_op, int rstrip);

#ifdef __cplusplus
}
#endif

#endif  /* _NPY_CORE_SRC_UMATH_STRING_UFUNCS_H_ */