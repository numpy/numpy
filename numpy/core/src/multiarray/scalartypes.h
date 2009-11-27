#ifndef _NPY_SCALARTYPES_H_
#define _NPY_SCALARTYPES_H_

NPY_NO_EXPORT void
initialize_numeric_types(void);

NPY_NO_EXPORT void
format_longdouble(char *buf, size_t buflen, longdouble val, unsigned int prec);

NPY_NO_EXPORT void
gentype_struct_free(void *ptr, void *arg);

NPY_NO_EXPORT int
_typenum_fromtypeobj(PyObject *type, int user);

NPY_NO_EXPORT void *
scalar_value(PyObject *scalar, PyArray_Descr *descr);

/*
 * XXX: those are defined here for 1.4.x only -> they are intended to become
 * public types, and are declared like all other types through the code
 * generator machinery for >= 1.5.0
 */
#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PyTypeObject PyTimeIntegerArrType_Type;
extern NPY_NO_EXPORT PyTypeObject PyDatetimeArrType_Type;
extern NPY_NO_EXPORT PyTypeObject PyTimedeltaArrType_Type;
#else
NPY_NO_EXPORT PyTypeObject PyTimeIntegerArrType_Type;
NPY_NO_EXPORT PyTypeObject PyDatetimeArrType_Type;
NPY_NO_EXPORT PyTypeObject PyTimedeltaArrType_Type;
#endif

#endif
