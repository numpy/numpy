#ifndef _NPY_SCALARTYPES_H_
#define _NPY_SCALARTYPES_H_

/* Internal look-up tables */
#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT unsigned char
_npy_can_cast_safely_table[NPY_NTYPES][NPY_NTYPES];
extern NPY_NO_EXPORT signed char
_npy_scalar_kinds_table[NPY_NTYPES];
extern NPY_NO_EXPORT signed char
_npy_type_promotion_table[NPY_NTYPES][NPY_NTYPES];
extern NPY_NO_EXPORT signed char
_npy_smallest_type_of_kind_table[NPY_NSCALARKINDS];
extern NPY_NO_EXPORT signed char
_npy_next_larger_type_table[NPY_NTYPES];
#else
NPY_NO_EXPORT unsigned char
_npy_can_cast_safely_table[NPY_NTYPES][NPY_NTYPES];
NPY_NO_EXPORT signed char
_npy_scalar_kinds_table[NPY_NTYPES];
NPY_NO_EXPORT signed char
_npy_type_promotion_table[NPY_NTYPES][NPY_NTYPES];
NPY_NO_EXPORT signed char
_npy_smallest_type_of_kind_table[NPY_NSCALARKINDS];
NPY_NO_EXPORT signed char
_npy_next_larger_type_table[NPY_NTYPES];
#endif

NPY_NO_EXPORT void
initialize_casting_tables(void);

NPY_NO_EXPORT void
initialize_numeric_types(void);

NPY_NO_EXPORT void
format_longdouble(char *buf, size_t buflen, npy_longdouble val, unsigned int prec);

#if PY_VERSION_HEX >= 0x03000000
NPY_NO_EXPORT void
gentype_struct_free(PyObject *ptr);
#else
NPY_NO_EXPORT void
gentype_struct_free(void *ptr, void *arg);
#endif

NPY_NO_EXPORT int
_typenum_fromtypeobj(PyObject *type, int user);

NPY_NO_EXPORT void *
scalar_value(PyObject *scalar, PyArray_Descr *descr);

#endif
