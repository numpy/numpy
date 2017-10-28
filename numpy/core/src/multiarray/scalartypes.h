#ifndef _NPY_SCALARTYPES_H_
#define _NPY_SCALARTYPES_H_

/* Internal look-up tables */
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

NPY_NO_EXPORT void
initialize_casting_tables(void);

NPY_NO_EXPORT void
initialize_numeric_types(void);

#if PY_VERSION_HEX >= 0x03000000
NPY_NO_EXPORT void
gentype_struct_free(PyObject *ptr);
#else
NPY_NO_EXPORT void
gentype_struct_free(void *ptr, void *arg);
#endif

NPY_NO_EXPORT int
is_anyscalar_exact(PyObject *obj);

NPY_NO_EXPORT int
_typenum_fromtypeobj(PyObject *type, int user);

NPY_NO_EXPORT void *
scalar_value(PyObject *scalar, PyArray_Descr *descr);

#endif
