#ifndef _NPY_PRIVATE_TYPEINFO_H_
#define _NPY_PRIVATE_TYPEINFO_H_

void typeinfo_init_structsequences(void);

extern PyTypeObject PyArray_typeinfoType;
extern PyTypeObject PyArray_typeinforangedType;

PyObject *
PyArray_typeinfo(
    char typechar, int typenum, int nbits, int align,
    PyTypeObject *type_obj);

PyObject *
PyArray_typeinforanged(
    char typechar, int typenum, int nbits, int align,
    PyObject *max, PyObject *min, PyTypeObject *type_obj);

#endif
