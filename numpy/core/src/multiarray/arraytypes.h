#ifndef _NPY_ARRAYTYPES_H_
#define _NPY_ARRAYTYPES_H_

extern NPY_NO_EXPORT PyArray_Descr LONG_Descr;
extern NPY_NO_EXPORT PyArray_Descr INT_Descr;

NPY_NO_EXPORT int
set_typeinfo(PyObject *dict);

#endif
