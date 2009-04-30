#ifndef _NPY_UCSNARROW_H_
#define _NPY_UCSNARROW_H_

#ifdef Py_UNICODE_WIDE
#error this should not be included if Py_UNICODE_WIDE is defined
int int int;
#endif

NPY_NO_EXPORT int
PyUCS2Buffer_FromUCS4(Py_UNICODE *ucs2, PyArray_UCS4 *ucs4, int ucs4length);

NPY_NO_EXPORT int
PyUCS2Buffer_AsUCS4(Py_UNICODE *ucs2, PyArray_UCS4 *ucs4, int ucs2len, int ucs4len);

NPY_NO_EXPORT PyObject *
MyPyUnicode_New(int length);

NPY_NO_EXPORT int
MyPyUnicode_Resize(PyUnicodeObject *uni, int length);

#endif
