#ifndef _NPY_UCSNARROW_H_
#define _NPY_UCSNARROW_H_

NPY_NO_EXPORT int
PyUCS2Buffer_FromUCS4(Py_UNICODE *ucs2, npy_ucs4 *ucs4, int ucs4length);

NPY_NO_EXPORT int
PyUCS2Buffer_AsUCS4(Py_UNICODE *ucs2, npy_ucs4 *ucs4, int ucs2len, int ucs4len);

NPY_NO_EXPORT PyObject *
MyPyUnicode_New(int length);

NPY_NO_EXPORT int
MyPyUnicode_Resize(PyUnicodeObject *uni, int length);

#endif
