#ifndef _NPY_UCSNARROW_H_
#define _NPY_UCSNARROW_H_

NPY_NO_EXPORT int
PyUCS2Buffer_FromUCS4(Py_UNICODE *ucs2, npy_ucs4 *ucs4, int ucs4length);

NPY_NO_EXPORT int
PyUCS2Buffer_AsUCS4(Py_UNICODE *ucs2, npy_ucs4 *ucs4, int ucs2len, int ucs4len);

NPY_NO_EXPORT PyUnicodeObject *
PyUnicode_FromUCS4(char *src, Py_ssize_t size, int swap, int align);

#endif
