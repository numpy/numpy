#ifndef _NPY_UCSNARROW_H_
#define _NPY_UCSNARROW_H_

NPY_NO_EXPORT PyUnicodeObject *
PyUnicode_FromUCS4(char *src, Py_ssize_t size, int swap, int align);

#endif
