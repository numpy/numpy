#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <locale.h>
#include <stdio.h>

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "npy_config.h"

/* Functions only needed on narrow builds of Python
   for converting back and forth between the NumPy Unicode data-type
   (always 4-byte)
   and the Python Unicode scalar (2-bytes on a narrow build).
*/

/* the ucs2 buffer must be large enough to hold 2*ucs4length characters
   due to the use of surrogate pairs.

   The return value is the number of ucs2 bytes used-up which
   is ucs4length + number of surrogate pairs found.

   values above 0xffff are converted to surrogate pairs.
*/
NPY_NO_EXPORT int
PyUCS2Buffer_FromUCS4(Py_UNICODE *ucs2, PyArray_UCS4 *ucs4, int ucs4length)
{
    int i;
    int numucs2 = 0;
    PyArray_UCS4 chr;
    for (i=0; i<ucs4length; i++) {
        chr = *ucs4++;
        if (chr > 0xffff) {
            numucs2++;
            chr -= 0x10000L;
            *ucs2++ = 0xD800 + (Py_UNICODE) (chr >> 10);
            *ucs2++ = 0xDC00 + (Py_UNICODE) (chr & 0x03FF);
        }
        else {
            *ucs2++ = (Py_UNICODE) chr;
        }
        numucs2++;
    }
    return numucs2;
}


/* This converts a UCS2 buffer of the given length to UCS4 buffer.
   It converts up to ucs4len characters of UCS2

   It returns the number of characters converted which can
   be less than ucs2len if there are surrogate pairs in ucs2.

   The return value is the actual size of the used part of the ucs4 buffer.
*/

NPY_NO_EXPORT int
PyUCS2Buffer_AsUCS4(Py_UNICODE *ucs2, PyArray_UCS4 *ucs4, int ucs2len, int ucs4len)
{
    int i;
    PyArray_UCS4 chr;
    Py_UNICODE ch;
    int numchars=0;

    for (i=0; (i < ucs2len) && (numchars < ucs4len); i++) {
        ch = *ucs2++;
        if (ch >= 0xd800 && ch <= 0xdfff) {
            /* surrogate pair */
            chr = ((PyArray_UCS4)(ch-0xd800)) << 10;
            chr += *ucs2++ + 0x2400;  /* -0xdc00 + 0x10000 */
            i++;
        }
        else {
            chr = (PyArray_UCS4) ch;
        }
        *ucs4++ = chr;
        numchars++;
    }
    return numchars;
}


NPY_NO_EXPORT PyObject *
MyPyUnicode_New(int length)
{
    PyUnicodeObject *unicode;
    unicode = PyObject_New(PyUnicodeObject, &PyUnicode_Type);
    if (unicode == NULL) return NULL;
    unicode->str = PyMem_NEW(Py_UNICODE, length+1);
    if (!unicode->str) {
        _Py_ForgetReference((PyObject *)unicode);
        PyObject_Del(unicode);
        return PyErr_NoMemory();
    }
    unicode->str[0] = 0;
    unicode->str[length] = 0;
    unicode->length = length;
    unicode->hash = -1;
    unicode->defenc = NULL;
    return (PyObject *)unicode;
}

NPY_NO_EXPORT int
MyPyUnicode_Resize(PyUnicodeObject *uni, int length)
{
    void *oldstr;

    oldstr = uni->str;
    PyMem_RESIZE(uni->str, Py_UNICODE, length+1);
    if (!uni->str) {
        uni->str = oldstr;
        PyErr_NoMemory();
        return -1;
    }
    uni->str[length] = 0;
    uni->length = length;
    return 0;
}
