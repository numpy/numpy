#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <locale.h>
#include <stdio.h>

#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "npy_config.h"

#include "npy_pycompat.h"
#include "ctors.h"

/*
 * Functions only needed on narrow builds of Python for converting back and
 * forth between the NumPy Unicode data-type (always 4-bytes) and the
 * Python Unicode scalar (2-bytes on a narrow build).
 */

/*
 * The ucs2 buffer must be large enough to hold 2*ucs4length characters
 * due to the use of surrogate pairs.
 *
 * The return value is the number of ucs2 bytes used-up which
 * is ucs4length + number of surrogate pairs found.
 *
 * Values above 0xffff are converted to surrogate pairs.
 */
NPY_NO_EXPORT int
PyUCS2Buffer_FromUCS4(Py_UNICODE *ucs2, npy_ucs4 *ucs4, int ucs4length)
{
    int i;
    int numucs2 = 0;
    npy_ucs4 chr;
    for (i = 0; i < ucs4length; i++) {
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


/*
 * This converts a UCS2 buffer of the given length to UCS4 buffer.
 * It converts up to ucs4len characters of UCS2
 *
 * It returns the number of characters converted which can
 * be less than ucs2len if there are surrogate pairs in ucs2.
 *
 * The return value is the actual size of the used part of the ucs4 buffer.
 */
NPY_NO_EXPORT int
PyUCS2Buffer_AsUCS4(Py_UNICODE *ucs2, npy_ucs4 *ucs4, int ucs2len, int ucs4len)
{
    int i;
    npy_ucs4 chr;
    Py_UNICODE ch;
    int numchars=0;

    for (i = 0; (i < ucs2len) && (numchars < ucs4len); i++) {
        ch = *ucs2++;
        if (ch >= 0xd800 && ch <= 0xdfff) {
            /* surrogate pair */
            chr = ((npy_ucs4)(ch-0xd800)) << 10;
            chr += *ucs2++ + 0x2400;  /* -0xdc00 + 0x10000 */
            i++;
        }
        else {
            chr = (npy_ucs4) ch;
        }
        *ucs4++ = chr;
        numchars++;
    }
    return numchars;
}

/*
 * Returns a PyUnicodeObject initialized from a buffer containing
 * UCS4 unicode.
 *
 * Parameters
 * ----------
 *  src: char *
 *      Pointer to buffer containing UCS4 unicode.
 *  size: Py_ssize_t
 *      Size of buffer in bytes.
 *  swap: int
 *      If true, the data will be swapped.
 *  align: int
 *      If true, the data will be aligned.
 *
 * Returns
 * -------
 * new_reference: PyUnicodeObject
 */
NPY_NO_EXPORT PyUnicodeObject *
PyUnicode_FromUCS4(char *src, Py_ssize_t size, int swap, int align)
{
    Py_ssize_t ucs4len = size / sizeof(npy_ucs4);
    npy_ucs4 *buf = (npy_ucs4 *)src;
    int alloc = 0;
    PyUnicodeObject *ret;

    /* swap and align if needed */
    if (swap || align) {
        buf = (npy_ucs4 *)malloc(size);
        if (buf == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        alloc = 1;
        memcpy(buf, src, size);
        if (swap) {
            byte_swap_vector(buf, ucs4len, sizeof(npy_ucs4));
        }
    }

    /* trim trailing zeros */
    while (ucs4len > 0 && buf[ucs4len - 1] == 0) {
        ucs4len--;
    }

    /* produce PyUnicode object */
#ifdef Py_UNICODE_WIDE
    {
        ret = (PyUnicodeObject *)PyUnicode_FromUnicode((Py_UNICODE*)buf,
                                                       (Py_ssize_t) ucs4len);
        if (ret == NULL) {
            goto fail;
        }
    }
#else
    {
        Py_ssize_t tmpsiz = 2 * sizeof(Py_UNICODE) * ucs4len;
        Py_ssize_t ucs2len;
        Py_UNICODE *tmp;

        if ((tmp = (Py_UNICODE *)malloc(tmpsiz)) == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        ucs2len = PyUCS2Buffer_FromUCS4(tmp, buf, ucs4len);
        ret = (PyUnicodeObject *)PyUnicode_FromUnicode(tmp, (Py_ssize_t) ucs2len);
        free(tmp);
        if (ret == NULL) {
            goto fail;
        }
    }
#endif

    if (alloc) {
        free(buf);
    }
    return ret;

fail:
    if (alloc) {
        free(buf);
    }
    return NULL;
}
