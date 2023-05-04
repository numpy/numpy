#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "npy_config.h"

#include "npy_pycompat.h"
#include "ctors.h"

/*
 * This file originally contained functions only needed on narrow builds of
 * Python for converting back and forth between the NumPy Unicode data-type
 * (always 4-bytes) and the Python Unicode scalar (2-bytes on a narrow build).
 *
 * This "narrow" interface is now deprecated in python and unused in NumPy.
 */

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
PyUnicode_FromUCS4(char const *src_char, Py_ssize_t size, int swap, int align)
{
    Py_ssize_t ucs4len = size / sizeof(npy_ucs4);
    npy_ucs4 const *src = (npy_ucs4 const *)src_char;
    npy_ucs4 *buf = NULL;

    /* swap and align if needed */
    if (swap || align) {
        buf = (npy_ucs4 *)malloc(size);
        if (buf == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(buf, src, size);
        if (swap) {
            byte_swap_vector(buf, ucs4len, sizeof(npy_ucs4));
        }
        src = buf;
    }

    /* trim trailing zeros */
    while (ucs4len > 0 && src[ucs4len - 1] == 0) {
        ucs4len--;
    }
    PyUnicodeObject *ret = (PyUnicodeObject *)PyUnicode_FromKindAndData(
        PyUnicode_4BYTE_KIND, src, ucs4len);
    free(buf);
    return ret;
}
