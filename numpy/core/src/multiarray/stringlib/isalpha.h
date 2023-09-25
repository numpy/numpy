#ifndef STRINGLIB_LENGTH_H
#error must include "stringlib/length.h" before including this module
#endif

static inline int
STRINGLIB(isalpha)(PyArrayIterObject **in_iter, PyArrayIterObject *out_iter)
{
    STRINGLIB_CHAR *data = (STRINGLIB_CHAR *) (*in_iter)->dataptr;
    Py_ssize_t len = STRINGLIB(get_length)(*in_iter);

    if (len == 0) {
        *out_iter->dataptr = (npy_bool) NPY_FALSE;
        return 1;
    }

    for (Py_ssize_t i = 0; i < len; i++) {
        if (!STRINGLIB_CHAR_ISALPHA(data[i])) {
            *out_iter->dataptr = (npy_bool) NPY_FALSE;
            return 1;
        }
    }
    *out_iter->dataptr = (npy_bool) NPY_TRUE;
    return 1;
}
