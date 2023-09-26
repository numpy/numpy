#ifndef STRINGLIB_LENGTH_H
#error must include "stringlib/length.h" before including this module
#endif

static inline int
STRINGLIB(add)(PyArrayIterObject **in_iters, PyArrayIterObject *out_iter)
{
    STRINGLIB_CHAR *left, *right;
    Py_ssize_t left_len, right_len, left_len_in_bytes, right_len_in_bytes;

    left = (STRINGLIB_CHAR *) in_iters[0]->dataptr;
    right = (STRINGLIB_CHAR *) in_iters[1]->dataptr;

    left_len = STRINGLIB(get_length)(in_iters[0]);
    left_len_in_bytes = left_len * sizeof(STRINGLIB_CHAR);

    right_len = STRINGLIB(get_length)(in_iters[1]);
    right_len_in_bytes = right_len * sizeof(STRINGLIB_CHAR);

    if (left_len > PY_SSIZE_T_MAX - right_len) {
        PyErr_SetString(PyExc_OverflowError,
                        "strings are too large to concat");
        return 0;
    }

    if (left_len == 0) {
        memcpy(out_iter->dataptr, right, right_len_in_bytes);
        memset(out_iter->dataptr + right_len_in_bytes, 0, PyArray_ITEMSIZE(out_iter->ao) - right_len_in_bytes);
    }
    if (right_len == 0) {
        memcpy(out_iter->dataptr, left, left_len_in_bytes);
        memset(out_iter->dataptr + left_len_in_bytes, 0, PyArray_ITEMSIZE(out_iter->ao) - left_len_in_bytes);
    }

    memcpy(out_iter->dataptr, left, left_len_in_bytes);
    memcpy(out_iter->dataptr + left_len_in_bytes, right, right_len_in_bytes);
    memset(out_iter->dataptr + left_len_in_bytes + right_len_in_bytes, 0,
           PyArray_ITEMSIZE(out_iter->ao) - right_len_in_bytes - left_len_in_bytes);
    return 1;
}
