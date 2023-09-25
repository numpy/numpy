#define STRINGLIB_LENGTH_H

static inline Py_ssize_t
STRINGLIB(get_length)(PyArrayIterObject *iter)
{
    STRINGLIB_CHAR *buffer = (STRINGLIB_CHAR *) iter->dataptr;
    STRINGLIB_CHAR *data = buffer + PyArray_ITEMSIZE(iter->ao) / sizeof(STRINGLIB_CHAR) - 1;
    while (data >= buffer && *data == '\0') {
        data--;
    }
    return data - buffer + 1;
}
