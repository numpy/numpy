/*
 * C side structures to provide capabilities to read Python file like objects
 * in chunks, or iterate through iterables with each result representing a
 * single line of a file.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"

#include "textreading/stream.h"

#define READ_CHUNKSIZE 1 << 14


typedef struct {
    stream stream;
    /* The Python file object being read. */
    PyObject *file;

    /* The `read` attribute of the file object. */
    PyObject *read;
    /* Amount to read each time we call `obj.read()` */
    PyObject *chunksize;

    /* Python str object holding the line most recently read from the file. */
    PyObject *chunk;

    /* Encoding compatible with Python's `PyUnicode_Encode` (may be NULL) */
    const char *encoding;
} python_chunks_from_file;


/*
 * Helper function to support byte objects as well as unicode strings.
 *
 * NOTE: Steals a reference to `str` (although usually returns it unmodified).
 */
static inline PyObject *
process_stringlike(PyObject *str, const char *encoding)
{
    if (PyBytes_Check(str)) {
        PyObject *ustr;
        ustr = PyUnicode_FromEncodedObject(str, encoding, NULL);
        if (ustr == NULL) {
            return NULL;
        }
        Py_DECREF(str);
        return ustr;
    }
    else if (!PyUnicode_Check(str)) {
        PyErr_SetString(PyExc_TypeError,
                "non-string returned while reading data");
        Py_DECREF(str);
        return NULL;
    }
    return str;
}


static inline void
buffer_info_from_unicode(PyObject *str, char **start, char **end, int *kind)
{
    Py_ssize_t length = PyUnicode_GET_LENGTH(str);
    *kind = PyUnicode_KIND(str);

    if (*kind == PyUnicode_1BYTE_KIND) {
        *start = (char *)PyUnicode_1BYTE_DATA(str);
    }
    else if (*kind == PyUnicode_2BYTE_KIND) {
        *start = (char *)PyUnicode_2BYTE_DATA(str);
        length *= sizeof(Py_UCS2);
    }
    else if (*kind == PyUnicode_4BYTE_KIND) {
        *start = (char *)PyUnicode_4BYTE_DATA(str);
        length *= sizeof(Py_UCS4);
    }
    *end = *start + length;
}


static int
fb_nextbuf(python_chunks_from_file *fb, char **start, char **end, int *kind)
{
    Py_XDECREF(fb->chunk);
    fb->chunk = NULL;

    PyObject *chunk = PyObject_CallFunctionObjArgs(fb->read, fb->chunksize, NULL);
    if (chunk == NULL) {
        return -1;
    }
    fb->chunk = process_stringlike(chunk, fb->encoding);
    if (fb->chunk == NULL) {
        return -1;
    }
    buffer_info_from_unicode(fb->chunk, start, end, kind);
    if (*start == *end) {
        return BUFFER_IS_FILEEND;
    }
    return BUFFER_MAY_CONTAIN_NEWLINE;
}


static int
fb_del(stream *strm)
{
    python_chunks_from_file *fb = (python_chunks_from_file *)strm;

    Py_XDECREF(fb->file);
    Py_XDECREF(fb->read);
    Py_XDECREF(fb->chunksize);
    Py_XDECREF(fb->chunk);

    PyMem_FREE(strm);

    return 0;
}


NPY_NO_EXPORT stream *
stream_python_file(PyObject *obj, const char *encoding)
{
    python_chunks_from_file *fb;

    fb = (python_chunks_from_file *)PyMem_Calloc(1, sizeof(python_chunks_from_file));
    if (fb == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    fb->stream.stream_nextbuf = (void *)&fb_nextbuf;
    fb->stream.stream_close = &fb_del;

    fb->encoding = encoding;
    Py_INCREF(obj);
    fb->file = obj;

    fb->read = PyObject_GetAttrString(obj, "read");
    if (fb->read == NULL) {
        goto fail;
    }
    fb->chunksize = PyLong_FromLong(READ_CHUNKSIZE);
    if (fb->chunksize == NULL) {
        goto fail;
    }

    return (stream *)fb;

fail:
    fb_del((stream *)fb);
    return NULL;
}


/*
 * Stream from a Python iterable by interpreting each item as a line in a file
 */
typedef struct {
    stream stream;
    /* The Python file object being read. */
    PyObject *iterator;

    /* Python str object holding the line most recently fetched */
    PyObject *line;

    /* Encoding compatible with Python's `PyUnicode_Encode` (may be NULL) */
    const char *encoding;
} python_lines_from_iterator;


static int
it_del(stream *strm)
{
    python_lines_from_iterator *it = (python_lines_from_iterator *)strm;

    Py_XDECREF(it->iterator);
    Py_XDECREF(it->line);

    PyMem_FREE(strm);
    return 0;
}


static int
it_nextbuf(python_lines_from_iterator *it, char **start, char **end, int *kind)
{
    Py_XDECREF(it->line);
    it->line = NULL;

    PyObject *line = PyIter_Next(it->iterator);
    if (line == NULL) {
        if (PyErr_Occurred()) {
            return -1;
        }
        *start = NULL;
        *end = NULL;
        return BUFFER_IS_FILEEND;
    }
    it->line = process_stringlike(line, it->encoding);
    if (it->line == NULL) {
        return -1;
    }

    buffer_info_from_unicode(it->line, start, end, kind);
    return BUFFER_IS_LINEND;
}


NPY_NO_EXPORT stream *
stream_python_iterable(PyObject *obj, const char *encoding)
{
    python_lines_from_iterator *it;

    if (!PyIter_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                "error reading from object, expected an iterable.");
        return NULL;
    }

    it = (python_lines_from_iterator *)PyMem_Calloc(1, sizeof(*it));
    if (it == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    it->stream.stream_nextbuf = (void *)&it_nextbuf;
    it->stream.stream_close = &it_del;

    it->encoding = encoding;
    Py_INCREF(obj);
    it->iterator = obj;

    return (stream *)it;
}
