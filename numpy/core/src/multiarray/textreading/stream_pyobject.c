/*
 * C side structures to provide capabilities to read Python file like objects
 * in chunks, or iterate through iterables with each result representing a
 * single line of a file.
 */

#include <stdio.h>
#include <stdlib.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"

#include "textreading/stream.h"

#define READ_CHUNKSIZE 1 << 14


typedef struct {
    /* The Python file object being read. */
    PyObject *file;

    /* The `read` attribute of the file object. */
    PyObject *read;
    /* Amount to read each time we call `obj.read()` */
    PyObject *chunksize;

    /* file position when the file_buffer was created. */
    off_t initial_file_pos;

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
static NPY_INLINE PyObject *
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


static NPY_INLINE void
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
    python_chunks_from_file *fb = (python_chunks_from_file *)strm->stream_data;

    Py_XDECREF(fb->file);
    Py_XDECREF(fb->read);
    Py_XDECREF(fb->chunksize);
    Py_XDECREF(fb->chunk);

    free(fb);
    free(strm);

    return 0;
}


stream *
stream_python_file(PyObject *obj, const char *encoding)
{
    python_chunks_from_file *fb;
    stream *strm;

    fb = (python_chunks_from_file *) malloc(sizeof(python_chunks_from_file));
    if (fb == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    fb->file = NULL;
    fb->read = NULL;
    fb->chunksize = NULL;
    fb->chunk = NULL;
    fb->encoding = encoding;

    strm = (stream *) malloc(sizeof(stream));
    if (strm == NULL) {
        PyErr_NoMemory();
        free(fb);
        return NULL;
    }

    fb->file = obj;
    Py_INCREF(fb->file);

    fb->read = PyObject_GetAttrString(obj, "read");
    if (fb->read == NULL) {
        goto fail;
    }
    fb->chunksize = PyLong_FromLong(READ_CHUNKSIZE);
    if (fb->chunksize == NULL) {
        goto fail;
    }

    strm->stream_data = (void *)fb;
    strm->stream_nextbuf = (void *)&fb_nextbuf;
    strm->stream_close = &fb_del;

    return strm;

fail:
    fb_del(strm);
    return NULL;
}


/*
 * Stream from a Python iterable by interpreting each item as a line in a file
 */
typedef struct {
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
    python_lines_from_iterator *it = (python_lines_from_iterator *)strm->stream_data;

    Py_XDECREF(it->iterator);
    Py_XDECREF(it->line);

    free(it);
    free(strm);

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


stream *
stream_python_iterable(PyObject *obj, const char *encoding)
{
    python_lines_from_iterator *it;
    stream *strm;

    it = (python_lines_from_iterator *)malloc(sizeof(*it));
    if (it == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    it->iterator = NULL;
    it->line = NULL;
    it->encoding = encoding;

    strm = (stream *) malloc(sizeof(stream));
    if (strm == NULL) {
        PyErr_NoMemory();
        free(it);
        return NULL;
    }
    if (!PyIter_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                "error reading from object, expected an iterable.");
        goto fail;
    }
    Py_INCREF(obj);
    it->iterator = obj;

    strm->stream_data = (void *)it;
    strm->stream_nextbuf = (void *)&it_nextbuf;
    strm->stream_close = &it_del;

    return strm;

fail:
    it_del(strm);
    return NULL;
}
