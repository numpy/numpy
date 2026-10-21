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
    /* UCS4 copy of `chunk`, if it is not ASCII */
    Py_UCS4 *chunk_ucs4;

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


/*
 * ASCII text is read in place, as its UTF-8 has one byte per character.  Other
 * text is read from a UCS4 copy, which the stream owns through `*ucs4`.
 */
static inline int
buffer_info_from_unicode(PyObject *str, Py_UCS4 **ucs4,
        char **start, char **end, int *kind)
{
    Py_ssize_t length = PyUnicode_GetLength(str);
    Py_ssize_t size;
    const char *utf8 = PyUnicode_AsUTF8AndSize(str, &size);
    if (utf8 != NULL && size == length) {
        *kind = (int)sizeof(Py_UCS1);
        *start = (char *)utf8;
        *end = *start + length;
        return 0;
    }
    PyErr_Clear();

    PyMem_Free(*ucs4);
    *ucs4 = PyUnicode_AsUCS4Copy(str);
    if (*ucs4 == NULL) {
        return -1;
    }
    *kind = (int)sizeof(Py_UCS4);
    *start = (char *)*ucs4;
    *end = *start + length * sizeof(Py_UCS4);
    return 0;
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
    if (buffer_info_from_unicode(
            fb->chunk, &fb->chunk_ucs4, start, end, kind) < 0) {
        return -1;
    }
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
    PyMem_Free(fb->chunk_ucs4);

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
    /* UCS4 copy of `line`, if it is not ASCII */
    Py_UCS4 *line_ucs4;

    /* Encoding compatible with Python's `PyUnicode_Encode` (may be NULL) */
    const char *encoding;
} python_lines_from_iterator;


static int
it_del(stream *strm)
{
    python_lines_from_iterator *it = (python_lines_from_iterator *)strm;

    Py_XDECREF(it->iterator);
    Py_XDECREF(it->line);
    PyMem_Free(it->line_ucs4);

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

    if (buffer_info_from_unicode(
            it->line, &it->line_ucs4, start, end, kind) < 0) {
        return -1;
    }
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
