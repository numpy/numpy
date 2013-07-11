#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "buffer.h"
#include "numpyos.h"
#include "arrayobject.h"

/*************************************************************************
 ****************   Implement Buffer Protocol ****************************
 *************************************************************************/

/* removed multiple segment interface */

#if !defined(NPY_PY3K)
static Py_ssize_t
array_getsegcount(PyArrayObject *self, Py_ssize_t *lenp)
{
    if (lenp) {
        *lenp = PyArray_NBYTES(self);
    }
    if (PyArray_ISONESEGMENT(self)) {
        return 1;
    }
    if (lenp) {
        *lenp = 0;
    }
    return 0;
}

static Py_ssize_t
array_getreadbuf(PyArrayObject *self, Py_ssize_t segment, void **ptrptr)
{
    if (segment != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "accessing non-existing array segment");
        return -1;
    }
    if (PyArray_ISONESEGMENT(self)) {
        *ptrptr = PyArray_DATA(self);
        return PyArray_NBYTES(self);
    }
    PyErr_SetString(PyExc_ValueError, "array is not a single segment");
    *ptrptr = NULL;
    return -1;
}


static Py_ssize_t
array_getwritebuf(PyArrayObject *self, Py_ssize_t segment, void **ptrptr)
{
    if (PyArray_FailUnlessWriteable(self, "buffer source array") < 0) {
        return -1;
    }
    return array_getreadbuf(self, segment, (void **) ptrptr);
}

static Py_ssize_t
array_getcharbuf(PyArrayObject *self, Py_ssize_t segment, constchar **ptrptr)
{
    return array_getreadbuf(self, segment, (void **) ptrptr);
}
#endif /* !defined(NPY_PY3K) */


/*************************************************************************
 * PEP 3118 buffer protocol
 *
 * Implementing PEP 3118 is somewhat convoluted because of the desirata:
 *
 * - Don't add new members to ndarray or descr structs, to preserve binary
 *   compatibility. (Also, adding the items is actually not very useful,
 *   since mutability issues prevent an 1 to 1 relationship between arrays
 *   and buffer views.)
 *
 * - Don't use bf_releasebuffer, because it prevents PyArg_ParseTuple("s#", ...
 *   from working. Breaking this would cause several backward compatibility
 *   issues already on Python 2.6.
 *
 * - Behave correctly when array is reshaped in-place, or it's dtype is
 *   altered.
 *
 * The solution taken below is to manually track memory allocated for
 * Py_buffers.
 *************************************************************************/

/*
 * Format string translator
 *
 * Translate PyArray_Descr to a PEP 3118 format string.
 */

/* Fast string 'class' */
typedef struct {
    char *s;
    int allocated;
    int pos;
} _tmp_string_t;

static int
_append_char(_tmp_string_t *s, char c)
{
    char *p;
    if (s->s == NULL) {
        s->s = (char*)malloc(16);
        s->pos = 0;
        s->allocated = 16;
    }
    if (s->pos >= s->allocated) {
        p = (char*)realloc(s->s, 2*s->allocated);
        if (p == NULL) {
            PyErr_SetString(PyExc_MemoryError, "memory allocation failed");
            return -1;
        }
        s->s = p;
        s->allocated *= 2;
    }
    s->s[s->pos] = c;
    ++s->pos;
    return 0;
}

static int
_append_str(_tmp_string_t *s, char *c)
{
    while (*c != '\0') {
        if (_append_char(s, *c)) return -1;
        ++c;
    }
    return 0;
}

/*
 * Return non-zero if a type is aligned in each item in the given array,
 * AND, the descr element size is a multiple of the alignment,
 * AND, the array data is positioned to alignment granularity.
 */
static int
_is_natively_aligned_at(PyArray_Descr *descr,
                        PyArrayObject *arr, Py_ssize_t offset)
{
    int k;

    if ((Py_ssize_t)(PyArray_DATA(arr)) % descr->alignment != 0) {
        return 0;
    }

    if (offset % descr->alignment != 0) {
        return 0;
    }

    if (descr->elsize % descr->alignment) {
        return 0;
    }

    for (k = 0; k < PyArray_NDIM(arr); ++k) {
        if (PyArray_DIM(arr, k) > 1) {
            if (PyArray_STRIDE(arr, k) % descr->alignment != 0) {
                return 0;
            }
        }
    }

    return 1;
}

static int
_buffer_format_string(PyArray_Descr *descr, _tmp_string_t *str,
                      PyArrayObject* arr, Py_ssize_t *offset,
                      char *active_byteorder)
{
    int k;
    char _active_byteorder = '@';
    Py_ssize_t _offset = 0;

    if (active_byteorder == NULL) {
        active_byteorder = &_active_byteorder;
    }
    if (offset == NULL) {
        offset = &_offset;
    }

    if (descr->subarray) {
        PyObject *item, *subarray_tuple;
        Py_ssize_t total_count = 1;
        Py_ssize_t dim_size;
        char buf[128];
        int old_offset;
        int ret;

        if (PyTuple_Check(descr->subarray->shape)) {
            subarray_tuple = descr->subarray->shape;
            Py_INCREF(subarray_tuple);
        }
        else {
            subarray_tuple = Py_BuildValue("(O)", descr->subarray->shape);
        }

        _append_char(str, '(');
        for (k = 0; k < PyTuple_GET_SIZE(subarray_tuple); ++k) {
            if (k > 0) {
                _append_char(str, ',');
            }
            item = PyTuple_GET_ITEM(subarray_tuple, k);
            dim_size = PyNumber_AsSsize_t(item, NULL);

            PyOS_snprintf(buf, sizeof(buf), "%ld", (long)dim_size);
            _append_str(str, buf);
            total_count *= dim_size;
        }
        _append_char(str, ')');

        Py_DECREF(subarray_tuple);

        old_offset = *offset;
        ret = _buffer_format_string(descr->subarray->base, str, arr, offset,
                                    active_byteorder);
        *offset = old_offset + (*offset - old_offset) * total_count;
        return ret;
    }
    else if (PyDataType_HASFIELDS(descr)) {
        int base_offset = *offset;

        _append_str(str, "T{");
        for (k = 0; k < PyTuple_GET_SIZE(descr->names); ++k) {
            PyObject *name, *item, *offset_obj, *tmp;
            PyArray_Descr *child;
            char *p;
            Py_ssize_t len;
            int new_offset;

            name = PyTuple_GET_ITEM(descr->names, k);
            item = PyDict_GetItem(descr->fields, name);

            child = (PyArray_Descr*)PyTuple_GetItem(item, 0);
            offset_obj = PyTuple_GetItem(item, 1);
            new_offset = base_offset + PyInt_AsLong(offset_obj);

            /* Insert padding manually */
            if (*offset > new_offset) {
                PyErr_SetString(PyExc_RuntimeError,
                                "This should never happen: Invalid offset in "
                                "buffer format string generation. Please "
                                "report a bug to the Numpy developers.");
                return -1;
            }
            while (*offset < new_offset) {
                _append_char(str, 'x');
                ++*offset;
            }

            /* Insert child item */
            _buffer_format_string(child, str, arr, offset,
                                  active_byteorder);

            /* Insert field name */
#if defined(NPY_PY3K)
            /* FIXME: XXX -- should it use UTF-8 here? */
            tmp = PyUnicode_AsUTF8String(name);
#else
            tmp = name;
#endif
            if (tmp == NULL || PyBytes_AsStringAndSize(tmp, &p, &len) < 0) {
                PyErr_SetString(PyExc_ValueError, "invalid field name");
                return -1;
            }
            _append_char(str, ':');
            while (len > 0) {
                if (*p == ':') {
                    Py_DECREF(tmp);
                    PyErr_SetString(PyExc_ValueError,
                                    "':' is not an allowed character in buffer "
                                    "field names");
                    return -1;
                }
                _append_char(str, *p);
                ++p;
                --len;
            }
            _append_char(str, ':');
#if defined(NPY_PY3K)
            Py_DECREF(tmp);
#endif
        }
        _append_char(str, '}');
    }
    else {
        int is_standard_size = 1;
        int is_native_only_type = (descr->type_num == NPY_LONGDOUBLE ||
                                   descr->type_num == NPY_CLONGDOUBLE);
#if NPY_SIZEOF_LONG_LONG != 8
        is_native_only_type = is_native_only_type || (
            descr->type_num == NPY_LONGLONG ||
            descr->type_num == NPY_ULONGLONG);
#endif

        *offset += descr->elsize;

        if (descr->byteorder == '=' &&
                _is_natively_aligned_at(descr, arr, *offset)) {
            /* Prefer native types, to cater for Cython */
            is_standard_size = 0;
            if (*active_byteorder != '@') {
                _append_char(str, '@');
                *active_byteorder = '@';
            }
        }
        else if (descr->byteorder == '=' && is_native_only_type) {
            /* Data types that have no standard size */
            is_standard_size = 0;
            if (*active_byteorder != '^') {
                _append_char(str, '^');
                *active_byteorder = '^';
            }
        }
        else if (descr->byteorder == '<' || descr->byteorder == '>' ||
                 descr->byteorder == '=') {
            is_standard_size = 1;
            if (*active_byteorder != descr->byteorder) {
                _append_char(str, descr->byteorder);
                *active_byteorder = descr->byteorder;
            }

            if (is_native_only_type) {
                /*
                 * It's not possible to express native-only data types
                 * in non-native npy_byte orders
                 */
                PyErr_Format(PyExc_ValueError,
                             "cannot expose native-only dtype '%c' in "
                             "non-native byte order '%c' via buffer interface",
                             descr->type, descr->byteorder);
            }
        }

        switch (descr->type_num) {
        case NPY_BOOL:         if (_append_char(str, '?')) return -1; break;
        case NPY_BYTE:         if (_append_char(str, 'b')) return -1; break;
        case NPY_UBYTE:        if (_append_char(str, 'B')) return -1; break;
        case NPY_SHORT:        if (_append_char(str, 'h')) return -1; break;
        case NPY_USHORT:       if (_append_char(str, 'H')) return -1; break;
        case NPY_INT:          if (_append_char(str, 'i')) return -1; break;
        case NPY_UINT:         if (_append_char(str, 'I')) return -1; break;
        case NPY_LONG:
            if (is_standard_size && (NPY_SIZEOF_LONG == 8)) {
                if (_append_char(str, 'q')) return -1;
            }
            else {
                if (_append_char(str, 'l')) return -1;
            }
            break;
        case NPY_ULONG:
            if (is_standard_size && (NPY_SIZEOF_LONG == 8)) {
                if (_append_char(str, 'Q')) return -1;
            }
            else {
                if (_append_char(str, 'L')) return -1;
            }
            break;
        case NPY_LONGLONG:     if (_append_char(str, 'q')) return -1; break;
        case NPY_ULONGLONG:    if (_append_char(str, 'Q')) return -1; break;
        case NPY_HALF:         if (_append_char(str, 'e')) return -1; break;
        case NPY_FLOAT:        if (_append_char(str, 'f')) return -1; break;
        case NPY_DOUBLE:       if (_append_char(str, 'd')) return -1; break;
        case NPY_LONGDOUBLE:   if (_append_char(str, 'g')) return -1; break;
        case NPY_CFLOAT:       if (_append_str(str, "Zf")) return -1; break;
        case NPY_CDOUBLE:      if (_append_str(str, "Zd")) return -1; break;
        case NPY_CLONGDOUBLE:  if (_append_str(str, "Zg")) return -1; break;
        /* XXX: datetime */
        /* XXX: timedelta */
        case NPY_OBJECT:       if (_append_char(str, 'O')) return -1; break;
        case NPY_STRING: {
            char buf[128];
            PyOS_snprintf(buf, sizeof(buf), "%ds", descr->elsize);
            if (_append_str(str, buf)) return -1;
            break;
        }
        case NPY_UNICODE: {
            /* Numpy Unicode is always 4-byte */
            char buf[128];
            assert(descr->elsize % 4 == 0);
            PyOS_snprintf(buf, sizeof(buf), "%dw", descr->elsize / 4);
            if (_append_str(str, buf)) return -1;
            break;
        }
        case NPY_VOID: {
            /* Insert padding bytes */
            char buf[128];
            PyOS_snprintf(buf, sizeof(buf), "%dx", descr->elsize);
            if (_append_str(str, buf)) return -1;
            break;
        }
        default:
            PyErr_Format(PyExc_ValueError,
                         "cannot include dtype '%c' in a buffer",
                         descr->type);
            return -1;
        }
    }

    return 0;
}


/*
 * Global information about all active buffers
 *
 * Note: because for backward compatibility we cannot define bf_releasebuffer,
 * we must manually keep track of the additional data required by the buffers.
 */

/* Additional per-array data required for providing the buffer interface */
typedef struct {
    char *format;
    int ndim;
    Py_ssize_t *strides;
    Py_ssize_t *shape;
} _buffer_info_t;

/*
 * { id(array): [list of pointers to _buffer_info_t, the last one is latest] }
 *
 * Because shape, strides, and format can be different for different buffers,
 * we may need to keep track of multiple buffer infos for each array.
 *
 * However, when none of them has changed, the same buffer info may be reused.
 *
 * Thread-safety is provided by GIL.
 */
static PyObject *_buffer_info_cache = NULL;

/* Fill in the info structure */
static _buffer_info_t*
_buffer_info_new(PyArrayObject *arr)
{
    _buffer_info_t *info;
    _tmp_string_t fmt = {0,0,0};
    int k;

    info = (_buffer_info_t*)malloc(sizeof(_buffer_info_t));

    /* Fill in format */
    if (_buffer_format_string(PyArray_DESCR(arr), &fmt, arr, NULL, NULL) != 0) {
        free(info);
        return NULL;
    }
    _append_char(&fmt, '\0');
    info->format = fmt.s;

    /* Fill in shape and strides */
    info->ndim = PyArray_NDIM(arr);

    if (info->ndim == 0) {
        info->shape = NULL;
        info->strides = NULL;
    }
    else {
        info->shape = (Py_ssize_t*)malloc(sizeof(Py_ssize_t)
                                          * PyArray_NDIM(arr) * 2 + 1);
        info->strides = info->shape + PyArray_NDIM(arr);
        for (k = 0; k < PyArray_NDIM(arr); ++k) {
            info->shape[k] = PyArray_DIMS(arr)[k];
            info->strides[k] = PyArray_STRIDES(arr)[k];
        }
    }

    return info;
}

/* Compare two info structures */
static Py_ssize_t
_buffer_info_cmp(_buffer_info_t *a, _buffer_info_t *b)
{
    Py_ssize_t c;
    int k;

    c = strcmp(a->format, b->format);
    if (c != 0) return c;

    c = a->ndim - b->ndim;
    if (c != 0) return c;

    for (k = 0; k < a->ndim; ++k) {
        c = a->shape[k] - b->shape[k];
        if (c != 0) return c;
        c = a->strides[k] - b->strides[k];
        if (c != 0) return c;
    }

    return 0;
}

static void
_buffer_info_free(_buffer_info_t *info)
{
    if (info->format) {
        free(info->format);
    }
    if (info->shape) {
        free(info->shape);
    }
    free(info);
}

/* Get buffer info from the global dictionary */
static _buffer_info_t*
_buffer_get_info(PyObject *arr)
{
    PyObject *key, *item_list, *item;
    _buffer_info_t *info = NULL, *old_info = NULL;

    if (_buffer_info_cache == NULL) {
        _buffer_info_cache = PyDict_New();
        if (_buffer_info_cache == NULL) {
            return NULL;
        }
    }

    /* Compute information */
    info = _buffer_info_new((PyArrayObject*)arr);
    if (info == NULL) {
        return NULL;
    }

    /* Check if it is identical with an old one; reuse old one, if yes */
    key = PyLong_FromVoidPtr((void*)arr);
    item_list = PyDict_GetItem(_buffer_info_cache, key);

    if (item_list != NULL) {
        Py_INCREF(item_list);
        if (PyList_GET_SIZE(item_list) > 0) {
            item = PyList_GetItem(item_list, PyList_GET_SIZE(item_list) - 1);
            old_info = (_buffer_info_t*)PyLong_AsVoidPtr(item);

            if (_buffer_info_cmp(info, old_info) == 0) {
                _buffer_info_free(info);
                info = old_info;
            }
        }
    }
    else {
        item_list = PyList_New(0);
        PyDict_SetItem(_buffer_info_cache, key, item_list);
    }

    if (info != old_info) {
        /* Needs insertion */
        item = PyLong_FromVoidPtr((void*)info);
        PyList_Append(item_list, item);
        Py_DECREF(item);
    }

    Py_DECREF(item_list);
    Py_DECREF(key);
    return info;
}

/* Clear buffer info from the global dictionary */
static void
_buffer_clear_info(PyObject *arr)
{
    PyObject *key, *item_list, *item;
    _buffer_info_t *info;
    int k;

    if (_buffer_info_cache == NULL) {
        return;
    }

    key = PyLong_FromVoidPtr((void*)arr);
    item_list = PyDict_GetItem(_buffer_info_cache, key);
    if (item_list != NULL) {
        for (k = 0; k < PyList_GET_SIZE(item_list); ++k) {
            item = PyList_GET_ITEM(item_list, k);
            info = (_buffer_info_t*)PyLong_AsVoidPtr(item);
            _buffer_info_free(info);
        }
        PyDict_DelItem(_buffer_info_cache, key);
    }

    Py_DECREF(key);
}

/*
 * Retrieving buffers
 */

static int
array_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    PyArrayObject *self;
    _buffer_info_t *info = NULL;

    self = (PyArrayObject*)obj;

    /* Check whether we can provide the wanted properties */
    if ((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS &&
        !PyArray_CHKFLAGS(self, NPY_ARRAY_C_CONTIGUOUS)) {
        PyErr_SetString(PyExc_ValueError, "ndarray is not C-contiguous");
        goto fail;
    }
    if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS &&
        !PyArray_CHKFLAGS(self, NPY_ARRAY_F_CONTIGUOUS)) {
        PyErr_SetString(PyExc_ValueError, "ndarray is not Fortran contiguous");
        goto fail;
    }
    if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS
        && !PyArray_ISONESEGMENT(self)) {
        PyErr_SetString(PyExc_ValueError, "ndarray is not contiguous");
        goto fail;
    }
    if ((flags & PyBUF_STRIDES) != PyBUF_STRIDES &&
        (flags & PyBUF_ND) == PyBUF_ND &&
        !PyArray_CHKFLAGS(self, NPY_ARRAY_C_CONTIGUOUS)) {
        /* Non-strided N-dim buffers must be C-contiguous */
        PyErr_SetString(PyExc_ValueError, "ndarray is not C-contiguous");
        goto fail;
    }
    if ((flags & PyBUF_WRITEABLE) == PyBUF_WRITEABLE) {
        if (PyArray_FailUnlessWriteable(self, "buffer source array") < 0) {
            goto fail;
        }
    }
    /*
     * If a read-only buffer is requested on a read-write array, we return a
     * read-write buffer, which is dubious behavior. But that's why this call
     * is guarded by PyArray_ISWRITEABLE rather than (flags &
     * PyBUF_WRITEABLE).
     */
    if (PyArray_ISWRITEABLE(self)) {
        if (array_might_be_written(self) < 0) {
            goto fail;
        }
    }

    if (view == NULL) {
        PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
        goto fail;
    }

    /* Fill in information */
    info = _buffer_get_info(obj);
    if (info == NULL) {
        goto fail;
    }

    view->buf = PyArray_DATA(self);
    view->suboffsets = NULL;
    view->itemsize = PyArray_ITEMSIZE(self);
    view->readonly = !PyArray_ISWRITEABLE(self);
    view->internal = NULL;
    view->len = PyArray_NBYTES(self);
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
        view->format = info->format;
    } else {
        view->format = NULL;
    }
    if ((flags & PyBUF_ND) == PyBUF_ND) {
        view->ndim = info->ndim;
        view->shape = info->shape;
    }
    else {
        view->ndim = 0;
        view->shape = NULL;
    }
    if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
        view->strides = info->strides;
    }
    else {
        view->strides = NULL;
    }
    view->obj = (PyObject*)self;

    Py_INCREF(self);
    return 0;

fail:
    return -1;
}


/*
 * NOTE: for backward compatibility (esp. with PyArg_ParseTuple("s#", ...))
 * we do *not* define bf_releasebuffer at all.
 *
 * Instead, any extra data allocated with the buffer is released only in
 * array_dealloc.
 *
 * Ensuring that the buffer stays in place is taken care by refcounting;
 * ndarrays do not reallocate if there are references to them, and a buffer
 * view holds one reference.
 */

NPY_NO_EXPORT void
_array_dealloc_buffer_info(PyArrayObject *self)
{
    int reset_error_state = 0;
    PyObject *ptype, *pvalue, *ptraceback;

    /* This function may be called when processing an exception --
     * we need to stash the error state to avoid confusing PyDict
     */

    if (PyErr_Occurred()) {
        reset_error_state = 1;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    }

    _buffer_clear_info((PyObject*)self);

    if (reset_error_state) {
        PyErr_Restore(ptype, pvalue, ptraceback);
    }
}


/*************************************************************************/

NPY_NO_EXPORT PyBufferProcs array_as_buffer = {
#if !defined(NPY_PY3K)
    (readbufferproc)array_getreadbuf,       /*bf_getreadbuffer*/
    (writebufferproc)array_getwritebuf,     /*bf_getwritebuffer*/
    (segcountproc)array_getsegcount,        /*bf_getsegcount*/
    (charbufferproc)array_getcharbuf,       /*bf_getcharbuffer*/
#endif
    (getbufferproc)array_getbuffer,
    (releasebufferproc)0,
};


/*************************************************************************
 * Convert PEP 3118 format string to PyArray_Descr
 */

NPY_NO_EXPORT PyArray_Descr*
_descriptor_from_pep3118_format(char *s)
{
    char *buf, *p;
    int in_name = 0;
    PyObject *descr;
    PyObject *str;
    PyObject *_numpy_internal;

    if (s == NULL) {
        return PyArray_DescrNewFromType(NPY_BYTE);
    }

    /* Strip whitespace, except from field names */
    buf = (char*)malloc(strlen(s) + 1);
    p = buf;
    while (*s != '\0') {
        if (*s == ':') {
            in_name = !in_name;
            *p = *s;
        }
        else if (in_name || !NumPyOS_ascii_isspace(*s)) {
            *p = *s;
        }
        ++p;
        ++s;
    }
    *p = '\0';

    str = PyUString_FromStringAndSize(buf, strlen(buf));
    free(buf);
    if (str == NULL) {
        return NULL;
    }

    /* Convert */
    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        Py_DECREF(str);
        return NULL;
    }
    descr = PyObject_CallMethod(
        _numpy_internal, "_dtype_from_pep3118", "O", str);
    Py_DECREF(str);
    Py_DECREF(_numpy_internal);
    if (descr == NULL) {
        PyErr_Format(PyExc_ValueError,
                     "'%s' is not a valid PEP 3118 buffer format string", buf);
        return NULL;
    }
    if (!PyArray_DescrCheck(descr)) {
        PyErr_Format(PyExc_RuntimeError,
                     "internal error: numpy.core._internal._dtype_from_pep3118 "
                     "did not return a valid dtype, got %s", buf);
        return NULL;
    }
    return (PyArray_Descr*)descr;
}
