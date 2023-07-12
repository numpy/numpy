#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "npy_buffer.h"
#include "common.h"
#include "numpyos.h"
#include "arrayobject.h"
#include "scalartypes.h"
#include "dtypemeta.h"

/*************************************************************************
 ****************   Implement Buffer Protocol ****************************
 *************************************************************************/

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
    size_t allocated;
    size_t pos;
} _tmp_string_t;

#define INIT_SIZE   16

static int
_append_char(_tmp_string_t *s, char c)
{
    if (s->pos >= s->allocated) {
        char *p;
        size_t to_alloc = (s->allocated == 0) ? INIT_SIZE : (2 * s->allocated);

        p = PyObject_Realloc(s->s, to_alloc);
        if (p == NULL) {
            PyErr_SetString(PyExc_MemoryError, "memory allocation failed");
            return -1;
        }
        s->s = p;
        s->allocated = to_alloc;
    }
    s->s[s->pos] = c;
    ++s->pos;
    return 0;
}

static int
_append_str(_tmp_string_t *s, char const *p)
{
    for (; *p != '\0'; p++) {
        if (_append_char(s, *p) < 0) {
            return -1;
        }
    }
    return 0;
}

/*
 * Append a PEP3118-formatted field name, ":name:", to str
 */
static int
_append_field_name(_tmp_string_t *str, PyObject *name)
{
    int ret = -1;
    char *p;
    Py_ssize_t len;
    PyObject *tmp;
    /* FIXME: XXX -- should it use UTF-8 here? */
    tmp = PyUnicode_AsUTF8String(name);
    if (tmp == NULL || PyBytes_AsStringAndSize(tmp, &p, &len) < 0) {
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError, "invalid field name");
        goto fail;
    }
    if (_append_char(str, ':') < 0) {
        goto fail;
    }
    while (len > 0) {
        if (*p == ':') {
            PyErr_SetString(PyExc_ValueError,
                            "':' is not an allowed character in buffer "
                            "field names");
            goto fail;
        }
        if (_append_char(str, *p) < 0) {
            goto fail;
        }
        ++p;
        --len;
    }
    if (_append_char(str, ':') < 0) {
        goto fail;
    }
    ret = 0;
fail:
    Py_XDECREF(tmp);
    return ret;
}

/*
 * Return non-zero if a type is aligned in each item in the given array,
 * AND, the descr element size is a multiple of the alignment,
 * AND, the array data is positioned to alignment granularity.
 */
static inline int
_is_natively_aligned_at(PyArray_Descr *descr,
                        PyArrayObject *arr, Py_ssize_t offset)
{
    int k;

    if (NPY_LIKELY(descr == PyArray_DESCR(arr))) {
        /*
         * If the descriptor is the arrays descriptor we can assume the
         * array's alignment is correct.
         */
        assert(offset == 0);
        if (PyArray_ISALIGNED(arr)) {
            assert(descr->elsize % descr->alignment == 0);
            return 1;
        }
        return 0;
    }

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

/*
 * Fill in str with an appropriate PEP 3118 format string, based on
 * descr. For structured dtypes, calls itself recursively. Each call extends
 * str at offset then updates offset, and uses  descr->byteorder, (and
 * possibly the byte order in obj) to determine the byte-order char.
 *
 * Returns 0 for success, -1 for failure
 */
static int
_buffer_format_string(PyArray_Descr *descr, _tmp_string_t *str,
                      PyObject* obj, Py_ssize_t *offset,
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
        Py_ssize_t old_offset;
        char buf[128];
        int ret;

        if (PyTuple_Check(descr->subarray->shape)) {
            subarray_tuple = descr->subarray->shape;
            Py_INCREF(subarray_tuple);
        }
        else {
            subarray_tuple = Py_BuildValue("(O)", descr->subarray->shape);
        }

        if (_append_char(str, '(') < 0) {
            ret = -1;
            goto subarray_fail;
        }
        for (k = 0; k < PyTuple_GET_SIZE(subarray_tuple); ++k) {
            if (k > 0) {
                if (_append_char(str, ',') < 0) {
                    ret = -1;
                    goto subarray_fail;
                }
            }
            item = PyTuple_GET_ITEM(subarray_tuple, k);
            dim_size = PyNumber_AsSsize_t(item, NULL);

            PyOS_snprintf(buf, sizeof(buf), "%ld", (long)dim_size);
            if (_append_str(str, buf) < 0) {
                ret = -1;
                goto subarray_fail;
            }
            total_count *= dim_size;
        }
        if (_append_char(str, ')') < 0) {
            ret = -1;
            goto subarray_fail;
        }

        old_offset = *offset;
        ret = _buffer_format_string(descr->subarray->base, str, obj, offset,
                                    active_byteorder);
        *offset = old_offset + (*offset - old_offset) * total_count;

    subarray_fail:
        Py_DECREF(subarray_tuple);
        return ret;
    }
    else if (PyDataType_HASFIELDS(descr)) {
        Py_ssize_t base_offset = *offset;

        if (_append_str(str, "T{") < 0) return -1;
        for (k = 0; k < PyTuple_GET_SIZE(descr->names); ++k) {
            PyObject *name, *item, *offset_obj;
            PyArray_Descr *child;
            Py_ssize_t new_offset;
            int ret;

            name = PyTuple_GET_ITEM(descr->names, k);
            item = PyDict_GetItem(descr->fields, name);

            child = (PyArray_Descr*)PyTuple_GetItem(item, 0);
            offset_obj = PyTuple_GetItem(item, 1);
            new_offset = PyLong_AsLong(offset_obj);
            if (error_converting(new_offset)) {
                return -1;
            }
            new_offset += base_offset;

            /* Insert padding manually */
            if (*offset > new_offset) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "dtypes with overlapping or out-of-order fields are not "
                    "representable as buffers. Consider reordering the fields."
                );
                return -1;
            }
            while (*offset < new_offset) {
                if (_append_char(str, 'x') < 0) return -1;
                ++*offset;
            }

            /* Insert child item */
            ret = _buffer_format_string(child, str, obj, offset,
                                  active_byteorder);
            if (ret < 0) {
                return -1;
            }

            /* Insert field name */
            if (_append_field_name(str, name) < 0) return -1;
        }
        if (_append_char(str, '}') < 0) return -1;
    }
    else {
        int is_standard_size = 1;
        int is_natively_aligned;
        int is_native_only_type = (descr->type_num == NPY_LONGDOUBLE ||
                                   descr->type_num == NPY_CLONGDOUBLE);
        if (sizeof(npy_longlong) != 8) {
            is_native_only_type = is_native_only_type || (
                descr->type_num == NPY_LONGLONG ||
                descr->type_num == NPY_ULONGLONG);
        }

        if (PyArray_IsScalar(obj, Generic)) {
            /* scalars are always natively aligned */
            is_natively_aligned = 1;
        }
        else {
            is_natively_aligned = _is_natively_aligned_at(descr,
                                              (PyArrayObject*)obj, *offset);
        }

        *offset += descr->elsize;

        if (descr->byteorder == '=' && is_natively_aligned) {
            /* Prefer native types, to cater for Cython */
            is_standard_size = 0;
            if (*active_byteorder != '@') {
                if (_append_char(str, '@') < 0) return -1;
                *active_byteorder = '@';
            }
        }
        else if (descr->byteorder == '=' && is_native_only_type) {
            /* Data types that have no standard size */
            is_standard_size = 0;
            if (*active_byteorder != '^') {
                if (_append_char(str, '^') < 0) return -1;
                *active_byteorder = '^';
            }
        }
        else if (descr->byteorder == '<' || descr->byteorder == '>' ||
                 descr->byteorder == '=') {
            is_standard_size = 1;
            if (*active_byteorder != descr->byteorder) {
                if (_append_char(str, descr->byteorder) < 0) return -1;
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
                return -1;
            }
        }

        switch (descr->type_num) {
        case NPY_BOOL:         if (_append_char(str, '?') < 0) return -1; break;
        case NPY_BYTE:         if (_append_char(str, 'b') < 0) return -1; break;
        case NPY_UBYTE:        if (_append_char(str, 'B') < 0) return -1; break;
        case NPY_SHORT:        if (_append_char(str, 'h') < 0) return -1; break;
        case NPY_USHORT:       if (_append_char(str, 'H') < 0) return -1; break;
        case NPY_INT:          if (_append_char(str, 'i') < 0) return -1; break;
        case NPY_UINT:         if (_append_char(str, 'I') < 0) return -1; break;
        case NPY_LONG:
            if (is_standard_size && (NPY_SIZEOF_LONG == 8)) {
                if (_append_char(str, 'q') < 0) return -1;
            }
            else {
                if (_append_char(str, 'l') < 0) return -1;
            }
            break;
        case NPY_ULONG:
            if (is_standard_size && (NPY_SIZEOF_LONG == 8)) {
                if (_append_char(str, 'Q') < 0) return -1;
            }
            else {
                if (_append_char(str, 'L') < 0) return -1;
            }
            break;
        case NPY_LONGLONG:     if (_append_char(str, 'q') < 0) return -1; break;
        case NPY_ULONGLONG:    if (_append_char(str, 'Q') < 0) return -1; break;
        case NPY_HALF:         if (_append_char(str, 'e') < 0) return -1; break;
        case NPY_FLOAT:        if (_append_char(str, 'f') < 0) return -1; break;
        case NPY_DOUBLE:       if (_append_char(str, 'd') < 0) return -1; break;
        case NPY_LONGDOUBLE:   if (_append_char(str, 'g') < 0) return -1; break;
        case NPY_CFLOAT:       if (_append_str(str, "Zf") < 0) return -1; break;
        case NPY_CDOUBLE:      if (_append_str(str, "Zd") < 0) return -1; break;
        case NPY_CLONGDOUBLE:  if (_append_str(str, "Zg") < 0) return -1; break;
        /* XXX NPY_DATETIME */
        /* XXX NPY_TIMEDELTA */
        case NPY_OBJECT:       if (_append_char(str, 'O') < 0) return -1; break;
        case NPY_STRING: {
            char buf[128];
            PyOS_snprintf(buf, sizeof(buf), "%ds", descr->elsize);
            if (_append_str(str, buf) < 0) return -1;
            break;
        }
        case NPY_UNICODE: {
            /* NumPy Unicode is always 4-byte */
            char buf[128];
            assert(descr->elsize % 4 == 0);
            PyOS_snprintf(buf, sizeof(buf), "%dw", descr->elsize / 4);
            if (_append_str(str, buf) < 0) return -1;
            break;
        }
        case NPY_VOID: {
            /* Insert padding bytes */
            char buf[128];
            PyOS_snprintf(buf, sizeof(buf), "%dx", descr->elsize);
            if (_append_str(str, buf) < 0) return -1;
            break;
        }
        default:
            if (NPY_DT_is_legacy(NPY_DTYPE(descr))) {
                PyErr_Format(PyExc_ValueError,
                             "cannot include dtype '%c' in a buffer",
                             descr->type);
            }
            else {
                PyErr_Format(PyExc_ValueError,
                             "cannot include dtype '%s' in a buffer",
                             ((PyTypeObject*)NPY_DTYPE(descr))->tp_name);
            }
            return -1;
        }
    }

    return 0;
}


/*
 * Information about all active buffers is stored as a linked list on
 * the ndarray. The initial pointer is currently tagged to have a chance of
 * detecting incompatible subclasses.
 *
 * Note: because for backward compatibility we cannot define bf_releasebuffer,
 * we must manually keep track of the additional data required by the buffers.
 */

/* Additional per-array data required for providing the buffer interface */
typedef struct _buffer_info_t_tag {
    char *format;
    int ndim;
    Py_ssize_t *strides;
    Py_ssize_t *shape;
    struct _buffer_info_t_tag *next;
} _buffer_info_t;


/* Fill in the info structure */
static _buffer_info_t*
_buffer_info_new(PyObject *obj, int flags)
{
    /*
     * Note that the buffer info is cached as PyLongObjects making them appear
     * like unreachable lost memory to valgrind.
     */
    _buffer_info_t *info;
    _tmp_string_t fmt = {NULL, 0, 0};
    int k;
    PyArray_Descr *descr = NULL;
    int err = 0;

    if (PyArray_IsScalar(obj, Void)) {
        info = PyObject_Malloc(sizeof(_buffer_info_t));
        if (info == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        info->ndim = 0;
        info->shape = NULL;
        info->strides = NULL;

        descr = PyArray_DescrFromScalar(obj);
        if (descr == NULL) {
            goto fail;
        }
    }
    else {
        assert(PyArray_Check(obj));
        PyArrayObject * arr = (PyArrayObject *)obj;

        info = PyObject_Malloc(sizeof(_buffer_info_t) +
                               sizeof(Py_ssize_t) * PyArray_NDIM(arr) * 2);
        if (info == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        /* Fill in shape and strides */
        info->ndim = PyArray_NDIM(arr);

        if (info->ndim == 0) {
            info->shape = NULL;
            info->strides = NULL;
        }
        else {
            info->shape = (npy_intp *)((char *)info + sizeof(_buffer_info_t));
            assert((size_t)info->shape % sizeof(npy_intp) == 0);
            info->strides = info->shape + PyArray_NDIM(arr);

            /*
             * Some buffer users may expect a contiguous buffer to have well
             * formatted strides also when a dimension is 1, but we do not
             * guarantee this internally. Thus, recalculate strides for
             * contiguous arrays.
             */
            int f_contiguous = (flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS;
            if (PyArray_IS_C_CONTIGUOUS(arr) && !(
                    f_contiguous && PyArray_IS_F_CONTIGUOUS(arr))) {
                Py_ssize_t sd = PyArray_ITEMSIZE(arr);
                for (k = info->ndim-1; k >= 0; --k) {
                    info->shape[k] = PyArray_DIMS(arr)[k];
                    info->strides[k] = sd;
                    sd *= info->shape[k];
                }
            }
            else if (PyArray_IS_F_CONTIGUOUS(arr)) {
                Py_ssize_t sd = PyArray_ITEMSIZE(arr);
                for (k = 0; k < info->ndim; ++k) {
                    info->shape[k] = PyArray_DIMS(arr)[k];
                    info->strides[k] = sd;
                    sd *= info->shape[k];
                }
            }
            else {
                for (k = 0; k < PyArray_NDIM(arr); ++k) {
                    info->shape[k] = PyArray_DIMS(arr)[k];
                    info->strides[k] = PyArray_STRIDES(arr)[k];
                }
            }
        }
        descr = PyArray_DESCR(arr);
        Py_INCREF(descr);
    }

    /* Fill in format */
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
        err = _buffer_format_string(descr, &fmt, obj, NULL, NULL);
        Py_DECREF(descr);
        if (err != 0) {
            goto fail;
        }
        if (_append_char(&fmt, '\0') < 0) {
            goto fail;
        }
        info->format = fmt.s;
    }
    else {
        Py_DECREF(descr);
        info->format = NULL;
    }
    info->next = NULL;
    return info;

fail:
    PyObject_Free(fmt.s);
    PyObject_Free(info);
    return NULL;
}

/* Compare two info structures */
static Py_ssize_t
_buffer_info_cmp(_buffer_info_t *a, _buffer_info_t *b)
{
    Py_ssize_t c;
    int k;

    if (a->format != NULL && b->format != NULL) {
        c = strcmp(a->format, b->format);
        if (c != 0) return c;
    }
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


/*
 * Tag the buffer info pointer by adding 2 (unless it is NULL to simplify
 * object initialization).
 * The linked list of buffer-infos was appended to the array struct in
 * NumPy 1.20. Tagging the pointer gives us a chance to raise/print
 * a useful error message instead of crashing hard if a C-subclass uses
 * the same field.
 */
static inline void *
buffer_info_tag(void *buffer_info)
{
    if (buffer_info == NULL) {
        return buffer_info;
    }
    else {
        return (void *)((uintptr_t)buffer_info + 3);
    }
}


static inline int
_buffer_info_untag(
        void *tagged_buffer_info, _buffer_info_t **buffer_info, PyObject *obj)
{
    if (tagged_buffer_info == NULL) {
        *buffer_info = NULL;
        return 0;
    }
    if (NPY_UNLIKELY(((uintptr_t)tagged_buffer_info & 0x7) != 3)) {
        PyErr_Format(PyExc_RuntimeError,
                "Object of type %S appears to be C subclassed NumPy array, "
                "void scalar, or allocated in a non-standard way."
                "NumPy reserves the right to change the size of these "
                "structures. Projects are required to take this into account "
                "by either recompiling against a specific NumPy version or "
                "padding the struct and enforcing a maximum NumPy version.",
                Py_TYPE(obj));
        return -1;
    }
    *buffer_info = (void *)((uintptr_t)tagged_buffer_info - 3);
    return 0;
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
 *
 * This is stored in the array's _buffer_info slot (currently as a void *).
 */
static void
_buffer_info_free_untagged(void *_buffer_info)
{
    _buffer_info_t *next = _buffer_info;
    while (next != NULL) {
        _buffer_info_t *curr = next;
        next = curr->next;
        if (curr->format) {
            PyObject_Free(curr->format);
        }
        /* Shape is allocated as part of info */
        PyObject_Free(curr);
    }
}


/*
 * Checks whether the pointer is tagged, and then frees the cache list.
 * (The tag check is only for transition due to changed structure size in 1.20)
 */
NPY_NO_EXPORT int
_buffer_info_free(void *buffer_info, PyObject *obj)
{
    _buffer_info_t *untagged_buffer_info;
    if (_buffer_info_untag(buffer_info, &untagged_buffer_info, obj) < 0) {
        return -1;
    }
    _buffer_info_free_untagged(untagged_buffer_info);
    return 0;
}


/*
 * Get the buffer info returning either the old one (passed in) or a new
 * buffer info which adds holds on to (and thus replaces) the old one.
 */
static _buffer_info_t*
_buffer_get_info(void **buffer_info_cache_ptr, PyObject *obj, int flags)
{
    _buffer_info_t *info = NULL;
    _buffer_info_t *stored_info;  /* First currently stored buffer info */

    if (_buffer_info_untag(*buffer_info_cache_ptr, &stored_info, obj) < 0) {
        return NULL;
    }
    _buffer_info_t *old_info = stored_info;

    /* Compute information (it would be nice to skip this in simple cases) */
    info = _buffer_info_new(obj, flags);
    if (info == NULL) {
        return NULL;
    }

    if (old_info != NULL && _buffer_info_cmp(info, old_info) != 0) {
        _buffer_info_t *next_info = old_info->next;
        old_info = NULL;  /* Can't use this one, but possibly next */

         if (info->ndim > 1 && next_info != NULL) {
             /*
              * Some arrays are C- and F-contiguous and if they have more
              * than one dimension, the buffer-info may differ between the
              * two because strides for length 1 dimension may be adjusted.
              * If we export both buffers, the first stored one may be
              * the one for the other contiguity, so check both.
              * This is generally very unlikely in all other cases, since
              * in all other cases the first one will match unless array
              * metadata was modified in-place (which is discouraged).
              */
             if (_buffer_info_cmp(info, next_info) == 0) {
                 old_info = next_info;
             }
         }
    }
    if (old_info != NULL) {
        /*
         * The two info->format are considered equal if one of them
         * has no format set (meaning the format is arbitrary and can
         * be modified). If the new info has a format, but we reuse
         * the old one, this transfers the ownership to the old one.
         */
        if (old_info->format == NULL) {
            old_info->format = info->format;
            info->format = NULL;
        }
        _buffer_info_free_untagged(info);
        info = old_info;
    }
    else {
        /* Insert new info as first item in the linked buffer-info list. */
        info->next = stored_info;
        *buffer_info_cache_ptr = buffer_info_tag(info);
    }

    return info;
}


/*
 * Retrieving buffers for ndarray
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

    if (view == NULL) {
        PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
        goto fail;
    }

    /* Fill in information (and add it to _buffer_info if necessary) */
    info = _buffer_get_info(
            &((PyArrayObject_fields *)self)->_buffer_info, obj, flags);
    if (info == NULL) {
        goto fail;
    }

    view->buf = PyArray_DATA(self);
    view->suboffsets = NULL;
    view->itemsize = PyArray_ITEMSIZE(self);
    /*
     * If a read-only buffer is requested on a read-write array, we return a
     * read-write buffer as per buffer protocol.
     * We set a requested buffer to readonly also if the array will be readonly
     * after a deprecation. This jumps the deprecation, but avoiding the
     * warning is not convenient here. A warning is given if a writeable
     * buffer is requested since `PyArray_FailUnlessWriteable` is called above
     * (and clears the `NPY_ARRAY_WARN_ON_WRITE` flag).
     */
    view->readonly = (!PyArray_ISWRITEABLE(self) ||
                      PyArray_CHKFLAGS(self, NPY_ARRAY_WARN_ON_WRITE));
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
 * Retrieving buffers for void scalar (which can contain any complex types),
 * defined in buffer.c since it requires the complex format building logic.
 */
NPY_NO_EXPORT int
void_getbuffer(PyObject *self, Py_buffer *view, int flags)
{
    PyVoidScalarObject *scalar = (PyVoidScalarObject *)self;

    if (flags & PyBUF_WRITABLE) {
        PyErr_SetString(PyExc_BufferError, "scalar buffer is readonly");
        return -1;
    }

    view->ndim = 0;
    view->shape = NULL;
    view->strides = NULL;
    view->suboffsets = NULL;
    view->len = scalar->descr->elsize;
    view->itemsize = scalar->descr->elsize;
    view->readonly = 1;
    view->suboffsets = NULL;
    Py_INCREF(self);
    view->obj = self;
    view->buf = scalar->obval;

    if (((flags & PyBUF_FORMAT) != PyBUF_FORMAT)) {
        /* It is unnecessary to find the correct format */
        view->format = NULL;
        return 0;
    }

    /*
     * If a format is being exported, we need to use _buffer_get_info
     * to find the correct format.  This format must also be stored, since
     * at least in theory it can change (in practice it should never change).
     */
    _buffer_info_t *info = _buffer_get_info(&scalar->_buffer_info, self, flags);
    if (info == NULL) {
        Py_DECREF(self);
        return -1;
    }
    view->format = info->format;
    return 0;
}


/*************************************************************************/

NPY_NO_EXPORT PyBufferProcs array_as_buffer = {
    (getbufferproc)array_getbuffer,
    (releasebufferproc)0,
};


/*************************************************************************
 * Convert PEP 3118 format string to PyArray_Descr
 */

static int
_descriptor_from_pep3118_format_fast(char const *s, PyObject **result);

static int
_pep3118_letter_to_type(char letter, int native, int complex);

NPY_NO_EXPORT PyArray_Descr*
_descriptor_from_pep3118_format(char const *s)
{
    char *buf, *p;
    int in_name = 0;
    int obtained;
    PyObject *descr;
    PyObject *str;
    PyObject *_numpy_internal;

    if (s == NULL) {
        return PyArray_DescrNewFromType(NPY_BYTE);
    }

    /* Fast path */
    obtained = _descriptor_from_pep3118_format_fast(s, &descr);
    if (obtained) {
        return (PyArray_Descr*)descr;
    }

    /* Strip whitespace, except from field names */
    buf = malloc(strlen(s) + 1);
    if (buf == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    p = buf;
    while (*s != '\0') {
        if (*s == ':') {
            in_name = !in_name;
            *p = *s;
            p++;
        }
        else if (in_name || !NumPyOS_ascii_isspace(*s)) {
            *p = *s;
            p++;
        }
        s++;
    }
    *p = '\0';

    str = PyUnicode_FromStringAndSize(buf, strlen(buf));
    if (str == NULL) {
        free(buf);
        return NULL;
    }

    /* Convert */
    _numpy_internal = PyImport_ImportModule("numpy.core._internal");
    if (_numpy_internal == NULL) {
        Py_DECREF(str);
        free(buf);
        return NULL;
    }
    descr = PyObject_CallMethod(
        _numpy_internal, "_dtype_from_pep3118", "O", str);
    Py_DECREF(str);
    Py_DECREF(_numpy_internal);
    if (descr == NULL) {
        PyObject *exc, *val, *tb;
        PyErr_Fetch(&exc, &val, &tb);
        PyErr_Format(PyExc_ValueError,
                     "'%s' is not a valid PEP 3118 buffer format string", buf);
        npy_PyErr_ChainExceptionsCause(exc, val, tb);
        free(buf);
        return NULL;
    }
    if (!PyArray_DescrCheck(descr)) {
        PyErr_Format(PyExc_RuntimeError,
                     "internal error: numpy.core._internal._dtype_from_pep3118 "
                     "did not return a valid dtype, got %s", buf);
        Py_DECREF(descr);
        free(buf);
        return NULL;
    }
    free(buf);
    return (PyArray_Descr*)descr;
}

/*
 * Fast path for parsing buffer strings corresponding to simple types.
 *
 * Currently, this deals only with single-element data types.
 */

static int
_descriptor_from_pep3118_format_fast(char const *s, PyObject **result)
{
    PyArray_Descr *descr;

    int is_standard_size = 0;
    char byte_order = '=';
    int is_complex = 0;

    int type_num = NPY_BYTE;
    int item_seen = 0;

    for (; *s != '\0'; ++s) {
        is_complex = 0;
        switch (*s) {
        case '@':
        case '^':
            /* ^ means no alignment; doesn't matter for a single element */
            byte_order = '=';
            is_standard_size = 0;
            break;
        case '<':
            byte_order = '<';
            is_standard_size = 1;
            break;
        case '>':
        case '!':
            byte_order = '>';
            is_standard_size = 1;
            break;
        case '=':
            byte_order = '=';
            is_standard_size = 1;
            break;
        case 'Z':
            is_complex = 1;
            ++s;
        default:
            if (item_seen) {
                /* Not a single-element data type */
                return 0;
            }
            type_num = _pep3118_letter_to_type(*s, !is_standard_size,
                                               is_complex);
            if (type_num < 0) {
                /* Something unknown */
                return 0;
            }
            item_seen = 1;
            break;
        }
    }

    if (!item_seen) {
        return 0;
    }

    descr = PyArray_DescrFromType(type_num);
    if (descr == NULL) {
        return 0;
    }
    if (byte_order == '=') {
        *result = (PyObject*)descr;
    }
    else {
        *result = (PyObject*)PyArray_DescrNewByteorder(descr, byte_order);
        Py_DECREF(descr);
        if (*result == NULL) {
            return 0;
        }
    }

    return 1;
}

static int
_pep3118_letter_to_type(char letter, int native, int complex)
{
    switch (letter)
    {
    case '?': return NPY_BOOL;
    case 'b': return NPY_BYTE;
    case 'B': return NPY_UBYTE;
    case 'h': return native ? NPY_SHORT : NPY_INT16;
    case 'H': return native ? NPY_USHORT : NPY_UINT16;
    case 'i': return native ? NPY_INT : NPY_INT32;
    case 'I': return native ? NPY_UINT : NPY_UINT32;
    case 'l': return native ? NPY_LONG : NPY_INT32;
    case 'L': return native ? NPY_ULONG : NPY_UINT32;
    case 'q': return native ? NPY_LONGLONG : NPY_INT64;
    case 'Q': return native ? NPY_ULONGLONG : NPY_UINT64;
    case 'e': return NPY_HALF;
    case 'f': return complex ? NPY_CFLOAT : NPY_FLOAT;
    case 'd': return complex ? NPY_CDOUBLE : NPY_DOUBLE;
    case 'g': return native ? (complex ? NPY_CLONGDOUBLE : NPY_LONGDOUBLE) : -1;
    default:
        /* Other unhandled cases */
        return -1;
    }
    return -1;
}
