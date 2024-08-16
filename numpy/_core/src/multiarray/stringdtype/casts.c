#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/halffloat.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"

#include "common.h"
#include "numpyos.h"
#include "umathmodule.h"
#include "gil_utils.h"
#include "static_string.h"
#include "dtypemeta.h"
#include "dtype.h"
#include "utf8_utils.h"


#define ANY_TO_STRING_RESOLVE_DESCRIPTORS(safety)                              \
        static NPY_CASTING any_to_string_##safety##_resolve_descriptors(       \
                PyObject *NPY_UNUSED(self),                                    \
                PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),                      \
                PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2], \
                npy_intp *NPY_UNUSED(view_offset))                             \
        {                                                                      \
            if (given_descrs[1] == NULL) {                                     \
                PyArray_Descr *new =                                           \
                        (PyArray_Descr *)new_stringdtype_instance(             \
                                NULL, 1);                                      \
                if (new == NULL) {                                             \
                    return (NPY_CASTING)-1;                                    \
                }                                                              \
                loop_descrs[1] = new;                                          \
            }                                                                  \
            else {                                                             \
                Py_INCREF(given_descrs[1]);                                    \
                loop_descrs[1] = given_descrs[1];                              \
            }                                                                  \
                                                                               \
            Py_INCREF(given_descrs[0]);                                        \
            loop_descrs[0] = given_descrs[0];                                  \
                                                                               \
            return NPY_##safety##_CASTING;                                     \
        }

ANY_TO_STRING_RESOLVE_DESCRIPTORS(SAFE)
ANY_TO_STRING_RESOLVE_DESCRIPTORS(SAME_KIND)


static NPY_CASTING
string_to_string_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                     PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                     PyArray_Descr *given_descrs[2],
                                     PyArray_Descr *loop_descrs[2],
                                     npy_intp *view_offset)
{
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = stringdtype_finalize_descr(given_descrs[0]);
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    PyArray_StringDTypeObject *descr0 = (PyArray_StringDTypeObject *)loop_descrs[0];
    PyArray_StringDTypeObject *descr1 = (PyArray_StringDTypeObject *)loop_descrs[1];

    if ((descr0->na_object != NULL) && (descr1->na_object == NULL)) {
        // Cast from a dtype with an NA to one without, so it's a lossy
        // unsafe cast. The other way around is still safe, because
        // there isn't an NA in the source to lossily convert.
        return NPY_UNSAFE_CASTING;
    }

    // views are only legal between descriptors that share allocators (e.g. the same object)
    if (descr0->allocator == descr1->allocator) {
        *view_offset = 0;
    };

    return NPY_NO_CASTING;
}

static int
string_to_string(PyArrayMethod_Context *context, char *const data[],
                 npy_intp const dimensions[], npy_intp const strides[],
                 NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_StringDTypeObject *idescr = (PyArray_StringDTypeObject *)context->descriptors[0];
    PyArray_StringDTypeObject *odescr = (PyArray_StringDTypeObject *)context->descriptors[1];
    int in_has_null = idescr->na_object != NULL;
    int out_has_null = odescr->na_object != NULL;
    const npy_static_string *in_na_name = &idescr->na_name;
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    npy_string_allocator *allocators[2] = {NULL, NULL};
    NpyString_acquire_allocators(2, context->descriptors, allocators);
    npy_string_allocator *iallocator = allocators[0];
    npy_string_allocator *oallocator = allocators[1];


    while (N--) {
        const npy_packed_static_string *s = (npy_packed_static_string *)in;
        npy_packed_static_string *os = (npy_packed_static_string *)out;
        if (!NpyString_share_memory(s, iallocator, os, oallocator)) {
            if (in_has_null && !out_has_null && NpyString_isnull(s)) {
                // lossy but this is an unsafe cast so this is OK
                if (NpyString_pack(oallocator, os, in_na_name->buf,
                                   in_na_name->size) < 0) {
                    npy_gil_error(PyExc_MemoryError,
                              "Failed to pack string in string to string "
                              "cast.");
                    goto fail;
                }
            }
            else if (free_and_copy(iallocator, oallocator, s, os,
                                   "string to string cast") < 0) {
                goto fail;
            }
        }

        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocators(2, allocators);

    return 0;

fail:

    NpyString_release_allocators(2, allocators);

    return -1;
}

static PyType_Slot s2s_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_string_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_string},
        {NPY_METH_unaligned_strided_loop, &string_to_string},
        {0, NULL}};

static char *s2s_name = "cast_StringDType_to_StringDType";

// unicode to string

static int
unicode_to_string(PyArrayMethod_Context *context, char *const data[],
                  npy_intp const dimensions[], npy_intp const strides[],
                  NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr *const *descrs = context->descriptors;
    PyArray_StringDTypeObject *sdescr = (PyArray_StringDTypeObject *)descrs[1];

    npy_string_allocator *allocator = NpyString_acquire_allocator(sdescr);

    long max_in_size = (descrs[0]->elsize) / sizeof(Py_UCS4);

    npy_intp N = dimensions[0];
    Py_UCS4 *in = (Py_UCS4 *)data[0];
    char *out = data[1];

    npy_intp in_stride = strides[0] / sizeof(Py_UCS4);
    npy_intp out_stride = strides[1];

    while (N--) {
        size_t out_num_bytes = 0;
        size_t num_codepoints = 0;
        if (utf8_size(in, max_in_size, &num_codepoints, &out_num_bytes) ==
            -1) {
            npy_gil_error(PyExc_TypeError, "Invalid unicode code point found");
            goto fail;
        }
        npy_static_string out_ss = {0, NULL};
        if (load_new_string((npy_packed_static_string *)out,
                            &out_ss, out_num_bytes, allocator,
                            "unicode to string cast") == -1) {
            goto fail;
        }
        // ignores const to fill in the buffer
        char *out_buf = (char *)out_ss.buf;
        for (size_t i = 0; i < num_codepoints; i++) {
            // get code point
            Py_UCS4 code = in[i];

            // will be filled with UTF-8 bytes
            char utf8_c[4] = {0};

            // we already checked for invalid code points above,
            // so no need to do error checking here
            size_t num_bytes = ucs4_code_to_utf8_char(code, utf8_c);

            // copy utf8_c into out_buf
            strncpy(out_buf, utf8_c, num_bytes);

            // increment out_buf by the size of the character
            out_buf += num_bytes;
        }

        // reset out_buf -- and thus out.buf -- to the beginning of the string
        out_buf -= out_num_bytes;

        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);

    return 0;

fail:

    NpyString_release_allocator(allocator);

    return -1;
}

static PyType_Slot u2s_slots[] = {{NPY_METH_resolve_descriptors,
                                   &any_to_string_SAME_KIND_resolve_descriptors},
                                  {NPY_METH_strided_loop, &unicode_to_string},
                                  {0, NULL}};

static char *u2s_name = "cast_Unicode_to_StringDType";

// string to unicode

static NPY_CASTING
string_to_fixed_width_resolve_descriptors(
        PyObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        // currently there's no way to determine the correct output
        // size, so set an error and bail
        PyErr_SetString(
                PyExc_TypeError,
                "Casting from StringDType to a fixed-width dtype with an "
                "unspecified size is not currently supported, specify "
                "an explicit size for the output dtype instead.");
        return (NPY_CASTING)-1;
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return NPY_SAME_KIND_CASTING;
}

static int
load_nullable_string(const npy_packed_static_string *ps,
                     npy_static_string *s,
                     int has_null,
                     int has_string_na,
                     const npy_static_string *default_string,
                     const npy_static_string *na_name,
                     npy_string_allocator *allocator,
                     char *context)
{
    int is_null = NpyString_load(allocator, ps, s);
    if (is_null == -1) {
        npy_gil_error(PyExc_MemoryError,
                      "Failed to load string in %s", context);
        return -1;
    }
    else if (is_null) {
        if (has_null && !has_string_na) {
            // lossy but not much else we can do
            *s = *na_name;
        }
        else {
            *s = *default_string;
        }
    }
    return 0;
}

static int
string_to_unicode(PyArrayMethod_Context *context, char *const data[],
                  npy_intp const dimensions[], npy_intp const strides[],
                  NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    int has_null = descr->na_object != NULL;
    int has_string_na = descr->has_string_na;
    const npy_static_string *default_string = &descr->default_string;
    const npy_static_string *na_name = &descr->na_name;
    npy_intp N = dimensions[0];
    char *in = data[0];
    Py_UCS4 *out = (Py_UCS4 *)data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(Py_UCS4);
    // max number of UCS4 characters that can fit in the output
    size_t max_out_size = (context->descriptors[1]->elsize) / sizeof(Py_UCS4);

    while (N--) {
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;
        npy_static_string s = {0, NULL};
        if (load_nullable_string(ps, &s, has_null, has_string_na,
                                 default_string, na_name, allocator,
                                 "in string to unicode cast") == -1) {
            goto fail;
        }

        unsigned char *this_string = (unsigned char *)(s.buf);
        size_t n_bytes = s.size;
        size_t tot_n_bytes = 0;

        if (n_bytes == 0) {
            for (int i=0; i < max_out_size; i++) {
                out[i] = (Py_UCS4)0;
            }
        }
        else {
            int i = 0;
            for (; i < max_out_size && tot_n_bytes < n_bytes; i++) {
                int num_bytes = utf8_char_to_ucs4_code(this_string, &out[i]);

                // move to next character
                this_string += num_bytes;
                tot_n_bytes += num_bytes;
            }
            for(; i < max_out_size; i++) {
                out[i] = (Py_UCS4)0;
            }
        }

        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);

    return 0;

fail:
    NpyString_release_allocator(allocator);

    return -1;
}

static PyType_Slot s2u_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_fixed_width_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_unicode},
        {0, NULL}};

static char *s2u_name = "cast_StringDType_to_Unicode";

// string to bool

static NPY_CASTING
string_to_bool_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                   PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                   PyArray_Descr *given_descrs[2],
                                   PyArray_Descr *loop_descrs[2],
                                   npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = PyArray_DescrNewFromType(NPY_BOOL);
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return NPY_UNSAFE_CASTING;
}

static int
string_to_bool(PyArrayMethod_Context *context, char *const data[],
               npy_intp const dimensions[], npy_intp const strides[],
               NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    int has_null = descr->na_object != NULL;
    int has_string_na = descr->has_string_na;
    int has_nan_na = descr->has_nan_na;
    const npy_static_string *default_string = &descr->default_string;

    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;
        npy_static_string s = {0, NULL};
        int is_null = NpyString_load(allocator, ps, &s);
        if (is_null == -1) {
            npy_gil_error(PyExc_MemoryError,
                          "Failed to load string in string to bool cast");
            goto fail;
        }
        else if (is_null) {
            if (has_null && !has_string_na) {
                if (has_nan_na) {
                    // numpy treats NaN as truthy, following python
                    *out = NPY_TRUE;
                }
                else {
                    *out = NPY_FALSE;
                }
            }
            else {
                *out = (npy_bool)(default_string->size == 0);
            }
        }
        else if (s.size == 0) {
            *out = NPY_FALSE;
        }
        else {
            *out = NPY_TRUE;
        }

        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);

    return 0;

fail:

    NpyString_release_allocator(allocator);

    return -1;
}

static PyType_Slot s2b_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_bool_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_bool},
        {0, NULL}};

static char *s2b_name = "cast_StringDType_to_Bool";

// bool to string

static int
bool_to_string(PyArrayMethod_Context *context, char *const data[],
               npy_intp const dimensions[], npy_intp const strides[],
               NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[1];
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);

    while (N--) {
        npy_packed_static_string *out_pss = (npy_packed_static_string *)out;
        char *ret_val = NULL;
        size_t size = 0;
        if ((npy_bool)(*in) == NPY_TRUE) {
            ret_val = "True";
            size = 4;
        }
        else if ((npy_bool)(*in) == NPY_FALSE) {
            ret_val = "False";
            size = 5;
        }
        else {
            npy_gil_error(PyExc_RuntimeError,
                          "invalid value encountered in bool to string cast");
            goto fail;
        }
        if (NpyString_pack(allocator, out_pss, ret_val, size) < 0) {
            npy_gil_error(PyExc_MemoryError,
                          "Failed to pack string in bool to string cast");
            goto fail;
        }
        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);

    return 0;

fail:

    NpyString_release_allocator(allocator);

    return -1;
}

static PyType_Slot b2s_slots[] = {{NPY_METH_resolve_descriptors,
                                   &any_to_string_SAFE_resolve_descriptors},
                                  {NPY_METH_strided_loop, &bool_to_string},
                                  {0, NULL}};

static char *b2s_name = "cast_Bool_to_StringDType";

// casts between string and (u)int dtypes


static int
load_non_nullable_string(char *in, int has_null, const npy_static_string *default_string,
                         npy_static_string *string_to_load, npy_string_allocator *allocator,
                         int has_gil)
{
    const npy_packed_static_string *ps = (npy_packed_static_string *)in;
    int isnull = NpyString_load(allocator, ps, string_to_load);
    if (isnull == -1) {
        char *msg = "Failed to load string for conversion to a non-nullable type";
        if (has_gil)
        {
            PyErr_SetString(PyExc_MemoryError, msg);
        }
        else {
            npy_gil_error(PyExc_MemoryError, msg);
        }
        return -1;
    }
    else if (isnull) {
        if (has_null) {
            char *msg = "Arrays with missing data cannot be converted to a non-nullable type";
            if (has_gil)
            {
                PyErr_SetString(PyExc_ValueError, msg);
            }
            else {
                npy_gil_error(PyExc_ValueError, msg);
            }
            return -1;
        }
        *string_to_load = *default_string;
    }
    return 0;
}

// note that this is only used to convert to numeric types and errors
// on nulls
static PyObject *
non_nullable_string_to_pystring(char *in, int has_null, const npy_static_string *default_string,
                                npy_string_allocator *allocator)
{
    npy_static_string s = {0, NULL};
    if (load_non_nullable_string(in, has_null, default_string, &s, allocator, 1) == -1) {
        return NULL;
    }
    PyObject *val_obj = PyUnicode_FromStringAndSize(s.buf, s.size);
    if (val_obj == NULL) {
        return NULL;
    }
    return val_obj;
}

static PyObject *
string_to_pylong(char *in, int has_null,
                 const npy_static_string *default_string,
                 npy_string_allocator *allocator)
{
    PyObject *val_obj = non_nullable_string_to_pystring(
            in, has_null, default_string, allocator);
    if (val_obj == NULL) {
        return NULL;
    }
    // interpret as an integer in base 10
    PyObject *pylong_value = PyLong_FromUnicodeObject(val_obj, 10);
    Py_DECREF(val_obj);
    return pylong_value;
}

static npy_longlong
stringbuf_to_uint(char *in, npy_ulonglong *value, int has_null,
                  const npy_static_string *default_string,
                  npy_string_allocator *allocator)
{
    PyObject *pylong_value =
            string_to_pylong(in, has_null, default_string, allocator);
    if (pylong_value == NULL) {
        return -1;
    }
    *value = PyLong_AsUnsignedLongLong(pylong_value);
    if (*value == (unsigned long long)-1 && PyErr_Occurred()) {
        Py_DECREF(pylong_value);
        return -1;
    }
    Py_DECREF(pylong_value);
    return 0;
}

static npy_longlong
stringbuf_to_int(char *in, npy_longlong *value, int has_null,
                 const npy_static_string *default_string,
                 npy_string_allocator *allocator)
{
    PyObject *pylong_value =
            string_to_pylong(in, has_null, default_string, allocator);
    if (pylong_value == NULL) {
        return -1;
    }
    *value = PyLong_AsLongLong(pylong_value);
    if (*value == -1 && PyErr_Occurred()) {
        Py_DECREF(pylong_value);
        return -1;
    }
    Py_DECREF(pylong_value);
    return 0;
}

static int
pyobj_to_string(PyObject *obj, char *out, npy_string_allocator *allocator)
{
    if (obj == NULL) {
        return -1;
    }
    PyObject *pystr_val = PyObject_Str(obj);
    Py_DECREF(obj);

    if (pystr_val == NULL) {
        return -1;
    }
    Py_ssize_t length;
    const char *cstr_val = PyUnicode_AsUTF8AndSize(pystr_val, &length);
    if (cstr_val == NULL) {
        Py_DECREF(pystr_val);
        return -1;
    }
    npy_packed_static_string *out_ss = (npy_packed_static_string *)out;
    if (NpyString_pack(allocator, out_ss, cstr_val, length) < 0) {
        npy_gil_error(PyExc_MemoryError,
                      "Failed to pack string while converting from python "
                      "string");
        Py_DECREF(pystr_val);
        return -1;
    }
    // implicitly deallocates cstr_val as well
    Py_DECREF(pystr_val);
    return 0;
}

static int
int_to_stringbuf(long long in, char *out, npy_string_allocator *allocator)
{
    PyObject *pylong_val = PyLong_FromLongLong(in);
    return pyobj_to_string(pylong_val, out, allocator);
}

static int
uint_to_stringbuf(unsigned long long in, char *out,
                  npy_string_allocator *allocator)
{
    PyObject *pylong_val = PyLong_FromUnsignedLongLong(in);
    return pyobj_to_string(pylong_val, out, allocator);
}

#define STRING_INT_CASTS(typename, typekind, shortname, numpy_tag,                \
                         printf_code, npy_longtype, longtype)                     \
        static NPY_CASTING string_to_##typename##_resolve_descriptors(            \
                PyObject *NPY_UNUSED(self),                                       \
                PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),                         \
                PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],    \
                npy_intp *NPY_UNUSED(view_offset))                                \
        {                                                                         \
            if (given_descrs[1] == NULL) {                                        \
                loop_descrs[1] = PyArray_DescrNewFromType(numpy_tag);             \
            }                                                                     \
            else {                                                                \
                Py_INCREF(given_descrs[1]);                                       \
                loop_descrs[1] = given_descrs[1];                                 \
            }                                                                     \
                                                                                  \
            Py_INCREF(given_descrs[0]);                                           \
            loop_descrs[0] = given_descrs[0];                                     \
                                                                                  \
            return NPY_UNSAFE_CASTING;                                            \
        }                                                                         \
                                                                                  \
        static int string_to_##                                                   \
        typename(PyArrayMethod_Context * context, char *const data[],             \
                 npy_intp const dimensions[], npy_intp const strides[],           \
                 NpyAuxData *NPY_UNUSED(auxdata))                                 \
        {                                                                         \
            PyArray_StringDTypeObject *descr =                                    \
                    ((PyArray_StringDTypeObject *)context->descriptors[0]);       \
            npy_string_allocator *allocator =                                     \
                    NpyString_acquire_allocator(descr);                           \
            int has_null = descr->na_object != NULL;                              \
            const npy_static_string *default_string = &descr->default_string;     \
                                                                                  \
            npy_intp N = dimensions[0];                                           \
            char *in = data[0];                                                   \
            npy_##typename *out = (npy_##typename *)data[1];                      \
                                                                                  \
            npy_intp in_stride = strides[0];                                      \
            npy_intp out_stride = strides[1] / sizeof(npy_##typename);            \
                                                                                  \
            while (N--) {                                                         \
                npy_longtype value;                                               \
                if (stringbuf_to_##typekind(in, &value, has_null, default_string, \
                                         allocator) != 0) {                       \
                    goto fail;                                                    \
                }                                                                 \
                *out = (npy_##typename)value;                                     \
                if (*out != value) {                                              \
                    /* out of bounds, raise error following NEP 50 behavior */    \
                    npy_gil_error(PyExc_OverflowError,                            \
                             "Integer %" #printf_code                             \
                             " is out of bounds "                                 \
                             "for " #typename,                                    \
                             value);                                              \
                    goto fail;                                                    \
                }                                                                 \
                in += in_stride;                                                  \
                out += out_stride;                                                \
            }                                                                     \
                                                                                  \
            NpyString_release_allocator(allocator);                               \
            return 0;                                                             \
                                                                                  \
          fail:                                                                   \
            NpyString_release_allocator(allocator);                               \
            return -1;                                                            \
        }                                                                         \
                                                                                  \
        static PyType_Slot s2##shortname##_slots[] = {                            \
                {NPY_METH_resolve_descriptors,                                    \
                 &string_to_##typename##_resolve_descriptors},                    \
                {NPY_METH_strided_loop, &string_to_##typename},                   \
                {0, NULL}};                                                       \
                                                                                  \
        static char *s2##shortname##_name = "cast_StringDType_to_" #typename;     \
                                                                                  \
        static int typename##_to_string(                                          \
                PyArrayMethod_Context *context, char *const data[],               \
                npy_intp const dimensions[], npy_intp const strides[],            \
                NpyAuxData *NPY_UNUSED(auxdata))                                  \
        {                                                                         \
            npy_intp N = dimensions[0];                                           \
            npy_##typename *in = (npy_##typename *)data[0];                       \
            char *out = data[1];                                                  \
                                                                                  \
            npy_intp in_stride = strides[0] / sizeof(npy_##typename);             \
            npy_intp out_stride = strides[1];                                     \
                                                                                  \
            PyArray_StringDTypeObject *descr =                                    \
                    (PyArray_StringDTypeObject *)context->descriptors[1];         \
            npy_string_allocator *allocator =                                     \
                    NpyString_acquire_allocator(descr);                           \
                                                                                  \
            while (N--) {                                                         \
                if (typekind##_to_stringbuf(                                      \
                            (longtype)*in, out, allocator) != 0) {                \
                    goto fail;                                                    \
                }                                                                 \
                                                                                  \
                in += in_stride;                                                  \
                out += out_stride;                                                \
            }                                                                     \
                                                                                  \
            NpyString_release_allocator(allocator);                               \
            return 0;                                                             \
                                                                                  \
          fail:                                                                   \
            NpyString_release_allocator(allocator);                               \
            return -1;                                                            \
        }                                                                         \
                                                                                  \
        static PyType_Slot shortname##2s_slots [] = {                             \
                {NPY_METH_resolve_descriptors,                                    \
                 &any_to_string_SAFE_resolve_descriptors},                        \
                {NPY_METH_strided_loop, &typename##_to_string},                   \
                {0, NULL}};                                                       \
                                                                                  \
        static char *shortname##2s_name = "cast_" #typename "_to_StringDType";

#define DTYPES_AND_CAST_SPEC(shortname, typename)                              \
        PyArray_DTypeMeta **s2##shortname##_dtypes = get_dtypes(               \
                &PyArray_StringDType,                                          \
                &PyArray_##typename##DType);                                   \
                                                                               \
        PyArrayMethod_Spec *StringTo##typename##CastSpec =                     \
                get_cast_spec(                                                 \
                        s2##shortname##_name, NPY_UNSAFE_CASTING,              \
                        NPY_METH_REQUIRES_PYAPI, s2##shortname##_dtypes,       \
                        s2##shortname##_slots);                                \
                                                                               \
        PyArray_DTypeMeta **shortname##2s_dtypes = get_dtypes(                 \
                &PyArray_##typename##DType,                                    \
                &PyArray_StringDType);                                         \
                                                                               \
        PyArrayMethod_Spec *typename##ToStringCastSpec = get_cast_spec(        \
                shortname##2s_name, NPY_SAFE_CASTING,                          \
                NPY_METH_REQUIRES_PYAPI, shortname##2s_dtypes,                 \
                shortname##2s_slots);

STRING_INT_CASTS(int8, int, i8, NPY_INT8, lli, npy_longlong, long long)
STRING_INT_CASTS(int16, int, i16, NPY_INT16, lli, npy_longlong, long long)
STRING_INT_CASTS(int32, int, i32, NPY_INT32, lli, npy_longlong, long long)
STRING_INT_CASTS(int64, int, i64, NPY_INT64, lli, npy_longlong, long long)

STRING_INT_CASTS(uint8, uint, u8, NPY_UINT8, llu, npy_ulonglong,
                 unsigned long long)
STRING_INT_CASTS(uint16, uint, u16, NPY_UINT16, llu, npy_ulonglong,
                 unsigned long long)
STRING_INT_CASTS(uint32, uint, u32, NPY_UINT32, llu, npy_ulonglong,
                 unsigned long long)
STRING_INT_CASTS(uint64, uint, u64, NPY_UINT64, llu, npy_ulonglong,
                 unsigned long long)

#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
// byte doesn't have a bitsized alias
STRING_INT_CASTS(byte, int, byte, NPY_BYTE, lli, npy_longlong, long long)
STRING_INT_CASTS(ubyte, uint, ubyte, NPY_UBYTE, llu, npy_ulonglong,
                 unsigned long long)
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
// short doesn't have a bitsized alias
STRING_INT_CASTS(short, int, short, NPY_SHORT, lli, npy_longlong, long long)
STRING_INT_CASTS(ushort, uint, ushort, NPY_USHORT, llu, npy_ulonglong,
                 unsigned long long)
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
// int doesn't have a bitsized alias
STRING_INT_CASTS(int, int, int, NPY_INT, lli, npy_longlong, long long)
STRING_INT_CASTS(uint, uint, uint, NPY_UINT, llu, npy_longlong, long long)
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
// long long doesn't have a bitsized alias
STRING_INT_CASTS(longlong, int, longlong, NPY_LONGLONG, lli, npy_longlong,
                 long long)
STRING_INT_CASTS(ulonglong, uint, ulonglong, NPY_ULONGLONG, llu, npy_ulonglong,
                 unsigned long long)
#endif

static PyObject *
string_to_pyfloat(char *in, int has_null,
                  const npy_static_string *default_string,
                  npy_string_allocator *allocator)
{
    PyObject *val_obj = non_nullable_string_to_pystring(
            in, has_null, default_string, allocator);
    if (val_obj == NULL) {
        return NULL;
    }
    PyObject *pyfloat_value = PyFloat_FromString(val_obj);
    Py_DECREF(val_obj);
    return pyfloat_value;
}

#define STRING_TO_FLOAT_CAST(typename, shortname, isinf_name,                 \
                             double_to_float)                                 \
    static int string_to_##                                                   \
            typename(PyArrayMethod_Context * context, char *const data[],     \
                     npy_intp const dimensions[], npy_intp const strides[],   \
                     NpyAuxData *NPY_UNUSED(auxdata))                         \
    {                                                                         \
        PyArray_StringDTypeObject *descr =                                    \
                (PyArray_StringDTypeObject *)context->descriptors[0];         \
        npy_string_allocator *allocator = NpyString_acquire_allocator(descr); \
        int has_null = (descr->na_object != NULL);                            \
        const npy_static_string *default_string = &descr->default_string;     \
                                                                              \
        npy_intp N = dimensions[0];                                           \
        char *in = data[0];                                                   \
        npy_##typename *out = (npy_##typename *)data[1];                      \
                                                                              \
        npy_intp in_stride = strides[0];                                      \
        npy_intp out_stride = strides[1] / sizeof(npy_##typename);            \
                                                                              \
        while (N--) {                                                         \
            PyObject *pyfloat_value = string_to_pyfloat(                      \
                    in, has_null, default_string, allocator);                 \
            if (pyfloat_value == NULL) {                                      \
                goto fail;                                                    \
            }                                                                 \
            double dval = PyFloat_AS_DOUBLE(pyfloat_value);                   \
            Py_DECREF(pyfloat_value);                                         \
            npy_##typename fval = (double_to_float)(dval);                    \
                                                                              \
            if (NPY_UNLIKELY(isinf_name(fval) && !(npy_isinf(dval)))) {       \
                if (PyUFunc_GiveFloatingpointErrors("cast",                   \
                                                    NPY_FPE_OVERFLOW) < 0) {  \
                    goto fail;                                                \
                }                                                             \
            }                                                                 \
                                                                              \
            *out = fval;                                                      \
                                                                              \
            in += in_stride;                                                  \
            out += out_stride;                                                \
        }                                                                     \
                                                                              \
        NpyString_release_allocator(allocator);                               \
        return 0;                                                             \
    fail:                                                                     \
        NpyString_release_allocator(allocator);                               \
        return -1;                                                            \
    }                                                                         \
                                                                              \
    static PyType_Slot s2##shortname##_slots[] = {                            \
            {NPY_METH_resolve_descriptors,                                    \
             &string_to_##typename##_resolve_descriptors},                    \
            {NPY_METH_strided_loop, &string_to_##typename},                   \
            {0, NULL}};                                                       \
                                                                              \
    static char *s2##shortname##_name = "cast_StringDType_to_" #typename;

#define STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(typename, npy_typename)        \
    static NPY_CASTING string_to_##typename##_resolve_descriptors(         \
            PyObject *NPY_UNUSED(self),                                    \
            PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),                      \
            PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2], \
            npy_intp *NPY_UNUSED(view_offset))                             \
    {                                                                      \
        if (given_descrs[1] == NULL) {                                     \
            loop_descrs[1] = PyArray_DescrNewFromType(NPY_##npy_typename); \
        }                                                                  \
        else {                                                             \
            Py_INCREF(given_descrs[1]);                                    \
            loop_descrs[1] = given_descrs[1];                              \
        }                                                                  \
                                                                           \
        Py_INCREF(given_descrs[0]);                                        \
        loop_descrs[0] = given_descrs[0];                                  \
                                                                           \
        return NPY_UNSAFE_CASTING;                                         \
    }

#define FLOAT_TO_STRING_CAST(typename, shortname, float_to_double)            \
    static int typename##_to_string(                                          \
            PyArrayMethod_Context *context, char *const data[],               \
            npy_intp const dimensions[], npy_intp const strides[],            \
            NpyAuxData *NPY_UNUSED(auxdata))                                  \
    {                                                                         \
        npy_intp N = dimensions[0];                                           \
        npy_##typename *in = (npy_##typename *)data[0];                       \
        char *out = data[1];                                                  \
        PyArray_Descr *float_descr = context->descriptors[0];                 \
                                                                              \
        npy_intp in_stride = strides[0] / sizeof(npy_##typename);             \
        npy_intp out_stride = strides[1];                                     \
                                                                              \
        PyArray_StringDTypeObject *descr =                                    \
                (PyArray_StringDTypeObject *)context->descriptors[1];         \
        npy_string_allocator *allocator = NpyString_acquire_allocator(descr); \
                                                                              \
        while (N--) {                                                         \
            PyObject *scalar_val = PyArray_Scalar(in, float_descr, NULL);     \
            if (pyobj_to_string(scalar_val, out, allocator) == -1) {          \
                goto fail;                                                    \
            }                                                                 \
                                                                              \
            in += in_stride;                                                  \
            out += out_stride;                                                \
        }                                                                     \
                                                                              \
        NpyString_release_allocator(allocator);                               \
        return 0;                                                             \
    fail:                                                                     \
        NpyString_release_allocator(allocator);                               \
        return -1;                                                            \
    }                                                                         \
                                                                              \
    static PyType_Slot shortname##2s_slots [] = {                             \
            {NPY_METH_resolve_descriptors,                                    \
             &any_to_string_SAFE_resolve_descriptors},                        \
            {NPY_METH_strided_loop, &typename##_to_string},                   \
            {0, NULL}};                                                       \
                                                                              \
    static char *shortname##2s_name = "cast_" #typename "_to_StringDType";

STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(float64, DOUBLE)

static int
string_to_float64(PyArrayMethod_Context *context, char *const data[],
                  npy_intp const dimensions[], npy_intp const strides[],
                  NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    int has_null = descr->na_object != NULL;
    const npy_static_string *default_string = &descr->default_string;
    npy_intp N = dimensions[0];
    char *in = data[0];
    npy_float64 *out = (npy_float64 *)data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(npy_float64);

    while (N--) {
        PyObject *pyfloat_value =
                string_to_pyfloat(in, has_null, default_string, allocator);
        if (pyfloat_value == NULL) {
            goto fail;
        }
        *out = (npy_float64)PyFloat_AS_DOUBLE(pyfloat_value);
        Py_DECREF(pyfloat_value);

        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);
    return 0;

fail:
    NpyString_release_allocator(allocator);
    return -1;
}

static PyType_Slot s2f64_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_float64_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_float64},
        {0, NULL}};

static char *s2f64_name = "cast_StringDType_to_float64";

FLOAT_TO_STRING_CAST(float64, f64, double)

STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(float32, FLOAT)
STRING_TO_FLOAT_CAST(float32, f32, npy_isinf, npy_float32)
FLOAT_TO_STRING_CAST(float32, f32, double)

STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(float16, HALF)
STRING_TO_FLOAT_CAST(float16, f16, npy_half_isinf, npy_double_to_half)
FLOAT_TO_STRING_CAST(float16, f16, npy_half_to_double)

// string to longdouble

STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(longdouble, LONGDOUBLE);

static int
string_to_longdouble(PyArrayMethod_Context *context, char *const data[],
                     npy_intp const dimensions[], npy_intp const strides[],
                     NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    int has_null = descr->na_object != NULL;
    const npy_static_string *default_string = &descr->default_string;
    npy_intp N = dimensions[0];
    char *in = data[0];
    npy_longdouble *out = (npy_longdouble *)data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(npy_longdouble);

    while (N--) {
        npy_static_string s = {0, NULL};
        if (load_non_nullable_string(in, has_null, default_string, &s, allocator, 0) == -1) {
            goto fail;
        }

        // allocate temporary null-terminated copy
        char *buf = PyMem_RawMalloc(s.size + 1);
        memcpy(buf, s.buf, s.size);
        buf[s.size] = '\0';

        char *end = NULL;
        errno = 0;
        npy_longdouble longdouble_value = NumPyOS_ascii_strtold(buf, &end);

        if (errno == ERANGE) {
            /* strtold returns INFINITY of the correct sign. */
            if (PyErr_Warn(PyExc_RuntimeWarning,
                           "overflow encountered in conversion from string") < 0) {
                PyMem_RawFree(buf);
                goto fail;
            }
        }
        else if (errno || end == buf || *end) {
            PyErr_Format(PyExc_ValueError,
                         "invalid literal for long double: %s (%s)",
                         buf,
                         strerror(errno));
            PyMem_RawFree(buf);
            goto fail;
        }
        PyMem_RawFree(buf);
        *out = longdouble_value;

        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);
    return 0;

fail:
    NpyString_release_allocator(allocator);
    return -1;
}

static PyType_Slot s2ld_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_longdouble_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_longdouble},
        {0, NULL}};

static char *s2ld_name = "cast_StringDType_to_longdouble";

// longdouble to string

// TODO: this is incorrect. The longdouble to unicode cast is also broken in
// the same way. To fix this we'd need an ldtoa implementation in NumPy. It's
// not in the standard library. Another option would be to use `snprintf` but we'd
// need to somehow pre-calculate the size of the result string.

FLOAT_TO_STRING_CAST(longdouble, ld, npy_longdouble)

// string to cfloat

static PyObject*
string_to_pycomplex(char *in, int has_null,
                    const npy_static_string *default_string,
                    npy_string_allocator *allocator)
{
    PyObject *val_obj = non_nullable_string_to_pystring(
            in, has_null, default_string, allocator);
    if (val_obj == NULL) {
        return NULL;
    }
    PyObject *args = PyTuple_Pack(1, val_obj);
    Py_DECREF(val_obj);
    if (args == NULL) {
        return NULL;
    }
    PyObject *pycomplex_value = PyComplex_Type.tp_new(&PyComplex_Type, args, NULL);
    Py_DECREF(args);
    return pycomplex_value;
}

#define STRING_TO_CFLOAT_CAST(ctype, suffix, ftype)                              \
        static int                                                               \
        string_to_##ctype(PyArrayMethod_Context *context, char *const data[],    \
                          npy_intp const dimensions[], npy_intp const strides[], \
                          NpyAuxData *NPY_UNUSED(auxdata))                       \
        {                                                                        \
        PyArray_StringDTypeObject *descr =                                       \
                (PyArray_StringDTypeObject *)context->descriptors[0];            \
        npy_string_allocator *allocator = NpyString_acquire_allocator(descr);    \
        int has_null = descr->na_object != NULL;                                 \
        const npy_static_string *default_string = &descr->default_string;        \
        npy_intp N = dimensions[0];                                              \
        char *in = data[0];                                                      \
        npy_##ctype *out = (npy_##ctype *)data[1];                               \
                                                                                 \
        npy_intp in_stride = strides[0];                                         \
        npy_intp out_stride = strides[1] / sizeof(npy_##ctype);                  \
                                                                                 \
        while (N--) {                                                            \
            PyObject *pycomplex_value = string_to_pycomplex(                     \
                    in, has_null, default_string, allocator);                    \
                                                                                 \
            if (pycomplex_value == NULL) {                                       \
                goto fail;                                                       \
            }                                                                    \
                                                                                 \
            Py_complex complex_value = PyComplex_AsCComplex(pycomplex_value);    \
            Py_DECREF(pycomplex_value);                                          \
                                                                                 \
            if (error_converting(complex_value.real)) {                          \
                goto fail;                                                       \
            }                                                                    \
                                                                                 \
            npy_csetreal##suffix(out, (npy_##ftype) complex_value.real);         \
            npy_csetimag##suffix(out, (npy_##ftype) complex_value.imag);         \
            in += in_stride;                                                     \
            out += out_stride;                                                   \
        }                                                                        \
                                                                                 \
        NpyString_release_allocator(allocator);                                  \
        return 0;                                                                \
                                                                                 \
fail:                                                                            \
        NpyString_release_allocator(allocator);                                  \
        return -1;                                                               \
        }                                                                        \
                                                                                 \
        static PyType_Slot s2##ctype##_slots[] = {                               \
                {NPY_METH_resolve_descriptors,                                   \
                 &string_to_##ctype##_resolve_descriptors},                      \
                {NPY_METH_strided_loop, &string_to_##ctype},                     \
                {0, NULL}};                                                      \
                                                                                 \
        static char *s2##ctype##_name = "cast_StringDType_to_" #ctype;

STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(cfloat, CFLOAT)
STRING_TO_CFLOAT_CAST(cfloat, f, float)

// cfloat to string

FLOAT_TO_STRING_CAST(cfloat, cfloat, npy_cfloat)

// string to cdouble

STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(cdouble, CDOUBLE)
STRING_TO_CFLOAT_CAST(cdouble, , double)

// cdouble to string

FLOAT_TO_STRING_CAST(cdouble, cdouble, npy_cdouble)

// string to clongdouble

STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(clongdouble, CLONGDOUBLE)
STRING_TO_CFLOAT_CAST(clongdouble, l, longdouble)

// longdouble to string

FLOAT_TO_STRING_CAST(clongdouble, clongdouble, npy_clongdouble)

// string to datetime

static NPY_CASTING
string_to_datetime_timedelta_resolve_descriptors(
        PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "Casting from StringDType to datetimes without a unit "
                        "is not currently supported");
        return (NPY_CASTING)-1;
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return NPY_UNSAFE_CASTING;
}

// numpy interprets empty strings and any case combination of the string
// 'nat' as equivalent to NaT in string casts
static int
is_nat_string(const npy_static_string *s) {
    return s->size == 0 || (s->size == 3 &&
             NumPyOS_ascii_tolower(s->buf[0]) == 'n' &&
             NumPyOS_ascii_tolower(s->buf[1]) == 'a' &&
             NumPyOS_ascii_tolower(s->buf[2]) == 't');
}

static int
string_to_datetime(PyArrayMethod_Context *context, char *const data[],
                   npy_intp const dimensions[], npy_intp const strides[],
                   NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    int has_null = descr->na_object != NULL;
    int has_string_na = descr->has_string_na;
    const npy_static_string *default_string = &descr->default_string;

    npy_intp N = dimensions[0];
    char *in = data[0];
    npy_datetime *out = (npy_datetime *)data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(npy_datetime);

    npy_datetimestruct dts;
    NPY_DATETIMEUNIT in_unit = -1;
    PyArray_DatetimeMetaData in_meta = {0, 1};
    npy_bool out_special;

    _PyArray_LegacyDescr *dt_descr = (_PyArray_LegacyDescr *)context->descriptors[1];
    PyArray_DatetimeMetaData *dt_meta =
            &(((PyArray_DatetimeDTypeMetaData *)dt_descr->c_metadata)->meta);

    while (N--) {
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;
        npy_static_string s = {0, NULL};
        int is_null = NpyString_load(allocator, ps, &s);
        if (is_null == -1) {
            PyErr_SetString(
                    PyExc_MemoryError,
                    "Failed to load string in string to datetime cast");
            goto fail;
        }
        if (is_null) {
            if (has_null && !has_string_na) {
                *out = NPY_DATETIME_NAT;
                goto next_step;
            }
            s = *default_string;
        }
        if (is_nat_string(&s)) {
            *out = NPY_DATETIME_NAT;
            goto next_step;
        }

        // actually parse the datetime string
        if (NpyDatetime_ParseISO8601Datetime(
                    (const char *)s.buf, s.size, in_unit, NPY_UNSAFE_CASTING,
                    &dts, &in_meta.base, &out_special) < 0) {
            goto fail;
        }
        if (NpyDatetime_ConvertDatetimeStructToDatetime64(dt_meta, &dts, out) <
            0) {
            goto fail;
        }

    next_step:
        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);
    return 0;

fail:
    NpyString_release_allocator(allocator);
    return -1;
}

static PyType_Slot s2dt_slots[] = {
        {NPY_METH_resolve_descriptors,
         &string_to_datetime_timedelta_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_datetime},
        {0, NULL}};

static char *s2dt_name = "cast_StringDType_to_Datetime";

// datetime to string

static int
datetime_to_string(PyArrayMethod_Context *context, char *const data[],
                   npy_intp const dimensions[], npy_intp const strides[],
                   NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    npy_datetime *in = (npy_datetime *)data[0];
    char *out = data[1];

    npy_intp in_stride = strides[0] / sizeof(npy_datetime);
    npy_intp out_stride = strides[1];

    _PyArray_LegacyDescr *dt_descr = (_PyArray_LegacyDescr *)context->descriptors[0];
    PyArray_DatetimeMetaData *dt_meta =
            &(((PyArray_DatetimeDTypeMetaData *)dt_descr->c_metadata)->meta);
    // buffer passed to numpy to build datetime string
    char datetime_buf[NPY_DATETIME_MAX_ISO8601_STRLEN];

    PyArray_StringDTypeObject *sdescr = (PyArray_StringDTypeObject *)context->descriptors[1];
    int has_null = sdescr->na_object != NULL;
    npy_string_allocator *allocator = NpyString_acquire_allocator(sdescr);

    while (N--) {
        npy_packed_static_string *out_pss = (npy_packed_static_string *)out;
        if (*in == NPY_DATETIME_NAT)
        {
            if (!has_null) {
                npy_static_string os = {3, "NaT"};
                if (NpyString_pack(allocator, out_pss, os.buf, os.size) < 0) {
                    npy_gil_error(
                            PyExc_MemoryError,
                            "Failed to pack string in datetime to string "
                            "cast");
                    goto fail;
                }
            }
            else if (NpyString_pack_null(allocator, out_pss) < 0) {
                npy_gil_error(
                        PyExc_MemoryError,
                        "Failed to pack string in datetime to string cast");
                goto fail;
            }
        }
        else {
            npy_datetimestruct dts;
            if (NpyDatetime_ConvertDatetime64ToDatetimeStruct(
                        dt_meta, *in, &dts) < 0) {
                goto fail;
            }

            // zero out buffer
            memset(datetime_buf, 0, NPY_DATETIME_MAX_ISO8601_STRLEN);

            if (NpyDatetime_MakeISO8601Datetime(
                        &dts, datetime_buf, NPY_DATETIME_MAX_ISO8601_STRLEN, 0,
                        0, dt_meta->base, -1, NPY_UNSAFE_CASTING) < 0) {
                goto fail;
            }

            if (NpyString_pack(allocator, out_pss, datetime_buf,
                               strlen(datetime_buf)) < 0) {
                PyErr_SetString(PyExc_MemoryError,
                                "Failed to pack string while converting "
                                "from a datetime.");
                goto fail;
            }
        }

        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);
    return 0;

fail:
    NpyString_release_allocator(allocator);
    return -1;
}

static PyType_Slot dt2s_slots[] = {
        {NPY_METH_resolve_descriptors,
         &any_to_string_SAFE_resolve_descriptors},
        {NPY_METH_strided_loop, &datetime_to_string},
        {0, NULL}};

static char *dt2s_name = "cast_Datetime_to_StringDType";

// string to timedelta

static int
string_to_timedelta(PyArrayMethod_Context *context, char *const data[],
                    npy_intp const dimensions[], npy_intp const strides[],
                    NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    int has_null = descr->na_object != NULL;
    int has_string_na = descr->has_string_na;
    const npy_static_string *default_string = &descr->default_string;

    npy_intp N = dimensions[0];
    char *in = data[0];
    npy_timedelta *out = (npy_timedelta *)data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(npy_timedelta);

    while (N--) {
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;
        npy_static_string s = {0, NULL};
        int is_null = NpyString_load(allocator, ps, &s);
        if (is_null == -1) {
            PyErr_SetString(
                    PyExc_MemoryError,
                    "Failed to load string in string to datetime cast");
            goto fail;
        }
        if (is_null) {
            if (has_null && !has_string_na) {
                *out = NPY_DATETIME_NAT;
                goto next_step;
            }
            s = *default_string;
        }
        if (is_nat_string(&s)) {
            *out = NPY_DATETIME_NAT;
            goto next_step;
        }

        PyObject *pystr = PyUnicode_FromStringAndSize(s.buf, s.size);
        if (pystr == NULL) {
            goto fail;
        }

        // interpret as integer in base 10
        PyObject *pylong_value = PyLong_FromUnicodeObject(pystr, 10);
        Py_DECREF(pystr);
        if (pylong_value == NULL) {
            goto fail;
        }

        npy_longlong value = PyLong_AsLongLong(pylong_value);
        Py_DECREF(pylong_value);
        if (value == -1 && PyErr_Occurred()) {
            goto fail;
        }

        *out = (npy_timedelta)value;

    next_step:
        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);
    return 0;

fail:
    NpyString_release_allocator(allocator);
    return -1;
}

static PyType_Slot s2td_slots[] = {
        {NPY_METH_resolve_descriptors,
         &string_to_datetime_timedelta_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_timedelta},
        {0, NULL}};

static char *s2td_name = "cast_StringDType_to_Timedelta";

// timedelta to string

static int
timedelta_to_string(PyArrayMethod_Context *context, char *const data[],
                   npy_intp const dimensions[], npy_intp const strides[],
                   NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    npy_timedelta *in = (npy_timedelta *)data[0];
    char *out = data[1];

    npy_intp in_stride = strides[0] / sizeof(npy_timedelta);
    npy_intp out_stride = strides[1];

    PyArray_StringDTypeObject *sdescr = (PyArray_StringDTypeObject *)context->descriptors[1];
    int has_null = sdescr->na_object != NULL;
    npy_string_allocator *allocator = NpyString_acquire_allocator(sdescr);

    while (N--) {
        npy_packed_static_string *out_pss = (npy_packed_static_string *)out;
        if (*in == NPY_DATETIME_NAT)
        {
            if (!has_null) {
                npy_static_string os = {3, "NaT"};
                if (NpyString_pack(allocator, out_pss, os.buf, os.size) < 0) {
                    npy_gil_error(
                            PyExc_MemoryError,
                            "Failed to pack string in timedelta to string "
                            "cast");
                    goto fail;
                }
            }
            else if (NpyString_pack_null(allocator, out_pss) < 0) {
                npy_gil_error(
                        PyExc_MemoryError,
                        "Failed to pack string in timedelta to string cast");
                goto fail;
            }
        }
        else if (int_to_stringbuf((long long)*in, out, allocator) < 0) {
            goto fail;
        }

        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);
    return 0;

fail:
    NpyString_release_allocator(allocator);
    return -1;
}

static PyType_Slot td2s_slots[] = {
        {NPY_METH_resolve_descriptors,
         &any_to_string_SAFE_resolve_descriptors},
        {NPY_METH_strided_loop, &timedelta_to_string},
        {0, NULL}};

static char *td2s_name = "cast_Timedelta_to_StringDType";

// string to void

static NPY_CASTING
string_to_void_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                   PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                   PyArray_Descr *given_descrs[2],
                                   PyArray_Descr *loop_descrs[2],
                                   npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        // currently there's no way to determine the correct output
        // size, so set an error and bail
        PyErr_SetString(
                PyExc_TypeError,
                "Casting from StringDType to a fixed-width dtype with an "
                "unspecified size is not currently supported, specify "
                "an explicit size for the output dtype instead.");
        return (NPY_CASTING)-1;
    }
    else {
        // reject structured voids
        if (PyDataType_NAMES(given_descrs[1]) != NULL || PyDataType_SUBARRAY(given_descrs[1]) != NULL) {
            PyErr_SetString(
                    PyExc_TypeError,
                    "Casting from StringDType to a structured dtype is not "
                    "supported.");
            return (NPY_CASTING)-1;
        }
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return NPY_UNSAFE_CASTING;
}

static int
string_to_void(PyArrayMethod_Context *context, char *const data[],
               npy_intp const dimensions[], npy_intp const strides[],
               NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    int has_null = descr->na_object != NULL;
    int has_string_na = descr->has_string_na;
    const npy_static_string *default_string = &descr->default_string;
    const npy_static_string *na_name = &descr->na_name;
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];
    size_t max_out_size = context->descriptors[1]->elsize;

    while (N--) {
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;
        npy_static_string s = {0, NULL};
        if (load_nullable_string(ps, &s, has_null, has_string_na,
                                 default_string, na_name, allocator,
                                 "in string to void cast") == -1) {
            goto fail;
        }

        // This might truncate a UTF-8 character. Should we warn if that
        // happens?  UTF-8 won't be round-trippable if there is truncation
        memcpy(out, s.buf, s.size > max_out_size ? max_out_size : s.size);
        if (s.size < max_out_size) {
            memset(out + s.size, 0, (max_out_size - s.size));
        }

        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);

    return 0;

fail:

    NpyString_release_allocator(allocator);

    return -1;
}

static PyType_Slot s2v_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_void_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_void},
        {0, NULL}};

static char *s2v_name = "cast_StringDType_to_Void";

// void to string

static int
void_to_string(PyArrayMethod_Context *context, char *const data[],
               npy_intp const dimensions[], npy_intp const strides[],
               NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr *const *descrs = context->descriptors;
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)descrs[1];

    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);

    long max_in_size = descrs[0]->elsize;

    npy_intp N = dimensions[0];
    unsigned char *in = (unsigned char *)data[0];
    char *out = data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while(N--) {
        size_t out_num_bytes = utf8_buffer_size(in, max_in_size);
        if (out_num_bytes < 0) {
            npy_gil_error(PyExc_TypeError,
                          "Invalid UTF-8 bytes found, cannot convert to UTF-8");
            goto fail;
        }
        npy_static_string out_ss = {0, NULL};
        if (load_new_string((npy_packed_static_string *)out,
                            &out_ss, out_num_bytes, allocator,
                            "void to string cast") == -1) {
            goto fail;
        }
        // ignores const to fill in the buffer
        char *out_buf = (char *)out_ss.buf;
        memcpy(out_buf, in, out_num_bytes);

        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);

    return 0;

fail:

    NpyString_release_allocator(allocator);

    return -1;
}

static PyType_Slot v2s_slots[] = {{NPY_METH_resolve_descriptors,
                                   &any_to_string_SAME_KIND_resolve_descriptors},
                                  {NPY_METH_strided_loop, &void_to_string},
                                  {0, NULL}};

static char *v2s_name = "cast_Void_to_StringDType";

// string to bytes

static int
string_to_bytes(PyArrayMethod_Context *context, char *const data[],
                npy_intp const dimensions[], npy_intp const strides[],
                NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    int has_null = descr->na_object != NULL;
    int has_string_na = descr->has_string_na;
    const npy_static_string *default_string = &descr->default_string;
    const npy_static_string *na_name = &descr->na_name;
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];
    size_t max_out_size = context->descriptors[1]->elsize;

    while (N--) {
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;
        npy_static_string s = {0, NULL};
        if (load_nullable_string(ps, &s, has_null, has_string_na,
                                 default_string, na_name, allocator,
                                 "in string to bytes cast") == -1) {
            goto fail;
        }

        for (size_t i=0; i<s.size; i++) {
            if (((unsigned char *)s.buf)[i] > 127) {
                NPY_ALLOW_C_API_DEF;
                NPY_ALLOW_C_API;
                PyObject *exc = PyObject_CallFunction(
                        PyExc_UnicodeEncodeError, "ss#nns", "ascii", s.buf,
                        (Py_ssize_t)s.size, (Py_ssize_t)i, (Py_ssize_t)(i+1), "ordinal not in range(128)");
                PyErr_SetObject(PyExceptionInstance_Class(exc), exc);
                Py_DECREF(exc);
                NPY_DISABLE_C_API;
                goto fail;
            }
        }

        memcpy(out, s.buf, s.size > max_out_size ? max_out_size : s.size);
        if (s.size < max_out_size) {
            memset(out + s.size, 0, (max_out_size - s.size));
        }

        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);

    return 0;

fail:

    NpyString_release_allocator(allocator);

    return -1;
}

static PyType_Slot s2bytes_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_fixed_width_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_bytes},
        {0, NULL}};

static char *s2bytes_name = "cast_StringDType_to_Bytes";

// bytes to string

static int
bytes_to_string(PyArrayMethod_Context *context, char *const data[],
                npy_intp const dimensions[], npy_intp const strides[],
                NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr *const *descrs = context->descriptors;
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)descrs[1];

    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);

    size_t max_in_size = descrs[0]->elsize;

    npy_intp N = dimensions[0];
    unsigned char *in = (unsigned char *)data[0];
    char *out = data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while(N--) {
        size_t out_num_bytes = max_in_size;

        // ignore trailing nulls
        while (out_num_bytes > 0 && in[out_num_bytes - 1] == 0) {
            out_num_bytes--;
        }

        npy_static_string out_ss = {0, NULL};
        if (load_new_string((npy_packed_static_string *)out,
                            &out_ss, out_num_bytes, allocator,
                            "void to string cast") == -1) {
            goto fail;
        }

        // ignores const to fill in the buffer
        char *out_buf = (char *)out_ss.buf;
        memcpy(out_buf, in, out_num_bytes);

        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);

    return 0;

fail:

    NpyString_release_allocator(allocator);

    return -1;
}


static PyType_Slot bytes2s_slots[] = {
        {NPY_METH_resolve_descriptors, &any_to_string_SAME_KIND_resolve_descriptors},
        {NPY_METH_strided_loop, &bytes_to_string},
        {0, NULL}};

static char *bytes2s_name = "cast_Bytes_to_StringDType";


PyArrayMethod_Spec *
get_cast_spec(const char *name, NPY_CASTING casting,
              NPY_ARRAYMETHOD_FLAGS flags, PyArray_DTypeMeta **dtypes,
              PyType_Slot *slots)
{
    PyArrayMethod_Spec *ret = PyMem_Malloc(sizeof(PyArrayMethod_Spec));

    ret->name = name;
    ret->nin = 1;
    ret->nout = 1;
    ret->casting = casting;
    ret->flags = flags;
    ret->dtypes = dtypes;
    ret->slots = slots;

    return ret;
}

PyArray_DTypeMeta **
get_dtypes(PyArray_DTypeMeta *dt1, PyArray_DTypeMeta *dt2)
{
    PyArray_DTypeMeta **ret = PyMem_Malloc(2 * sizeof(PyArray_DTypeMeta *));

    ret[0] = dt1;
    ret[1] = dt2;

    return ret;
}

PyArrayMethod_Spec **
get_casts()
{
    char *t2t_name = s2s_name;

    PyArray_DTypeMeta **t2t_dtypes =
            get_dtypes(&PyArray_StringDType,
                       &PyArray_StringDType);

    PyArrayMethod_Spec *ThisToThisCastSpec =
            get_cast_spec(t2t_name, NPY_UNSAFE_CASTING,
                          NPY_METH_SUPPORTS_UNALIGNED, t2t_dtypes, s2s_slots);

    int num_casts = 43;

#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
    num_casts += 4;
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
    num_casts += 4;
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    num_casts += 4;
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
    num_casts += 4;
#endif

    PyArray_DTypeMeta **u2s_dtypes = get_dtypes(
            &PyArray_UnicodeDType, &PyArray_StringDType);

    PyArrayMethod_Spec *UnicodeToStringCastSpec = get_cast_spec(
            u2s_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            u2s_dtypes, u2s_slots);

    PyArray_DTypeMeta **s2u_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_UnicodeDType);

    PyArrayMethod_Spec *StringToUnicodeCastSpec = get_cast_spec(
            s2u_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2u_dtypes, s2u_slots);

    PyArray_DTypeMeta **s2b_dtypes =
            get_dtypes(&PyArray_StringDType, &PyArray_BoolDType);

    PyArrayMethod_Spec *StringToBoolCastSpec = get_cast_spec(
            s2b_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2b_dtypes, s2b_slots);

    PyArray_DTypeMeta **b2s_dtypes =
            get_dtypes(&PyArray_BoolDType, &PyArray_StringDType);

    PyArrayMethod_Spec *BoolToStringCastSpec = get_cast_spec(
            b2s_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            b2s_dtypes, b2s_slots);

    DTYPES_AND_CAST_SPEC(i8, Int8)
    DTYPES_AND_CAST_SPEC(i16, Int16)
    DTYPES_AND_CAST_SPEC(i32, Int32)
    DTYPES_AND_CAST_SPEC(i64, Int64)
    DTYPES_AND_CAST_SPEC(u8, UInt8)
    DTYPES_AND_CAST_SPEC(u16, UInt16)
    DTYPES_AND_CAST_SPEC(u32, UInt32)
    DTYPES_AND_CAST_SPEC(u64, UInt64)
#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
    DTYPES_AND_CAST_SPEC(byte, Byte)
    DTYPES_AND_CAST_SPEC(ubyte, UByte)
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
    DTYPES_AND_CAST_SPEC(short, Short)
    DTYPES_AND_CAST_SPEC(ushort, UShort)
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    DTYPES_AND_CAST_SPEC(int, Int)
    DTYPES_AND_CAST_SPEC(uint, UInt)
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
    DTYPES_AND_CAST_SPEC(longlong, LongLong)
    DTYPES_AND_CAST_SPEC(ulonglong, ULongLong)
#endif

    DTYPES_AND_CAST_SPEC(f64, Double)
    DTYPES_AND_CAST_SPEC(f32, Float)
    DTYPES_AND_CAST_SPEC(f16, Half)

    PyArray_DTypeMeta **s2dt_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_DatetimeDType);

    PyArrayMethod_Spec *StringToDatetimeCastSpec = get_cast_spec(
            s2dt_name, NPY_UNSAFE_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI,
            s2dt_dtypes, s2dt_slots);

    PyArray_DTypeMeta **dt2s_dtypes = get_dtypes(
            &PyArray_DatetimeDType, &PyArray_StringDType);

    PyArrayMethod_Spec *DatetimeToStringCastSpec = get_cast_spec(
            dt2s_name, NPY_SAFE_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI,
            dt2s_dtypes, dt2s_slots);

    PyArray_DTypeMeta **s2td_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_TimedeltaDType);

    PyArrayMethod_Spec *StringToTimedeltaCastSpec = get_cast_spec(
            s2td_name, NPY_UNSAFE_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI,
            s2td_dtypes, s2td_slots);

    PyArray_DTypeMeta **td2s_dtypes = get_dtypes(
            &PyArray_TimedeltaDType, &PyArray_StringDType);

    PyArrayMethod_Spec *TimedeltaToStringCastSpec = get_cast_spec(
            td2s_name, NPY_SAFE_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI,
            td2s_dtypes, td2s_slots);

    PyArray_DTypeMeta **s2ld_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_LongDoubleDType);

    PyArrayMethod_Spec *StringToLongDoubleCastSpec = get_cast_spec(
            s2ld_name, NPY_UNSAFE_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2ld_dtypes, s2ld_slots);

    PyArray_DTypeMeta **ld2s_dtypes = get_dtypes(
            &PyArray_LongDoubleDType, &PyArray_StringDType);

    PyArrayMethod_Spec *LongDoubleToStringCastSpec = get_cast_spec(
            ld2s_name, NPY_SAFE_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI,
            ld2s_dtypes, ld2s_slots);

    DTYPES_AND_CAST_SPEC(cfloat, CFloat)
    DTYPES_AND_CAST_SPEC(cdouble, CDouble)
    DTYPES_AND_CAST_SPEC(clongdouble, CLongDouble)

    PyArray_DTypeMeta **s2v_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_VoidDType);

    PyArrayMethod_Spec *StringToVoidCastSpec = get_cast_spec(
            s2v_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2v_dtypes, s2v_slots);

    PyArray_DTypeMeta **v2s_dtypes = get_dtypes(
            &PyArray_VoidDType, &PyArray_StringDType);

    PyArrayMethod_Spec *VoidToStringCastSpec = get_cast_spec(
            v2s_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            v2s_dtypes, v2s_slots);

    PyArray_DTypeMeta **s2bytes_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_BytesDType);

    PyArrayMethod_Spec *StringToBytesCastSpec = get_cast_spec(
            s2bytes_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2bytes_dtypes, s2bytes_slots);

    PyArray_DTypeMeta **bytes2s_dtypes = get_dtypes(
            &PyArray_BytesDType, &PyArray_StringDType);

    PyArrayMethod_Spec *BytesToStringCastSpec = get_cast_spec(
            bytes2s_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            bytes2s_dtypes, bytes2s_slots);

    PyArrayMethod_Spec **casts =
            PyMem_Malloc((num_casts + 1) * sizeof(PyArrayMethod_Spec *));

    int cast_i = 0;

    casts[cast_i++] = ThisToThisCastSpec;
    casts[cast_i++] = UnicodeToStringCastSpec;
    casts[cast_i++] = StringToUnicodeCastSpec;
    casts[cast_i++] = StringToBoolCastSpec;
    casts[cast_i++] = BoolToStringCastSpec;
    casts[cast_i++] = StringToInt8CastSpec;
    casts[cast_i++] = Int8ToStringCastSpec;
    casts[cast_i++] = StringToInt16CastSpec;
    casts[cast_i++] = Int16ToStringCastSpec;
    casts[cast_i++] = StringToInt32CastSpec;
    casts[cast_i++] = Int32ToStringCastSpec;
    casts[cast_i++] = StringToInt64CastSpec;
    casts[cast_i++] = Int64ToStringCastSpec;
    casts[cast_i++] = StringToUInt8CastSpec;
    casts[cast_i++] = UInt8ToStringCastSpec;
    casts[cast_i++] = StringToUInt16CastSpec;
    casts[cast_i++] = UInt16ToStringCastSpec;
    casts[cast_i++] = StringToUInt32CastSpec;
    casts[cast_i++] = UInt32ToStringCastSpec;
    casts[cast_i++] = StringToUInt64CastSpec;
    casts[cast_i++] = UInt64ToStringCastSpec;
#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
    casts[cast_i++] = StringToByteCastSpec;
    casts[cast_i++] = ByteToStringCastSpec;
    casts[cast_i++] = StringToUByteCastSpec;
    casts[cast_i++] = UByteToStringCastSpec;
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
    casts[cast_i++] = StringToShortCastSpec;
    casts[cast_i++] = ShortToStringCastSpec;
    casts[cast_i++] = StringToUShortCastSpec;
    casts[cast_i++] = UShortToStringCastSpec;
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    casts[cast_i++] = StringToIntCastSpec;
    casts[cast_i++] = IntToStringCastSpec;
    casts[cast_i++] = StringToUIntCastSpec;
    casts[cast_i++] = UIntToStringCastSpec;
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
    casts[cast_i++] = StringToLongLongCastSpec;
    casts[cast_i++] = LongLongToStringCastSpec;
    casts[cast_i++] = StringToULongLongCastSpec;
    casts[cast_i++] = ULongLongToStringCastSpec;
#endif
    casts[cast_i++] = StringToDoubleCastSpec;
    casts[cast_i++] = DoubleToStringCastSpec;
    casts[cast_i++] = StringToFloatCastSpec;
    casts[cast_i++] = FloatToStringCastSpec;
    casts[cast_i++] = StringToHalfCastSpec;
    casts[cast_i++] = HalfToStringCastSpec;
    casts[cast_i++] = StringToDatetimeCastSpec;
    casts[cast_i++] = DatetimeToStringCastSpec;
    casts[cast_i++] = StringToTimedeltaCastSpec;
    casts[cast_i++] = TimedeltaToStringCastSpec;
    casts[cast_i++] = StringToLongDoubleCastSpec;
    casts[cast_i++] = LongDoubleToStringCastSpec;
    casts[cast_i++] = StringToCFloatCastSpec;
    casts[cast_i++] = CFloatToStringCastSpec;
    casts[cast_i++] = StringToCDoubleCastSpec;
    casts[cast_i++] = CDoubleToStringCastSpec;
    casts[cast_i++] = StringToCLongDoubleCastSpec;
    casts[cast_i++] = CLongDoubleToStringCastSpec;
    casts[cast_i++] = StringToVoidCastSpec;
    casts[cast_i++] = VoidToStringCastSpec;
    casts[cast_i++] = StringToBytesCastSpec;
    casts[cast_i++] = BytesToStringCastSpec;
    casts[cast_i++] = NULL;

    assert(casts[num_casts] == NULL);
    assert(cast_i == num_casts + 1);

    return casts;
}
