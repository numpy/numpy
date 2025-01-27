#include <cmath>
#include <type_traits>

#include "numpy/npy_common.h"
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

#include "casts.h"

// Get a c string representation of a type number.
static const char *
typenum_to_cstr(NPY_TYPES typenum) {
    switch (typenum) {
        case NPY_BOOL:
            return "bool";
        case NPY_BYTE:
            return "byte";
        case NPY_UBYTE:
            return "unsigned byte";
        case NPY_SHORT:
            return "short";
        case NPY_USHORT:
            return "unsigned short";
        case NPY_INT:
            return "int";
        case NPY_UINT:
            return "unsigned int";
        case NPY_LONG:
            return "long";
        case NPY_ULONG:
            return "unsigned long";
        case NPY_LONGLONG:
            return "long long";
        case NPY_ULONGLONG:
            return "unsigned long long";
        case NPY_HALF:
            return "half";
        case NPY_FLOAT:
            return "float";
        case NPY_DOUBLE:
            return "double";
        case NPY_LONGDOUBLE:
            return "long double";
        case NPY_CFLOAT:
            return "complex float";
        case NPY_CDOUBLE:
            return "complex double";
        case NPY_CLONGDOUBLE:
            return "complex long double";
        case NPY_OBJECT:
            return "object";
        case NPY_STRING:
            return "string";
        case NPY_UNICODE:
            return "unicode";
        case NPY_VOID:
            return "void";
        case NPY_DATETIME:
            return "datetime";
        case NPY_TIMEDELTA:
            return "timedelta";
        case NPY_CHAR:
            return "char";
        case NPY_NOTYPE:
            return "no type";
        case NPY_USERDEF:
            return "user defined";
        case NPY_VSTRING:
            return "vstring";
        default:
            return "unknown";
    }
}

static PyArray_DTypeMeta **
get_dtypes(PyArray_DTypeMeta *dt1, PyArray_DTypeMeta *dt2)
{
    // If either argument is NULL, an error has happened; return NULL.
    if ((dt1 == NULL) || (dt2 == NULL)) {
        return NULL;
    }
    PyArray_DTypeMeta **ret = (PyArray_DTypeMeta **)PyMem_Malloc(2 * sizeof(PyArray_DTypeMeta *));
    if (ret == NULL) {
        return reinterpret_cast<PyArray_DTypeMeta **>(PyErr_NoMemory());
    }

    ret[0] = dt1;
    ret[1] = dt2;

    return ret;
}


template<NPY_CASTING safety>
static NPY_CASTING
any_to_string_resolve_descriptors(
        PyObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        PyArray_Descr *new_instance =
                (PyArray_Descr *)new_stringdtype_instance(NULL, 1);
        if (new_instance == NULL) {
            return (NPY_CASTING)-1;
        }
        loop_descrs[1] = new_instance;
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return safety;
}


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
        {NPY_METH_resolve_descriptors, (void *)&string_to_string_resolve_descriptors},
        {NPY_METH_strided_loop, (void *)&string_to_string},
        {NPY_METH_unaligned_strided_loop, (void *)&string_to_string},
        {0, NULL}};

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
                                   (void *)&any_to_string_resolve_descriptors<NPY_SAME_KIND_CASTING>},
                                  {NPY_METH_strided_loop, (void *)&unicode_to_string},
                                  {0, NULL}};

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
                     const char *context)
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
            for (size_t i=0; i < max_out_size; i++) {
                out[i] = (Py_UCS4)0;
            }
        }
        else {
            size_t i = 0;
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
        {NPY_METH_resolve_descriptors, (void *)&string_to_fixed_width_resolve_descriptors},
        {NPY_METH_strided_loop, (void *)&string_to_unicode},
        {0, NULL}};

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
        {NPY_METH_resolve_descriptors, (void *)&string_to_bool_resolve_descriptors},
        {NPY_METH_strided_loop, (void *)&string_to_bool},
        {0, NULL}};

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
        const char *ret_val = NULL;
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
                                   (void *)&any_to_string_resolve_descriptors<NPY_SAFE_CASTING>},
                                  {NPY_METH_strided_loop, (void *)&bool_to_string},
                                  {0, NULL}};

// casts between string and (u)int dtypes


static int
load_non_nullable_string(char *in, int has_null, const npy_static_string *default_string,
                         npy_static_string *string_to_load, npy_string_allocator *allocator,
                         int has_gil)
{
    const npy_packed_static_string *ps = (npy_packed_static_string *)in;
    int isnull = NpyString_load(allocator, ps, string_to_load);
    if (isnull == -1) {
        const char *msg = "Failed to load string for conversion to a non-nullable type";
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
            const char *msg = "Arrays with missing data cannot be converted to a non-nullable type";
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

template<typename NpyLongType>
static npy_longlong
stringbuf_to_int(char *in, NpyLongType *value, int has_null,
                 const npy_static_string *default_string,
                 npy_string_allocator *allocator)
{
    PyObject *pylong_value =
            string_to_pylong(in, has_null, default_string, allocator);
    if (pylong_value == NULL) {
        return -1;
    }

    if constexpr (std::is_same_v<NpyLongType, npy_ulonglong>) {
        *value = PyLong_AsUnsignedLongLong(pylong_value);
        if (*value == (unsigned long long)-1 && PyErr_Occurred()) {
            goto fail;
        }
    } else {
        *value = PyLong_AsLongLong(pylong_value);
        if (*value == -1 && PyErr_Occurred()) {
            goto fail;
        }
    }
    Py_DECREF(pylong_value);
    return 0;

fail:
    Py_DECREF(pylong_value);
    return -1;
}

// steals reference to obj
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

template<NPY_TYPES typenum>
static NPY_CASTING
string_to_int_resolve_descriptors(
    PyObject *NPY_UNUSED(self),
    PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
    PyArray_Descr *given_descrs[2],
    PyArray_Descr *loop_descrs[2],
    npy_intp *NPY_UNUSED(view_offset)
) {
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = PyArray_DescrNewFromType(typenum);
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return NPY_UNSAFE_CASTING;
}

// Example template parameters:
// NpyType: npy_int8
// NpyLongType: npy_longlong
// typenum: NPY_BYTE
template <typename NpyType, typename NpyLongType, NPY_TYPES typenum>
static int
string_to_int(
    PyArrayMethod_Context * context,
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    NpyAuxData *NPY_UNUSED(auxdata)
) {
    PyArray_StringDTypeObject *descr =
            ((PyArray_StringDTypeObject *)context->descriptors[0]);
    npy_string_allocator *allocator =
            NpyString_acquire_allocator(descr);
    int has_null = descr->na_object != NULL;
    const npy_static_string *default_string = &descr->default_string;

    npy_intp N = dimensions[0];
    char *in = data[0];
    NpyType *out = (NpyType *)data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(NpyType);

    while (N--) {
        NpyLongType value;
        if (stringbuf_to_int<NpyLongType>(in, &value, has_null, default_string, allocator) != 0) {
            npy_gil_error(PyExc_RuntimeError, "Encountered problem converting string dtype to integer dtype.");
            goto fail;
        }
        *out = (NpyType)value;

        // Cast back to NpyLongType to check for out-of-bounds errors
        if (static_cast<NpyLongType>(*out) != value) {
            // out of bounds, raise error following NEP 50 behavior
            const char *errmsg = NULL;
            if constexpr (std::is_same_v<NpyLongType, npy_ulonglong>) {
                errmsg = "Integer %llu is out of bounds for %s";
            } else if constexpr (std::is_same_v<NpyLongType, npy_longlong>) {
                errmsg = "Integer %lli is out of bounds for %s";
            } else {
                errmsg = "Unrecognized integer type %i is out of bounds for %s";
            }
            npy_gil_error(PyExc_OverflowError, errmsg, value, typenum_to_cstr(typenum));
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

template<typename NpyType, typename NpyLongType, NPY_TYPES typenum>
static PyType_Slot s2int_slots[] = {
    {NPY_METH_resolve_descriptors, (void *)&string_to_int_resolve_descriptors<typenum>},
    {NPY_METH_strided_loop, (void *)&string_to_int<NpyType, NpyLongType, typenum>},
    {0, NULL}
};

static const char *
make_s2type_name(NPY_TYPES typenum) {
    const char *prefix = "cast_StringDType_to_";
    size_t plen = strlen(prefix);

    const char *type_name = typenum_to_cstr(typenum);
    size_t nlen = strlen(type_name);

    char *buf = (char *)PyMem_RawCalloc(sizeof(char), plen + nlen + 1);
    if (buf == NULL) {
        npy_gil_error(PyExc_MemoryError, "Failed allocate memory for cast");
        return NULL;
    }

    // memcpy instead of strcpy to avoid stringop-truncation warning, since
    // we are not including the trailing null character
    memcpy(buf, prefix, plen);
    strncat(buf, type_name, nlen);
    return buf;
}

static const char *
make_type2s_name(NPY_TYPES typenum) {
    const char *prefix = "cast_";
    size_t plen = strlen(prefix);

    const char *type_name = typenum_to_cstr(typenum);
    size_t nlen = strlen(type_name);

    const char *suffix = "_to_StringDType";
    size_t slen = strlen(suffix);

    char *buf = (char *)PyMem_RawCalloc(sizeof(char), plen + nlen + slen + 1);

    // memcpy instead of strcpy to avoid stringop-truncation warning, since
    // we are not including the trailing null character
    memcpy(buf, prefix, plen);
    strncat(buf, type_name, nlen);
    strncat(buf, suffix, slen);
    return buf;
}


static int
int_to_stringbuf(long long in, char *out, npy_string_allocator *allocator)
{
    PyObject *pylong_val = PyLong_FromLongLong(in);
    // steals reference to pylong_val
    return pyobj_to_string(pylong_val, out, allocator);
}

static int
int_to_stringbuf(unsigned long long in, char *out, npy_string_allocator *allocator)
{
    PyObject *pylong_val = PyLong_FromUnsignedLongLong(in);
    // steals reference to pylong_val
    return pyobj_to_string(pylong_val, out, allocator);
}

template<typename NpyType, typename TClongType, NPY_TYPES typenum>
static int
type_to_string(
    PyArrayMethod_Context *context,
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    NpyAuxData *NPY_UNUSED(auxdata)
) {
    npy_intp N = dimensions[0];
    NpyType *in = (NpyType *)data[0];
    char *out = data[1];

    npy_intp in_stride = strides[0] / sizeof(NpyType);
    npy_intp out_stride = strides[1];

    PyArray_StringDTypeObject *descr =
            (PyArray_StringDTypeObject *)context->descriptors[1];
    npy_string_allocator *allocator =
            NpyString_acquire_allocator(descr);

    while (N--) {
        if (int_to_stringbuf((TClongType)*in, out, allocator) != 0) {
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

template<typename NpyType, typename TClongType, NPY_TYPES typenum>
static PyType_Slot int2s_slots[] = {
        {NPY_METH_resolve_descriptors,
         (void *)&any_to_string_resolve_descriptors<NPY_SAFE_CASTING>},
        {NPY_METH_strided_loop, (void *)&type_to_string<NpyType, TClongType, typenum>},
        {0, NULL}};

static PyArray_DTypeMeta **
get_s2type_dtypes(NPY_TYPES typenum) {
    return get_dtypes(&PyArray_StringDType, typenum_to_dtypemeta(typenum));
}

template<typename NpyType, typename NpyLongType, NPY_TYPES typenum>
static PyArrayMethod_Spec *
getStringToIntCastSpec() {
    return get_cast_spec(
        make_s2type_name(typenum),
        NPY_UNSAFE_CASTING,
        NPY_METH_REQUIRES_PYAPI,
        get_s2type_dtypes(typenum),
        s2int_slots<NpyType, NpyLongType, typenum>
    );
}


static PyArray_DTypeMeta **
get_type2s_dtypes(NPY_TYPES typenum) {
    return get_dtypes(typenum_to_dtypemeta(typenum), &PyArray_StringDType);
}

template<typename NpyType, typename TClongType, NPY_TYPES typenum>
static PyArrayMethod_Spec *
getIntToStringCastSpec() {
    return get_cast_spec(
        make_type2s_name(typenum),
        NPY_SAFE_CASTING,
        NPY_METH_REQUIRES_PYAPI,
        get_type2s_dtypes(typenum),
        int2s_slots<NpyType, TClongType, typenum>
    );
}

static PyObject *
string_to_pyfloat(
    char *in,
    int has_null,
    const npy_static_string *default_string,
    npy_string_allocator *allocator
) {
    PyObject *val_obj = non_nullable_string_to_pystring(
            in, has_null, default_string, allocator);
    if (val_obj == NULL) {
        return NULL;
    }
    PyObject *pyfloat_value = PyFloat_FromString(val_obj);
    Py_DECREF(val_obj);
    return pyfloat_value;
}

template<
    typename NpyType,
    NPY_TYPES typenum,
    bool (*npy_is_inf)(NpyType) = nullptr,
    bool (*double_is_inf)(double) = nullptr,
    NpyType (*double_to_float)(double) = nullptr
>
static int
string_to_float(
    PyArrayMethod_Context * context,
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    NpyAuxData *NPY_UNUSED(auxdata)
) {
    PyArray_StringDTypeObject *descr =
            (PyArray_StringDTypeObject *)context->descriptors[0];
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    int has_null = (descr->na_object != NULL);
    const npy_static_string *default_string = &descr->default_string;

    npy_intp N = dimensions[0];
    char *in = data[0];
    NpyType *out = (NpyType *)data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(NpyType);

    while (N--) {
        PyObject *pyfloat_value = string_to_pyfloat(
            in, has_null, default_string, allocator
        );
        if (pyfloat_value == NULL) {
            goto fail;
        }
        double dval = PyFloat_AS_DOUBLE(pyfloat_value);
        Py_DECREF(pyfloat_value);
        NpyType fval = (double_to_float)(dval);

        if (NPY_UNLIKELY(npy_is_inf(fval) && !(double_is_inf(dval)))) {
            if (PyUFunc_GiveFloatingpointErrors("cast",
                                                NPY_FPE_OVERFLOW) < 0) {
                goto fail;
            }
        }

        *out = fval;

        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);
    return 0;
fail:
    NpyString_release_allocator(allocator);
    return -1;
}

// Since PyFloat is already 64bit, there's no way it can overflow, making
// that check unnecessary - which is why we have a specialized template
// for this case and not the others.
template<>
int
string_to_float<npy_float64, NPY_DOUBLE>(
    PyArrayMethod_Context * context,
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    NpyAuxData *NPY_UNUSED(auxdata)
) {
    PyArray_StringDTypeObject *descr =
            (PyArray_StringDTypeObject *)context->descriptors[0];
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    int has_null = (descr->na_object != NULL);
    const npy_static_string *default_string = &descr->default_string;

    npy_intp N = dimensions[0];
    char *in = data[0];
    npy_float64 *out = (npy_float64 *)data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(npy_float64);

    while (N--) {
        PyObject *pyfloat_value = string_to_pyfloat(
            in, has_null, default_string, allocator
        );
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

// Long double types do not fit in a (64-bit) PyFloat, so we handle this
// case specially here.
template<>
int
string_to_float<npy_longdouble, NPY_LONGDOUBLE>(
    PyArrayMethod_Context *context,
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    NpyAuxData *NPY_UNUSED(auxdata)
) {
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
        char *buf = (char *)PyMem_RawMalloc(s.size + 1);
        memcpy(buf, s.buf, s.size);
        buf[s.size] = '\0';

        char *end = NULL;
        errno = 0;
        npy_longdouble longdouble_value = NumPyOS_ascii_strtold(buf, &end);

        if (errno == ERANGE) {
            /* strtold returns INFINITY of the correct sign. */
            if (
                npy_gil_warning(
                    PyExc_RuntimeWarning,
                    1,
                    "overflow encountered in conversion from string"
                ) < 0
            ) {
                PyMem_RawFree(buf);
                goto fail;
            }
        }
        else if (errno || end == buf || *end) {
            npy_gil_error(PyExc_ValueError,
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

template<NPY_TYPES typenum>
static NPY_CASTING
string_to_float_resolve_descriptors(
    PyObject *NPY_UNUSED(self),
    PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
    PyArray_Descr *given_descrs[2],
    PyArray_Descr *loop_descrs[2],
    npy_intp *NPY_UNUSED(view_offset)
) {
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = PyArray_DescrNewFromType(typenum);
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return NPY_UNSAFE_CASTING;
}

template<
    typename NpyType,
    NPY_TYPES typenum,
    bool (*npy_is_inf)(NpyType) = nullptr,
    bool (*double_is_inf)(double) = nullptr,
    NpyType (*double_to_float)(double) = nullptr
>
static PyType_Slot s2float_slots[] = {
        {NPY_METH_resolve_descriptors, (void *)&string_to_float_resolve_descriptors<typenum>},
        {NPY_METH_strided_loop, (void *)&string_to_float<NpyType, typenum, npy_is_inf, double_is_inf, double_to_float>},
        {0, NULL}};

template<typename NpyType>
static int
float_to_string(
    PyArrayMethod_Context *context,
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    NpyAuxData *NPY_UNUSED(auxdata)
) {
    npy_intp N = dimensions[0];
    NpyType *in = (NpyType *)data[0];
    char *out = data[1];
    PyArray_Descr *float_descr = context->descriptors[0];

    npy_intp in_stride = strides[0] / sizeof(NpyType);
    npy_intp out_stride = strides[1];

    PyArray_StringDTypeObject *descr =
            (PyArray_StringDTypeObject *)context->descriptors[1];
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    // borrowed reference
    PyObject *na_object = descr->na_object;

    while (N--) {
        PyObject *scalar_val = PyArray_Scalar(in, float_descr, NULL);
        if (descr->has_nan_na) {
            // check for case when scalar_val is the na_object and store a null string
            int na_cmp = na_eq_cmp(scalar_val, na_object);
            if (na_cmp < 0) {
                Py_DECREF(scalar_val);
                goto fail;
            }
            if (na_cmp) {
                Py_DECREF(scalar_val);
                if (NpyString_pack_null(allocator, (npy_packed_static_string *)out) < 0) {
                    PyErr_SetString(PyExc_MemoryError,
                                    "Failed to pack null string during float "
                                    "to string cast");
                    goto fail;
                }
                goto next_step;
            }
        }
        // steals reference to scalar_val
        if (pyobj_to_string(scalar_val, out, allocator) == -1) {
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

template <typename NpyType>
static PyType_Slot float2s_slots [] = {
    {NPY_METH_resolve_descriptors, (void *)&any_to_string_resolve_descriptors<NPY_SAFE_CASTING>},
    {NPY_METH_strided_loop, (void *)&float_to_string<NpyType>},
    {0, NULL}
};

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

template <
    typename NpyComplexType,
    typename NpyFloatType,
    void npy_csetrealfunc(NpyComplexType*, NpyFloatType),
    void npy_csetimagfunc(NpyComplexType*, NpyFloatType)
>
static int
string_to_complex_float(
    PyArrayMethod_Context *context,
    char *const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    NpyAuxData *NPY_UNUSED(auxdata)
) {
    PyArray_StringDTypeObject *descr =
            (PyArray_StringDTypeObject *)context->descriptors[0];
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    int has_null = descr->na_object != NULL;
    const npy_static_string *default_string = &descr->default_string;
    npy_intp N = dimensions[0];
    char *in = data[0];
    NpyComplexType *out = (NpyComplexType *)data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(NpyComplexType);

    while (N--) {
        PyObject *pycomplex_value = string_to_pycomplex(
                in, has_null, default_string, allocator);

        if (pycomplex_value == NULL) {
            goto fail;
        }

        Py_complex complex_value = PyComplex_AsCComplex(pycomplex_value);
        Py_DECREF(pycomplex_value);

        if (error_converting(complex_value.real)) {
            goto fail;
        }

        npy_csetrealfunc(out, (NpyFloatType) complex_value.real);
        npy_csetimagfunc(out, (NpyFloatType) complex_value.real);
        in += in_stride;
        out += out_stride;
    }

    NpyString_release_allocator(allocator);
    return 0;

fail:
    NpyString_release_allocator(allocator);
    return -1;
}

template <
    typename NpyComplexType,
    typename NpyFloatType,
    NPY_TYPES typenum,
    void npy_csetrealfunc(NpyComplexType*, NpyFloatType),
    void npy_csetimagfunc(NpyComplexType*, NpyFloatType)
>
static PyType_Slot s2ctype_slots[] = {
    {NPY_METH_resolve_descriptors, (void *)&string_to_float_resolve_descriptors<typenum>},
    {NPY_METH_strided_loop, (void *)&string_to_complex_float<NpyComplexType, NpyFloatType, npy_csetrealfunc, npy_csetimagfunc>},
    {0, NULL}
};


template <
    typename NpyComplexType,
    typename NpyFloatType,
    NPY_TYPES typenum,
    void npy_csetrealfunc(NpyComplexType*, NpyFloatType),
    void npy_csetimagfunc(NpyComplexType*, NpyFloatType)
>
static PyArrayMethod_Spec *
getStringToComplexCastSpec() {
    return get_cast_spec(
        make_s2type_name(typenum),
        NPY_UNSAFE_CASTING,
        NPY_METH_REQUIRES_PYAPI,
        get_s2type_dtypes(typenum),
        s2ctype_slots<NpyComplexType, NpyFloatType, typenum, npy_csetrealfunc, npy_csetimagfunc>
    );
}

template<
    typename NpyType,
    NPY_TYPES typenum,
    bool (*npy_is_inf)(NpyType) = nullptr,
    bool (*double_is_inf)(double) = nullptr,
    NpyType (*double_to_float)(double) = nullptr,
    NPY_ARRAYMETHOD_FLAGS flags = NPY_METH_REQUIRES_PYAPI
>
static PyArrayMethod_Spec *
getStringToFloatCastSpec(
) {
    return get_cast_spec(
        make_s2type_name(typenum),
        NPY_UNSAFE_CASTING,
        flags,
        get_s2type_dtypes(typenum),
        s2float_slots<NpyType, typenum, npy_is_inf, double_is_inf, double_to_float>
    );
}

template<
    typename NpyType,
    NPY_TYPES typenum,
    NPY_ARRAYMETHOD_FLAGS flags = NPY_METH_REQUIRES_PYAPI
>
static PyArrayMethod_Spec *
getFloatToStringCastSpec() {
    return get_cast_spec(
        make_type2s_name(typenum),
        NPY_SAFE_CASTING,
        flags,
        get_type2s_dtypes(typenum),
        float2s_slots<NpyType>
    );
}

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
    NPY_DATETIMEUNIT in_unit = NPY_FR_ERROR;
    PyArray_DatetimeMetaData in_meta = {NPY_FR_Y, 1};
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
    {NPY_METH_resolve_descriptors, (void *)&string_to_datetime_timedelta_resolve_descriptors},
    {NPY_METH_strided_loop, (void *)&string_to_datetime},
    {0, NULL}
};

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
    {NPY_METH_resolve_descriptors, (void *)&any_to_string_resolve_descriptors<NPY_SAFE_CASTING>},
    {NPY_METH_strided_loop, (void *)&datetime_to_string},
    {0, NULL}
};

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
                in += in_stride;
                out += out_stride;
                continue;
            }
            s = *default_string;
        }
        if (is_nat_string(&s)) {
            *out = NPY_DATETIME_NAT;
            in += in_stride;
            out += out_stride;
            continue;
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
    {NPY_METH_resolve_descriptors, (void *)&string_to_datetime_timedelta_resolve_descriptors},
    {NPY_METH_strided_loop, (void *)&string_to_timedelta},
    {0, NULL}
};

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
    {NPY_METH_resolve_descriptors, (void *)&any_to_string_resolve_descriptors<NPY_SAFE_CASTING>},
    {NPY_METH_strided_loop, (void *)&timedelta_to_string},
    {0, NULL}
};

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
    {NPY_METH_resolve_descriptors, (void *)&string_to_void_resolve_descriptors},
    {NPY_METH_strided_loop, (void *)&string_to_void},
    {0, NULL}
};

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

static PyType_Slot v2s_slots[] = {
    {NPY_METH_resolve_descriptors, (void *)&any_to_string_resolve_descriptors<NPY_SAME_KIND_CASTING>},
    {NPY_METH_strided_loop, (void *)&void_to_string},
    {0, NULL}
};

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
                PyObject *str = PyUnicode_FromStringAndSize(s.buf, s.size);

                if (str == NULL) {
                    PyErr_SetString(
                        PyExc_UnicodeEncodeError, "Invalid character encountered during unicode encoding."
                    );
                    goto fail;
                }

                PyObject *exc = PyObject_CallFunction(
                    PyExc_UnicodeEncodeError,
                    "sOnns",
                    "ascii",
                    str,
                    (Py_ssize_t)i,
                    (Py_ssize_t)(i+1),
                    "ordinal not in range(128)"
                );

                if (exc == NULL) {
                    Py_DECREF(str);
                    goto fail;
                }

                PyErr_SetObject(PyExceptionInstance_Class(exc), exc);
                Py_DECREF(exc);
                Py_DECREF(str);
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
    {NPY_METH_resolve_descriptors, (void *)&string_to_fixed_width_resolve_descriptors},
    {NPY_METH_strided_loop, (void *)&string_to_bytes},
    {0, NULL}
};

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
    {NPY_METH_resolve_descriptors, (void *)&any_to_string_resolve_descriptors<NPY_SAME_KIND_CASTING>},
    {NPY_METH_strided_loop, (void *)&bytes_to_string},
    {0, NULL}
};

static PyArrayMethod_Spec *
get_cast_spec(
    const char *name,
    NPY_CASTING casting,
    NPY_ARRAYMETHOD_FLAGS flags,
    PyArray_DTypeMeta **dtypes,
    PyType_Slot *slots
) {
    // If dtypes or slots are NULL, an error has happened; return NULL.
    if ((slots == NULL) || (dtypes == NULL)) {
        return NULL;
    }

    PyArrayMethod_Spec *ret = (PyArrayMethod_Spec *)PyMem_Malloc(sizeof(PyArrayMethod_Spec));
    if (ret == NULL) {
        return reinterpret_cast<PyArrayMethod_Spec *>(PyErr_NoMemory());
    }

    ret->name = name;
    ret->nin = 1;
    ret->nout = 1;
    ret->casting = casting;
    ret->flags = flags;
    ret->dtypes = dtypes;
    ret->slots = slots;

    return ret;
}

// Check if the argument is inf using `isinf_func`, and cast the result
// to a bool; if `isinf_func` is unspecified, use std::isinf.
// Needed to ensure the right return type for getStringToFloatCastSpec.
template<typename T>
static bool
is_inf(T x) {
    return std::isinf(x);
}
template<typename T, int (*isinf_func)(T)>
static bool
is_inf(T x) {
    return static_cast<bool>(isinf_func(x));
}

// Cast the argument to the given type.
// Needed because getStringToFloatCastSpec takes a function rather than
// a type (for casting) as its double_to_float template parameter
template<typename NpyType>
static NpyType
to_float(double x) {
    return static_cast<NpyType>(x);
}

NPY_NO_EXPORT PyArrayMethod_Spec **
get_casts() {
    PyArray_DTypeMeta **t2t_dtypes = get_dtypes(
        &PyArray_StringDType,
        &PyArray_StringDType
    );

    PyArrayMethod_Spec *ThisToThisCastSpec = get_cast_spec(
        make_s2type_name(NPY_VSTRING),
        NPY_UNSAFE_CASTING,
        NPY_METH_SUPPORTS_UNALIGNED,
        t2t_dtypes,
        s2s_slots
    );

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
        make_type2s_name(NPY_UNICODE),
        NPY_SAME_KIND_CASTING,
        NPY_METH_NO_FLOATINGPOINT_ERRORS,
        u2s_dtypes,
        u2s_slots
    );

    PyArray_DTypeMeta **s2u_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_UnicodeDType);

    PyArrayMethod_Spec *StringToUnicodeCastSpec = get_cast_spec(
        make_s2type_name(NPY_UNICODE),
        NPY_SAME_KIND_CASTING,
        NPY_METH_NO_FLOATINGPOINT_ERRORS,
        s2u_dtypes,
        s2u_slots
    );

    PyArray_DTypeMeta **s2b_dtypes =
            get_dtypes(&PyArray_StringDType, &PyArray_BoolDType);

    PyArrayMethod_Spec *StringToBoolCastSpec = get_cast_spec(
        make_s2type_name(NPY_BOOL),
        NPY_SAME_KIND_CASTING,
        NPY_METH_NO_FLOATINGPOINT_ERRORS,
        s2b_dtypes,
        s2b_slots
    );

    PyArray_DTypeMeta **b2s_dtypes =
            get_dtypes(&PyArray_BoolDType, &PyArray_StringDType);

    PyArrayMethod_Spec *BoolToStringCastSpec = get_cast_spec(
        make_type2s_name(NPY_BOOL),
        NPY_SAME_KIND_CASTING,
        NPY_METH_NO_FLOATINGPOINT_ERRORS,
        b2s_dtypes,
        b2s_slots
    );

    PyArray_DTypeMeta **s2dt_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_DatetimeDType);

    PyArrayMethod_Spec *StringToDatetimeCastSpec = get_cast_spec(
        make_s2type_name(NPY_DATETIME),
        NPY_UNSAFE_CASTING,
        static_cast<NPY_ARRAYMETHOD_FLAGS>(NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI),
        s2dt_dtypes,
        s2dt_slots
    );

    PyArray_DTypeMeta **dt2s_dtypes = get_dtypes(
            &PyArray_DatetimeDType, &PyArray_StringDType);

    PyArrayMethod_Spec *DatetimeToStringCastSpec = get_cast_spec(
        make_type2s_name(NPY_DATETIME),
        NPY_SAFE_CASTING,
        static_cast<NPY_ARRAYMETHOD_FLAGS>(NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI),
        dt2s_dtypes,
        dt2s_slots
    );

    PyArray_DTypeMeta **s2td_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_TimedeltaDType);

    PyArrayMethod_Spec *StringToTimedeltaCastSpec = get_cast_spec(
        make_s2type_name(NPY_TIMEDELTA),
        NPY_UNSAFE_CASTING,
        static_cast<NPY_ARRAYMETHOD_FLAGS>(NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI),
        s2td_dtypes,
        s2td_slots
    );

    PyArray_DTypeMeta **td2s_dtypes = get_dtypes(
            &PyArray_TimedeltaDType, &PyArray_StringDType);

    PyArrayMethod_Spec *TimedeltaToStringCastSpec = get_cast_spec(
        make_type2s_name(NPY_TIMEDELTA),
        NPY_SAFE_CASTING,
        static_cast<NPY_ARRAYMETHOD_FLAGS>(NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI),
        td2s_dtypes,
        td2s_slots
    );

    PyArray_DTypeMeta **s2v_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_VoidDType);

    PyArrayMethod_Spec *StringToVoidCastSpec = get_cast_spec(
        make_s2type_name(NPY_VOID),
        NPY_SAME_KIND_CASTING,
        NPY_METH_NO_FLOATINGPOINT_ERRORS,
        s2v_dtypes,
        s2v_slots
    );

    PyArray_DTypeMeta **v2s_dtypes = get_dtypes(
            &PyArray_VoidDType, &PyArray_StringDType);

    PyArrayMethod_Spec *VoidToStringCastSpec = get_cast_spec(
        make_type2s_name(NPY_VOID),
        NPY_SAME_KIND_CASTING,
        NPY_METH_NO_FLOATINGPOINT_ERRORS,
        v2s_dtypes,
        v2s_slots
    );

    PyArray_DTypeMeta **s2bytes_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_BytesDType);

    PyArrayMethod_Spec *StringToBytesCastSpec = get_cast_spec(
        make_s2type_name(NPY_BYTE),
        NPY_SAME_KIND_CASTING,
        NPY_METH_NO_FLOATINGPOINT_ERRORS,
        s2bytes_dtypes,
        s2bytes_slots
    );

    PyArray_DTypeMeta **bytes2s_dtypes = get_dtypes(
            &PyArray_BytesDType, &PyArray_StringDType);

    PyArrayMethod_Spec *BytesToStringCastSpec = get_cast_spec(
        make_type2s_name(NPY_BYTE),
        NPY_SAME_KIND_CASTING,
        NPY_METH_NO_FLOATINGPOINT_ERRORS,
        bytes2s_dtypes,
        bytes2s_slots
    );

    PyArrayMethod_Spec **casts = (PyArrayMethod_Spec **)PyMem_Malloc(
        (num_casts + 1) * sizeof(PyArrayMethod_Spec *)
    );
    if (casts == NULL) {
        return reinterpret_cast<PyArrayMethod_Spec **>(PyErr_NoMemory());
    }

    int cast_i = 0;

    casts[cast_i++] = ThisToThisCastSpec;
    casts[cast_i++] = UnicodeToStringCastSpec;
    casts[cast_i++] = StringToUnicodeCastSpec;
    casts[cast_i++] = StringToBoolCastSpec;
    casts[cast_i++] = BoolToStringCastSpec;

    casts[cast_i++] = getStringToIntCastSpec<npy_int8,  npy_longlong, NPY_INT8>();
    casts[cast_i++] = getStringToIntCastSpec<npy_int16, npy_longlong, NPY_INT16>();
    casts[cast_i++] = getStringToIntCastSpec<npy_int32, npy_longlong, NPY_INT32>();
    casts[cast_i++] = getStringToIntCastSpec<npy_int64, npy_longlong, NPY_INT64>();
    casts[cast_i++] = getIntToStringCastSpec<npy_int8,  long long, NPY_INT8>();
    casts[cast_i++] = getIntToStringCastSpec<npy_int16, long long, NPY_INT16>();
    casts[cast_i++] = getIntToStringCastSpec<npy_int32, long long, NPY_INT32>();
    casts[cast_i++] = getIntToStringCastSpec<npy_int64, long long, NPY_INT64>();

    casts[cast_i++] = getStringToIntCastSpec<npy_uint8,  npy_ulonglong, NPY_UINT8>();
    casts[cast_i++] = getStringToIntCastSpec<npy_uint16, npy_ulonglong, NPY_UINT16>();
    casts[cast_i++] = getStringToIntCastSpec<npy_uint32, npy_ulonglong, NPY_UINT32>();
    casts[cast_i++] = getStringToIntCastSpec<npy_uint64, npy_ulonglong, NPY_UINT64>();
    casts[cast_i++] = getIntToStringCastSpec<npy_uint8,  unsigned long long, NPY_UINT8>();
    casts[cast_i++] = getIntToStringCastSpec<npy_uint16, unsigned long long, NPY_UINT16>();
    casts[cast_i++] = getIntToStringCastSpec<npy_uint32, unsigned long long, NPY_UINT32>();
    casts[cast_i++] = getIntToStringCastSpec<npy_uint64, unsigned long long, NPY_UINT64>();

#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
    casts[cast_i++] = getStringToIntCastSpec<npy_byte,  npy_longlong,  NPY_BYTE>();
    casts[cast_i++] = getStringToIntCastSpec<npy_ubyte, npy_ulonglong, NPY_UBYTE>();
    casts[cast_i++] = getIntToStringCastSpec<npy_byte,  long long,          NPY_BYTE>();
    casts[cast_i++] = getIntToStringCastSpec<npy_ubyte, unsigned long long, NPY_UBYTE>();
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
    casts[cast_i++] = getStringToIntCastSpec<npy_short,  npy_longlong,  NPY_SHORT>();
    casts[cast_i++] = getStringToIntCastSpec<npy_ushort, npy_ulonglong, NPY_USHORT>();
    casts[cast_i++] = getIntToStringCastSpec<npy_short,  long long,          NPY_SHORT>();
    casts[cast_i++] = getIntToStringCastSpec<npy_ushort, unsigned long long, NPY_USHORT>();
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    casts[cast_i++] = getStringToIntCastSpec<npy_int,  npy_longlong,  NPY_INT>();
    casts[cast_i++] = getStringToIntCastSpec<npy_uint, npy_ulonglong, NPY_UINT>();
    casts[cast_i++] = getIntToStringCastSpec<npy_int,  long long,          NPY_INT>();
    casts[cast_i++] = getIntToStringCastSpec<npy_uint, unsigned long long, NPY_UINT>();
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
    casts[cast_i++] = getStringToIntCastSpec<npy_longlong, npy_longlong,  NPY_LONGLONG>();
    casts[cast_i++] = getStringToIntCastSpec<npy_longlong, npy_ulonglong, NPY_ULONGLONG>();
    casts[cast_i++] = getIntToStringCastSpec<npy_longlong, long long,          NPY_LONGLONG>();
    casts[cast_i++] = getIntToStringCastSpec<npy_longlong, unsigned long long, NPY_ULONGLONG>();
#endif

    casts[cast_i++] = getStringToFloatCastSpec<npy_float16, NPY_HALF,  is_inf<npy_half, npy_half_isinf>, is_inf<double>, npy_double_to_half>();
    casts[cast_i++] = getStringToFloatCastSpec<npy_float32, NPY_FLOAT, is_inf<npy_float32>, is_inf<double>, to_float<npy_float32>>();
    casts[cast_i++] = getFloatToStringCastSpec<npy_float16, NPY_HALF>();
    casts[cast_i++] = getFloatToStringCastSpec<npy_float32, NPY_FLOAT>();

    // Special handling for f64 and longdouble types because they don't fit in a PyFloat
    casts[cast_i++] = getStringToFloatCastSpec<npy_float64,    NPY_DOUBLE>();
    casts[cast_i++] = getFloatToStringCastSpec<npy_float64,    NPY_DOUBLE>();

    // TODO: this is incorrect. The longdouble to unicode cast is also broken in
    // the same way. To fix this we'd need an ldtoa implementation in NumPy. It's
    // not in the standard library. Another option would be to use `snprintf` but we'd
    // need to somehow pre-calculate the size of the result string.
    //
    // TODO: Add a concrete implementation to properly handle 80-bit long doubles on Linux.
    casts[cast_i++] = getStringToFloatCastSpec<npy_longdouble, NPY_LONGDOUBLE, nullptr, nullptr, nullptr, static_cast<NPY_ARRAYMETHOD_FLAGS>(NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI)>();
    casts[cast_i++] = getFloatToStringCastSpec<npy_longdouble, NPY_LONGDOUBLE, static_cast<NPY_ARRAYMETHOD_FLAGS>(NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI)>();

    casts[cast_i++] = getStringToComplexCastSpec<npy_cfloat,      npy_float,      NPY_CFLOAT,      npy_csetrealf, npy_csetimagf>();
    casts[cast_i++] = getStringToComplexCastSpec<npy_cdouble,     npy_double,     NPY_CDOUBLE,     npy_csetreal,  npy_csetimag>();
    casts[cast_i++] = getStringToComplexCastSpec<npy_clongdouble, npy_longdouble, NPY_CLONGDOUBLE, npy_csetreall, npy_csetimagl>();
    casts[cast_i++] = getFloatToStringCastSpec<npy_cfloat,      NPY_CFLOAT>();
    casts[cast_i++] = getFloatToStringCastSpec<npy_cdouble,     NPY_CDOUBLE>();
    casts[cast_i++] = getFloatToStringCastSpec<npy_clongdouble, NPY_CLONGDOUBLE>();

    casts[cast_i++] = StringToDatetimeCastSpec;
    casts[cast_i++] = DatetimeToStringCastSpec;
    casts[cast_i++] = StringToTimedeltaCastSpec;
    casts[cast_i++] = TimedeltaToStringCastSpec;
    casts[cast_i++] = StringToVoidCastSpec;
    casts[cast_i++] = VoidToStringCastSpec;
    casts[cast_i++] = StringToBytesCastSpec;
    casts[cast_i++] = BytesToStringCastSpec;
    casts[cast_i++] = NULL;

    // Check that every cast spec is valid
    if (PyErr_Occurred() != NULL) {
        return NULL;
    }
    for (int i = 0; i<num_casts; i++) {
        assert(casts[i] != NULL);
    }

    assert(casts[num_casts] == NULL);
    assert(cast_i == num_casts + 1);

    return casts;
}
