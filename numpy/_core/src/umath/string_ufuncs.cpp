#include <Python.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"

#include "numpyos.h"
#include "dispatching.h"
#include "dtypemeta.h"
#include "common_dtype.h"
#include "convert_datatype.h"

#include "string_ufuncs.h"
#include "string_fastsearch.h"
#include "string_buffer.h"


enum class STARTPOSITION {
    FRONT, BACK
};


template <typename character>
static inline int
character_cmp(character a, character b)
{
    if (a == b) {
        return 0;
    }
    else if (a < b) {
        return -1;
    }
    else {
        return 1;
    }
}


template <typename character>
static inline int
string_rstrip(const character *str, int elsize)
{
    /*
     * Ignore/"trim" trailing whitespace (and 0s).  Note that this function
     * does not support unicode whitespace (and never has).
     */
    while (elsize > 0) {
        character c = str[elsize-1];
        if (c != (character)0 && !NumPyOS_ascii_isspace(c)) {
            break;
        }
        elsize--;
    }
    return elsize;
}


/*
 * Helper to fixup start/end slice values.
 *
 * This function is taken from CPython's unicode module
 * (https://github.com/python/cpython/blob/0b718e6407da65b838576a2459d630824ca62155/Objects/bytes_methods.c#L495)
 * in order to remain compatible with how CPython handles
 * start/end arguments to str function like find/rfind etc.
 */
static inline void
adjust_offsets(npy_int64 *start, npy_int64 *end, npy_int64 len)
{
    if (*end > len) {
        *end = len;
    }
    else if (*end < 0) {
        *end += len;
        if (*end < 0) {
            *end = 0;
        }
    }

    if (*start < 0) {
        *start += len;
        if (*start < 0) {
            *start = 0;
        }
    }
}


template <ENCODING enc>
static inline npy_bool
tailmatch(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end,
          STARTPOSITION direction)
{
    npy_int64 len1 = buf1.num_codepoints();
    npy_int64 len2 = buf2.num_codepoints();

    adjust_offsets(&start, &end, len1);
    end -= len2;
    if (end < start) {
        return 0;
    }

    if (len2 == 0) {
        return 1;
    }

    npy_int64 offset;
    npy_int64 end_sub = len2 - 1;
    if (direction == STARTPOSITION::BACK) {
        offset = end;
    }
    else {
        offset = start;
    }

    if (buf1[offset] == buf2[0] && buf1[offset+end_sub] == buf2[end_sub]) {
        return !(buf1 + offset).buffer_memcmp(buf2, len2);
    }

    return 0;
}


/*
 * Compare two strings of different length.  Note that either string may be
 * zero padded (trailing zeros are ignored in other words, the shorter word
 * is always padded with zeros).
 */
template <bool rstrip, typename character>
static inline int
string_cmp(const character *str1, int elsize1, const character *str2, int elsize2)
{
    int len1 = elsize1, len2 = elsize2;
    if (rstrip) {
        len1 = string_rstrip(str1, elsize1);
        len2 = string_rstrip(str2, elsize2);
    }

    int n = PyArray_MIN(len1, len2);

    if (sizeof(character) == 1) {
        /*
         * TODO: `memcmp` makes things 2x faster for longer words that match
         *       exactly, but at least 2x slower for short or mismatching ones.
         */
        int cmp = memcmp(str1, str2, n);
        if (cmp != 0) {
            return cmp;
        }
        str1 += n;
        str2 += n;
    }
    else {
        for (int i = 0; i < n; i++) {
            int cmp = character_cmp(*str1, *str2);
            if (cmp != 0) {
                return cmp;
            }
            str1++;
            str2++;
        }
    }
    if (len1 > len2) {
        for (int i = n; i < len1; i++) {
            int cmp = character_cmp(*str1, (character)0);
            if (cmp != 0) {
                return cmp;
            }
            str1++;
        }
    }
    else if (len2 > len1) {
        for (int i = n; i < len2; i++) {
            int cmp = character_cmp((character)0, *str2);
            if (cmp != 0) {
                return cmp;
            }
            str2++;
        }
    }
    return 0;
}


template <ENCODING enc>
static inline void
string_add(Buffer<enc> buf1, Buffer<enc> buf2, char *out)
{
    npy_int64 len1 = buf1.num_codepoints();
    npy_int64 len2 = buf2.num_codepoints();
    buf1.buffer_memcpy(out, (size_t) len1);
    buf2.buffer_memcpy_with_offset(out, (size_t) len1, (size_t) len2);
}


static inline npy_bool
string_isdecimal(Buffer<ENCODING::UTF32> buf)
{
    npy_int64 len = buf.num_codepoints();

    if (len == 0) {
        return (npy_bool) 0;
    }

    for (npy_int64 i = 0; i < len; i++) {
        npy_bool isdecimal = (npy_bool) Py_UNICODE_ISDECIMAL(*buf);
        if (!isdecimal) {
            return isdecimal;
        }
        buf++;
    }
    return (npy_bool) 1;
}


static inline npy_bool
string_isnumeric(Buffer<ENCODING::UTF32> buf)
{
    npy_int64 len = buf.num_codepoints();

    if (len == 0) {
        return (npy_bool) 0;
    }

    for (npy_int64 i = 0; i < len; i++) {
        npy_bool isnumeric = (npy_bool) Py_UNICODE_ISNUMERIC(*buf);
        if (!isnumeric) {
            return isnumeric;
        }
        buf++;
    }
    return (npy_bool) 1;
}


template <ENCODING enc>
static inline npy_intp
string_find(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    npy_int64 len1 = buf1.num_codepoints();
    npy_int64 len2 = buf2.num_codepoints();

    adjust_offsets(&start, &end, len1);
    if (end - start < len2) {
        return (npy_intp) -1;
    }

    if (len2 == 0) {
        return (npy_intp) start;
    }
    if (len2 == 1) {
        npy_ucs4 ch = *buf2;
        npy_intp result = (npy_intp) findchar<enc>(buf1 + start, end - start, ch);
        if (result == -1) {
            return (npy_intp) -1;
        }
        else {
            return result + (npy_intp) start;
        }
    }

    npy_intp pos = fastsearch<enc>(buf1 + start, end - start, buf2, len2, -1, FAST_SEARCH);
    if (pos >= 0) {
        pos += start;
    }
    return pos;
}


template <ENCODING enc>
static inline npy_intp
string_rfind(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    npy_int64 len1 = buf1.num_codepoints();
    npy_int64 len2 = buf2.num_codepoints();

    adjust_offsets(&start, &end, len1);
    if (end - start < len2) {
        return (npy_intp) -1;
    }

    if (len2 == 0) {
        return (npy_intp) end;
    }
    if (len2 == 1) {
        npy_ucs4 ch = *buf2;
        npy_intp result = (npy_intp) rfindchar(buf1 + start, end - start, ch);
        if (result == -1) {
            return (npy_intp) -1;
        }
        else {
            return result + (npy_intp) start;
        }
    }

    npy_intp pos = (npy_intp) fastsearch<enc>(buf1 + start, end - start, buf2, len2, -1, FAST_RSEARCH);
    if (pos >= 0) {
        pos += start;
    }
    return pos;
}


/*
 * Count the number of occurences of buf2 in buf1 between
 * start (inclusive) and end (exclusive)
 */
template <ENCODING enc>
static inline npy_intp
string_count(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    npy_int64 len1 = buf1.num_codepoints();
    npy_int64 len2 = buf2.num_codepoints();

    adjust_offsets(&start, &end, len1);
    if (end < start || end - start < len2) {
        return (npy_intp) 0;
    }

    if (len2 == 0) {
        return (end - start) < PY_SSIZE_T_MAX ? end - start + 1 : PY_SSIZE_T_MAX;
    }

    npy_intp count = (npy_intp) fastsearch<enc>(buf1 + start, end - start, buf2, len2,
                                                PY_SSIZE_T_MAX, FAST_COUNT);
    if (count < 0) {
        return 0;
    }
    return count;
}


/*
 * Helper for templating, avoids warnings about uncovered switch paths.
 */
enum class COMP {
    EQ, NE, LT, LE, GT, GE,
};

static char const *
comp_name(COMP comp) {
    switch(comp) {
        case COMP::EQ: return "equal";
        case COMP::NE: return "not_equal";
        case COMP::LT: return "less";
        case COMP::LE: return "less_equal";
        case COMP::GT: return "greater";
        case COMP::GE: return "greater_equal";
        default:
            assert(0);
            return nullptr;
    }
}


template <bool rstrip, COMP comp, typename character>
static int
string_comparison_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    /*
     * Note, fetching `elsize` from the descriptor is OK even without the GIL,
     * however it may be that this should be moved into `auxdata` eventually,
     * which may also be slightly faster/cleaner (but more involved).
     */
    int elsize1 = context->descriptors[0]->elsize / sizeof(character);
    int elsize2 = context->descriptors[1]->elsize / sizeof(character);

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        int cmp = string_cmp<rstrip>(
                (character *)in1, elsize1, (character *)in2, elsize2);
        npy_bool res;
        switch (comp) {
            case COMP::EQ:
                res = cmp == 0;
                break;
            case COMP::NE:
                res = cmp != 0;
                break;
            case COMP::LT:
                res = cmp < 0;
                break;
            case COMP::LE:
                res = cmp <= 0;
                break;
            case COMP::GT:
                res = cmp > 0;
                break;
            case COMP::GE:
                res = cmp >= 0;
                break;
        }
        *(npy_bool *)out = res;

        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }
    return 0;
}


template <ENCODING enc>
static int
string_add_loop(PyArrayMethod_Context *context,
                char *const data[], npy_intp const dimensions[],
                npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        string_add<enc>(buf1, buf2, out);

        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    return 0;
}


template <ENCODING enc>
static int
string_len_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in, elsize);
        *(npy_intp *)out = buf.num_codepoints();

        in += strides[0];
        out += strides[1];
    }

    return 0;
}


template <ENCODING enc>
static int
string_isalpha_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in, elsize);
        *(npy_bool *)out = (npy_bool) buf.isalpha();

        in += strides[0];
        out += strides[1];
    }

    return 0;
}


template <ENCODING enc>
static int
string_isdigit_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in, elsize);
        *(npy_bool *)out = (npy_bool) buf.isdigit();

        in += strides[0];
        out += strides[1];
    }

    return 0;
}


template <ENCODING enc>
static int
string_isspace_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in, elsize);
        *(npy_bool *)out = (npy_bool) buf.isspace();

        in += strides[0];
        out += strides[1];
    }

    return 0;
}


static int
string_isdecimal_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<ENCODING::UTF32> buf(in, elsize);
        npy_bool res = string_isdecimal(buf);
        *(npy_bool *)out = res;

        in += strides[0];
        out += strides[1];
    }

    return 0;
}


static int
string_isnumeric_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<ENCODING::UTF32> buf(in, elsize);
        npy_bool res = string_isnumeric(buf);
        *(npy_bool *)out = res;

        in += strides[0];
        out += strides[1];
    }

    return 0;
}


template <ENCODING enc>
static int
string_find_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *in4 = data[3];
    char *out = data[4];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        npy_intp idx = string_find<enc>(buf1, buf2, *(npy_int64 *)in3, *(npy_int64 *)in4);
        *(npy_intp *)out = idx;

        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }
    return 0;
}


template <ENCODING enc>
static int
string_rfind_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *in4 = data[3];
    char *out = data[4];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        npy_intp idx = string_rfind<enc>(buf1, buf2, *(npy_int64 *)in3, *(npy_int64 *)in4);
        *(npy_intp *)out = idx;

        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }
    return 0;
}


template <ENCODING enc>
static int
string_count_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *in4 = data[3];
    char *out = data[4];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        npy_intp count = string_count<enc>(buf1, buf2, *(npy_int64 *)in3, *(npy_int64 *)in4);
        *(npy_intp *)out = count;

        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }
    return 0;
}


template <ENCODING enc>
static int
string_startswith_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *in4 = data[3];
    char *out = data[4];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        npy_bool match = tailmatch<enc>(buf1, buf2, *(npy_int64 *)in3, *(npy_int64 *)in4,
                                  STARTPOSITION::FRONT);
        *(npy_bool *)out = match;

        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }
    return 0;
}


template <ENCODING enc>
static int
string_endswith_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *in4 = data[3];
    char *out = data[4];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        npy_bool match = tailmatch<enc>(buf1, buf2, *(npy_int64 *)in3, *(npy_int64 *)in4,
                                  STARTPOSITION::BACK);
        *(npy_bool *)out = match;

        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }
    return 0;
}


/* Resolve descriptors & promoter functions */

static NPY_CASTING
string_addition_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[3]),
        PyArray_Descr *given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
    if (loop_descrs[1] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    loop_descrs[2] = PyArray_DescrNew(loop_descrs[0]);
    if (loop_descrs[2] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    loop_descrs[2]->elsize += loop_descrs[1]->elsize;

    return NPY_NO_CASTING;
}


static int
string_find_rfind_count_promoter(PyUFuncObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[0] = op_dtypes[0];
    Py_INCREF(op_dtypes[1]);
    new_op_dtypes[1] = op_dtypes[1];
    new_op_dtypes[2] = PyArray_DTypeFromTypeNum(NPY_INT64);
    new_op_dtypes[3] = PyArray_DTypeFromTypeNum(NPY_INT64);
    new_op_dtypes[4] = PyArray_DTypeFromTypeNum(NPY_DEFAULT_INT);
    return 0;
}


static int
string_startswith_endswith_promoter(PyUFuncObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[0] = op_dtypes[0];
    Py_INCREF(op_dtypes[1]);
    new_op_dtypes[1] = op_dtypes[1];
    new_op_dtypes[2] = PyArray_DTypeFromTypeNum(NPY_INT64);
    new_op_dtypes[3] = PyArray_DTypeFromTypeNum(NPY_INT64);
    new_op_dtypes[4] = PyArray_DTypeFromTypeNum(NPY_BOOL);
    return 0;
}


/*
 * Machinery to add the string loops to the existing ufuncs.
 */

static int
install_promoter(PyObject *umath, const char *ufunc_name,
                 PyObject *promoterinfo)
{
    PyObject *name = PyUnicode_FromString(ufunc_name);
    if (name == nullptr) {
        return -1;
    }
    PyObject *ufunc = PyObject_GetItem(umath, name);
    Py_DECREF(name);
    if (ufunc == nullptr) {
        return -1;
    }

    int res = PyUFunc_AddLoop((PyUFuncObject *)ufunc, promoterinfo, 0);
    Py_DECREF(ufunc);
    return res;
}


/*
 * This function replaces the strided loop with the passed in one,
 * and registers it with the given ufunc.
 */
static int
add_loop(PyObject *umath, const char *ufunc_name,
         PyArrayMethod_Spec *spec, PyArrayMethod_StridedLoop *loop)
{
    PyObject *name = PyUnicode_FromString(ufunc_name);
    if (name == nullptr) {
        return -1;
    }
    PyObject *ufunc = PyObject_GetItem(umath, name);
    Py_DECREF(name);
    if (ufunc == nullptr) {
        return -1;
    }
    spec->slots[0].pfunc = (void *)loop;

    int res = PyUFunc_AddLoopFromSpec_int(ufunc, spec, 1);
    Py_DECREF(ufunc);
    return res;
}


template<bool rstrip, typename character, COMP...>
struct add_loops;

template<bool rstrip, typename character>
struct add_loops<rstrip, character> {
    int operator()(PyObject*, PyArrayMethod_Spec*) {
        return 0;
    }
};

template<bool rstrip, typename character, COMP comp, COMP... comps>
struct add_loops<rstrip, character, comp, comps...> {
    int operator()(PyObject* umath, PyArrayMethod_Spec* spec) {
        PyArrayMethod_StridedLoop* loop = string_comparison_loop<rstrip, comp, character>;

        if (add_loop(umath, comp_name(comp), spec, loop) < 0) {
            return -1;
        }
        else {
            return add_loops<rstrip, character, comps...>()(umath, spec);
        }
    }
};


static int
init_comparison(PyObject *umath)
{
    int res = -1;
    /* NOTE: This should receive global symbols? */
    PyArray_DTypeMeta *String = PyArray_DTypeFromTypeNum(NPY_STRING);
    PyArray_DTypeMeta *Unicode = PyArray_DTypeFromTypeNum(NPY_UNICODE);
    PyArray_DTypeMeta *Bool = PyArray_DTypeFromTypeNum(NPY_BOOL);

    /* We start with the string loops: */
    PyArray_DTypeMeta *dtypes[] = {String, String, Bool};
    /*
     * We only have one loop right now, the strided one.  The default type
     * resolver ensures native byte order/canonical representation.
     */
    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, nullptr},
        {0, nullptr}
    };

    PyArrayMethod_Spec spec = {};
    spec.name = "templated_string_comparison";
    spec.nin = 2;
    spec.nout = 1;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    /* All String loops */
    using string_looper = add_loops<false, npy_byte, COMP::EQ, COMP::NE, COMP::LT, COMP::LE, COMP::GT, COMP::GE>;
    if (string_looper()(umath, &spec) < 0) {
        goto finish;
    }

    /* All Unicode loops */
    using ucs_looper = add_loops<false, npy_ucs4, COMP::EQ, COMP::NE, COMP::LT, COMP::LE, COMP::GT, COMP::GE>;
    dtypes[0] = Unicode;
    dtypes[1] = Unicode;
    if (ucs_looper()(umath, &spec) < 0) {
        goto finish;
    }

    res = 0;
  finish:
    Py_DECREF(String);
    Py_DECREF(Unicode);
    Py_DECREF(Bool);
    return res;
}


static int
init_promoter(PyObject *umath, const char *name, int nin, int nout,
              promoter_function promoter)
{
    PyObject *promoter_obj = PyCapsule_New((void *) promoter, "numpy._ufunc_promoter", NULL);
    if (promoter_obj == NULL) {
        return -1;
    }

    PyObject *dtypes_tuple = PyTuple_New(nin + nout);
    if (dtypes_tuple == NULL) {
        Py_DECREF(promoter_obj);
        return -1;
    }
    for (int i = 0; i < nin + nout; i++) {
        PyTuple_SET_ITEM(dtypes_tuple, i, Py_None);
    }

    PyObject *info = PyTuple_Pack(2, dtypes_tuple, promoter_obj);
    Py_DECREF(dtypes_tuple);
    Py_DECREF(promoter_obj);
    if (info == NULL) {
        return -1;
    }

    if (install_promoter(umath, name, info) < 0) {
        return -1;
    }

    return 0;
}


template <ENCODING enc>
static int
init_ufunc(PyObject *umath, const char *name, const char *specname, int nin, int nout,
           NPY_TYPES *typenums, PyArrayMethod_StridedLoop loop,
           resolve_descriptors_function resolve_descriptors)
{
    int res = -1;

    PyArray_DTypeMeta **dtypes = (PyArray_DTypeMeta **) PyMem_Malloc(
        (nin + nout) * sizeof(PyArray_DTypeMeta *));
    if (dtypes == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    for (int i = 0; i < nin+nout; i++) {
        if (typenums[i] == NPY_OBJECT && enc == ENCODING::UTF32) {
            dtypes[i] = PyArray_DTypeFromTypeNum(NPY_UNICODE);
        }
        else if (typenums[i] == NPY_OBJECT && enc == ENCODING::ASCII) {
            dtypes[i] = PyArray_DTypeFromTypeNum(NPY_STRING);
        }
        else {
            dtypes[i] = PyArray_DTypeFromTypeNum(typenums[i]);
        }
    }

    PyType_Slot slots[3];
    slots[0] = {NPY_METH_strided_loop, nullptr};
    slots[2] = {0, nullptr};
    if (resolve_descriptors != NULL) {
        slots[1] = {NPY_METH_resolve_descriptors, (void *) resolve_descriptors};
    }
    else {
        slots[1] = {0, nullptr};
    }

    PyArrayMethod_Spec spec = {};
    spec.name = specname;
    spec.nin = nin;
    spec.nout = nout;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    if (add_loop(umath, name, &spec, loop) < 0) {
        goto finish;
    }

    res = 0;
  finish:
    for (int i = 0; i < nin+nout; i++) {
        Py_DECREF(dtypes[i]);
    }
    PyMem_Free((void *) dtypes);
    return res;
}


NPY_NO_EXPORT int
init_string_ufuncs(PyObject *umath)
{
    NPY_TYPES dtypes[] = {NPY_STRING, NPY_STRING, NPY_STRING, NPY_STRING, NPY_STRING};

    if (init_comparison(umath) < 0) {
        return -1;
    }

    // We use NPY_OBJECT as a sentinel value here, and this will be replaced by the
    // corresponding string dtype (either NPY_STRING or NPY_UNICODE).
    dtypes[0] = dtypes[1] = dtypes[2] = NPY_OBJECT;
    if (init_ufunc<ENCODING::ASCII>(
            umath, "add", "templated_string_add", 2, 1, dtypes,
            string_add_loop<ENCODING::ASCII>, string_addition_resolve_descriptors) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "add", "templated_string_add", 2, 1, dtypes,
            string_add_loop<ENCODING::UTF32>, string_addition_resolve_descriptors) < 0) {
        return -1;
    }

    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_DEFAULT_INT;
    if (init_ufunc<ENCODING::ASCII>(
            umath, "str_len", "templated_string_len", 1, 1, dtypes,
            string_len_loop<ENCODING::ASCII>, NULL) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "str_len", "templated_string_len", 1, 1, dtypes,
            string_len_loop<ENCODING::UTF32>, NULL) < 0) {
        return -1;
    }

    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_BOOL;
    if (init_ufunc<ENCODING::ASCII>(
            umath, "isalpha", "templated_string_isalpha", 1, 1, dtypes,
            string_isalpha_loop<ENCODING::ASCII>, NULL) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "isalpha", "templated_string_isalpha", 1, 1, dtypes,
            string_isalpha_loop<ENCODING::UTF32>, NULL) < 0) {
        return -1;
    }

    dtypes[0] = dtypes[1] = NPY_OBJECT;
    dtypes[2] = dtypes[3] = NPY_INT64;
    dtypes[4] = NPY_DEFAULT_INT;
    if (init_ufunc<ENCODING::ASCII>(
            umath, "find", "templated_string_find", 4, 1, dtypes,
            string_find_loop<ENCODING::ASCII>, NULL) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "find", "templated_string_find", 4, 1, dtypes,
            string_find_loop<ENCODING::UTF32>, NULL) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::ASCII>(
            umath, "rfind", "templated_string_rfind", 4, 1, dtypes,
            string_rfind_loop<ENCODING::ASCII>, NULL) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "rfind", "templated_string_rfind", 4, 1, dtypes,
            string_rfind_loop<ENCODING::UTF32>, NULL) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::ASCII>(
            umath, "count", "templated_string_count", 4, 1, dtypes,
            string_count_loop<ENCODING::ASCII>, NULL) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "count", "templated_string_count", 4, 1, dtypes,
            string_count_loop<ENCODING::UTF32>, NULL) < 0) {
        return -1;
    }
    if (init_promoter(umath, "find", 4, 1, string_find_rfind_count_promoter) < 0) {
        return -1;
    }
    if (init_promoter(umath, "rfind", 4, 1, string_find_rfind_count_promoter) < 0) {
        return -1;
    }
    if (init_promoter(umath, "count", 4, 1, string_find_rfind_count_promoter) < 0) {
        return -1;
    }

    dtypes[0] = dtypes[1] = NPY_OBJECT;
    dtypes[2] = dtypes[3] = NPY_INT64;
    dtypes[4] = NPY_BOOL;
    if (init_ufunc<ENCODING::ASCII>(
            umath, "startswith", "templated_string_startswith", 4, 1, dtypes,
            string_startswith_loop<ENCODING::ASCII>, NULL) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "startswith", "templated_string_startswith", 4, 1, dtypes,
            string_startswith_loop<ENCODING::UTF32>, NULL) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::ASCII>(
            umath, "endswith", "templated_string_endswith", 4, 1, dtypes,
            string_endswith_loop<ENCODING::ASCII>, NULL) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "endswith", "templated_string_endswith", 4, 1, dtypes,
            string_endswith_loop<ENCODING::UTF32>, NULL) < 0) {
        return -1;
    }
    if (init_promoter(umath, "startswith", 4, 1, string_startswith_endswith_promoter) < 0) {
        return -1;
    }
    if (init_promoter(umath, "endswith", 4, 1, string_startswith_endswith_promoter) < 0) {
        return -1;
    }

    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_BOOL;
    if (init_ufunc<ENCODING::ASCII>(
            umath, "isdigit", "templated_string_isdigit", 1, 1, dtypes,
            string_isdigit_loop<ENCODING::ASCII>, NULL) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "isdigit", "templated_string_isdigit", 1, 1, dtypes,
            string_isdigit_loop<ENCODING::UTF32>, NULL) < 0) {
        return -1;
    }

    if (init_ufunc<ENCODING::ASCII>(
            umath, "isspace", "templated_string_isspace", 1, 1, dtypes,
            string_isspace_loop<ENCODING::ASCII>, NULL) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "isspace", "templated_string_isspace", 1, 1, dtypes,
            string_isspace_loop<ENCODING::UTF32>, NULL) < 0) {
        return -1;
    }

    if (init_ufunc<ENCODING::UTF32>(
            umath, "isdecimal", "templated_string_isdecimal", 1, 1, dtypes,
            string_isdecimal_loop, NULL) < 0) {
        return -1;
    }

    if (init_ufunc<ENCODING::UTF32>(
            umath, "isnumeric", "templated_string_isnumeric", 1, 1, dtypes,
            string_isnumeric_loop, NULL) < 0) {
        return -1;
    }

    return 0;
}


template <bool rstrip, typename character>
static PyArrayMethod_StridedLoop *
get_strided_loop(int comp)
{
    switch (comp) {
        case Py_EQ:
            return string_comparison_loop<rstrip, COMP::EQ, character>;
        case Py_NE:
            return string_comparison_loop<rstrip, COMP::NE, character>;
        case Py_LT:
            return string_comparison_loop<rstrip, COMP::LT, character>;
        case Py_LE:
            return string_comparison_loop<rstrip, COMP::LE, character>;
        case Py_GT:
            return string_comparison_loop<rstrip, COMP::GT, character>;
        case Py_GE:
            return string_comparison_loop<rstrip, COMP::GE, character>;
        default:
            assert(false);  /* caller ensures this */
    }
    return nullptr;
}


/*
 * This function is used for `compare_chararrays` and currently also void
 * comparisons (unstructured voids).  The first could probably be deprecated
 * and removed but is used by `np.char.chararray` the latter should also be
 * moved to the ufunc probably (removing the need for manual looping).
 *
 * The `rstrip` mechanism is presumably for some fortran compat, but the
 * question is whether it would not be better to have/use `rstrip` on such
 * an array first...
 *
 * NOTE: This function is also used for unstructured voids, this works because
 *       `npy_byte` is correct.
 */
NPY_NO_EXPORT PyObject *
_umath_strings_richcompare(
        PyArrayObject *self, PyArrayObject *other, int cmp_op, int rstrip)
{
    NpyIter *iter = nullptr;
    PyObject *result = nullptr;

    char **dataptr = nullptr;
    npy_intp *strides = nullptr;
    npy_intp *countptr = nullptr;
    npy_intp size = 0;

    PyArrayMethod_Context context = {};
    NpyIter_IterNextFunc *iternext = nullptr;

    npy_uint32 it_flags = (
            NPY_ITER_EXTERNAL_LOOP | NPY_ITER_ZEROSIZE_OK |
            NPY_ITER_BUFFERED | NPY_ITER_GROWINNER);
    npy_uint32 op_flags[3] = {
            NPY_ITER_READONLY | NPY_ITER_ALIGNED,
            NPY_ITER_READONLY | NPY_ITER_ALIGNED,
            NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_ALIGNED};

    PyArrayMethod_StridedLoop *strided_loop = nullptr;
    NPY_BEGIN_THREADS_DEF;

    if (PyArray_TYPE(self) != PyArray_TYPE(other)) {
        /*
         * Comparison between Bytes and Unicode is not defined in Py3K;
         * we follow.
         * TODO: This makes no sense at all for `compare_chararrays`, kept
         *       only under the assumption that we are more likely to deprecate
         *       than fix it to begin with.
         */
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    PyArrayObject *ops[3] = {self, other, nullptr};
    PyArray_Descr *descrs[3] = {nullptr, nullptr, PyArray_DescrFromType(NPY_BOOL)};
    /* TODO: ensuring native byte order is not really necessary for == and != */
    descrs[0] = NPY_DT_CALL_ensure_canonical(PyArray_DESCR(self));
    if (descrs[0] == nullptr) {
        goto finish;
    }
    descrs[1] = NPY_DT_CALL_ensure_canonical(PyArray_DESCR(other));
    if (descrs[1] == nullptr) {
        goto finish;
    }

    /*
     * Create the iterator:
     */
    iter = NpyIter_AdvancedNew(
            3, ops, it_flags, NPY_KEEPORDER, NPY_SAFE_CASTING, op_flags, descrs,
            -1, nullptr, nullptr, 0);
    if (iter == nullptr) {
        goto finish;
    }

    size = NpyIter_GetIterSize(iter);
    if (size == 0) {
        result = (PyObject *)NpyIter_GetOperandArray(iter)[2];
        Py_INCREF(result);
        goto finish;
    }

    iternext = NpyIter_GetIterNext(iter, nullptr);
    if (iternext == nullptr) {
        goto finish;
    }

    /*
     * Prepare the inner-loop and execute it (we only need descriptors to be
     * passed in).
     */
    context.descriptors = descrs;

    dataptr = NpyIter_GetDataPtrArray(iter);
    strides = NpyIter_GetInnerStrideArray(iter);
    countptr = NpyIter_GetInnerLoopSizePtr(iter);

    if (rstrip == 0) {
        /* NOTE: Also used for VOID, so can be STRING, UNICODE, or VOID: */
        if (descrs[0]->type_num != NPY_UNICODE) {
            strided_loop = get_strided_loop<false, npy_byte>(cmp_op);
        }
        else {
            strided_loop = get_strided_loop<false, npy_ucs4>(cmp_op);
        }
    }
    else {
        if (descrs[0]->type_num != NPY_UNICODE) {
            strided_loop = get_strided_loop<true, npy_byte>(cmp_op);
        }
        else {
            strided_loop = get_strided_loop<true, npy_ucs4>(cmp_op);
        }
    }

    NPY_BEGIN_THREADS_THRESHOLDED(size);

    do {
         /* We know the loop cannot fail */
         strided_loop(&context, dataptr, countptr, strides, nullptr);
    } while (iternext(iter) != 0);

    NPY_END_THREADS;

    result = (PyObject *)NpyIter_GetOperandArray(iter)[2];
    Py_INCREF(result);

 finish:
    if (NpyIter_Deallocate(iter) < 0) {
        Py_CLEAR(result);
    }
    Py_XDECREF(descrs[0]);
    Py_XDECREF(descrs[1]);
    Py_XDECREF(descrs[2]);
    return result;
}
