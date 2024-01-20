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

enum class STRIPTYPE {
    LEFTSTRIP, RIGHTSTRIP, BOTHSTRIP
};


/*
 * Helper to fixup start/end slice values.
 *
 * This function is taken from CPython's unicode module
 * (https://github.com/python/cpython/blob/0b718e6407da65b838576a2459d630824ca62155/Objects/bytes_methods.c#L495)
 * in order to remain compatible with how CPython handles
 * start/end arguments to str function like find/rfind etc.
 */
static inline void
adjust_offsets(npy_int64 *start, npy_int64 *end, size_t len)
{
    if (*end > static_cast<npy_int64>(len)) {
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
    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();

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


template <ENCODING enc>
static inline void
string_add(Buffer<enc> buf1, Buffer<enc> buf2, Buffer<enc> out)
{
    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();
    buf1.buffer_memcpy(out, len1);
    buf2.buffer_memcpy(out + len1, len2);
    out.buffer_fill_with_zeros_after_index(len1 + len2);
}


static inline npy_bool
string_isdecimal(Buffer<ENCODING::UTF32> buf)
{
    size_t len = buf.num_codepoints();

    if (len == 0) {
        return (npy_bool) 0;
    }

    for (size_t i = 0; i < len; i++) {
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
    size_t len = buf.num_codepoints();

    if (len == 0) {
        return (npy_bool) 0;
    }

    for (size_t i = 0; i < len; i++) {
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
    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();

    adjust_offsets(&start, &end, len1);
    if (end - start < static_cast<npy_int64>(len2)) {
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
    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();

    adjust_offsets(&start, &end, len1);
    if (end - start < static_cast<npy_int64>(len2)) {
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


template <ENCODING enc>
static inline void
string_lrstrip_whitespace(Buffer<enc> buf, Buffer<enc> out, STRIPTYPE striptype)
{
    size_t len = buf.num_codepoints();
    if (len == 0) {
        out.buffer_fill_with_zeros_after_index(0);
        return;
    }

    size_t i = 0;
    if (striptype != STRIPTYPE::RIGHTSTRIP) {
        while (i < len) {
            if (!buf.isspace(i)) {
                break;
            }
            i++;
        }
    }

    npy_intp j = len - 1;  // Could also turn negative if we're stripping the whole string
    if (striptype != STRIPTYPE::LEFTSTRIP) {
        while (j >= static_cast<npy_intp>(i)) {
            if (!buf.isspace(j)) {
                break;
            }
            j--;
        }
    }

    (buf + i).buffer_memcpy(out, j - i + 1);
    out.buffer_fill_with_zeros_after_index(j - i + 1);
}


template <ENCODING enc>
static inline void
string_lrstrip_chars(Buffer<enc> buf1, Buffer<enc> buf2, Buffer<enc> out, STRIPTYPE striptype)
{
    size_t len1 = buf1.num_codepoints();
    if (len1 == 0) {
        out.buffer_fill_with_zeros_after_index(0);
        return;
    }

    size_t len2 = buf2.num_codepoints();
    if (len2 == 0) {
        buf1.buffer_memcpy(out, len1);
        out.buffer_fill_with_zeros_after_index(len1);
        return;
    }

    size_t i = 0;
    if (striptype != STRIPTYPE::RIGHTSTRIP) {
        while (i < len1) {
            if (findchar(buf2, len2, buf1[i]) < 0) {
                break;
            }
            i++;
        }
    }

    npy_intp j = len1 - 1;  // Could also turn negative if we're stripping the whole string
    if (striptype != STRIPTYPE::LEFTSTRIP) {
        while (j >= static_cast<npy_intp>(i)) {
            if (findchar(buf2, len2, buf1[j]) < 0) {
                break;
            }
            j--;
        }
    }

    (buf1 + i).buffer_memcpy(out, j - i + 1);
    out.buffer_fill_with_zeros_after_index(j - i + 1);
}


/*
 * Count the number of occurences of buf2 in buf1 between
 * start (inclusive) and end (exclusive)
 */
template <ENCODING enc>
static inline npy_intp
string_count(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();

    adjust_offsets(&start, &end, len1);
    if (end - start < static_cast<npy_int64>(len2)) {
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


template <ENCODING enc>
static inline npy_intp
findslice_for_replace(Buffer<enc> buf1, npy_int64 len1, Buffer<enc> buf2, npy_int64 len2)
{
    if (len2 == 0) {
        return 0;
    }
    if (len2 == 1) {
        return (npy_intp) findchar(buf1, len1, *buf2);
    }
    return (npy_intp) fastsearch(buf1, len1, buf2, len2, -1, FAST_SEARCH);
}


template <ENCODING enc>
static inline void
string_replace(Buffer<enc> buf1, Buffer<enc> buf2, Buffer<enc> buf3, npy_int64 count,
               Buffer<enc> out)
{
    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();
    size_t len3 = buf3.num_codepoints();
    Buffer<enc> end1 = buf1 + len1;

    // Only try to replace if replacements are possible.
    if (count <= 0                      // There's nothing to replace.
        || len1 < len2                  // Input is too small to have a match.
        || (len2 <= 0 && len3 <= 0)     // Match and replacement strings both empty.
        || (len2 == len3 && buf2.strcmp(buf3) == 0)) {  // Match and replacement are the same.

        goto copy_rest;
    }

    if (len2 > 0) {
        for (npy_int64 time = 0; time < count; time++) {
            npy_intp pos = findslice_for_replace(buf1, end1 - buf1, buf2, len2);
            if (pos < 0) {
                break;
            }

            buf1.buffer_memcpy(out, pos);
            out += pos;
            buf1 += pos;

            buf3.buffer_memcpy(out, len3);
            out += len3;
            buf1 += len2;
        }
    }
    else {  // If match string empty, interleave.
        while (count > 0) {
            buf3.buffer_memcpy(out, len3);
            out += len3;

            if (--count <= 0) {
                break;
            }

            buf1.buffer_memcpy(out, 1);
            buf1 += 1;
            out += 1;
        }
    }

copy_rest:
    buf1.buffer_memcpy(out, end1 - buf1);
    out.buffer_fill_with_zeros_after_index(end1 - buf1);
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


template <bool rstrip, COMP comp, ENCODING enc>
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
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        int cmp = buf1.strcmp(buf2, rstrip);
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
    int outsize = context->descriptors[2]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        Buffer<enc> outbuf(out, outsize);
        string_add<enc>(buf1, buf2, outbuf);

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
string_replace_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;
    int elsize3 = context->descriptors[2]->elsize;
    int outsize = context->descriptors[4]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *in4 = data[3];
    char *out = data[4];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        Buffer<enc> buf3(in3, elsize3);
        Buffer<enc> outbuf(out, outsize);
        string_replace(buf1, buf2, buf3, *(npy_int64 *) in4, outbuf);

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


template <ENCODING enc>
static int
string_strip_whitespace_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;
    int outsize = context->descriptors[1]->elsize;

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in, elsize);
        Buffer<enc> outbuf(out, outsize);
        string_lrstrip_whitespace(buf, outbuf, STRIPTYPE::BOTHSTRIP);

        in += strides[0];
        out += strides[1];
    }

    return 0;
}


template <ENCODING enc>
static int
string_lstrip_whitespace_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;
    int outsize = context->descriptors[1]->elsize;

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in, elsize);
        Buffer<enc> outbuf(out, outsize);
        string_lrstrip_whitespace(buf, outbuf, STRIPTYPE::LEFTSTRIP);

        in += strides[0];
        out += strides[1];
    }

    return 0;
}


template <ENCODING enc>
static int
string_rstrip_whitespace_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;
    int outsize = context->descriptors[1]->elsize;

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in, elsize);
        Buffer<enc> outbuf(out, outsize);
        string_lrstrip_whitespace(buf, outbuf, STRIPTYPE::RIGHTSTRIP);

        in += strides[0];
        out += strides[1];
    }

    return 0;
}


template <ENCODING enc>
static int
string_strip_chars_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;
    int outsize = context->descriptors[2]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        Buffer<enc> outbuf(out, outsize);
        string_lrstrip_chars(buf1, buf2, outbuf, STRIPTYPE::BOTHSTRIP);

        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    return 0;
}


template <ENCODING enc>
static int
string_lstrip_chars_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;
    int outsize = context->descriptors[2]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        Buffer<enc> outbuf(out, outsize);
        string_lrstrip_chars(buf1, buf2, outbuf, STRIPTYPE::LEFTSTRIP);

        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    return 0;
}


template <ENCODING enc>
static int
string_rstrip_chars_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;
    int outsize = context->descriptors[2]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        Buffer<enc> outbuf(out, outsize);
        string_lrstrip_chars(buf1, buf2, outbuf, STRIPTYPE::RIGHTSTRIP);

        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
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


static NPY_CASTING
string_strip_whitespace_resolve_descriptors(
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

    Py_INCREF(loop_descrs[0]);
    loop_descrs[1] = loop_descrs[0];

    return NPY_NO_CASTING;
}


static NPY_CASTING
string_strip_chars_resolve_descriptors(
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

    Py_INCREF(loop_descrs[0]);
    loop_descrs[2] = loop_descrs[0];

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
    new_op_dtypes[2] = NPY_DT_NewRef(&PyArray_Int64DType);
    new_op_dtypes[3] = NPY_DT_NewRef(&PyArray_Int64DType);
    new_op_dtypes[4] = PyArray_DTypeFromTypeNum(NPY_DEFAULT_INT);
    return 0;
}


static int
string_replace_promoter(PyUFuncObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[0] = op_dtypes[0];

    Py_INCREF(op_dtypes[1]);
    new_op_dtypes[1] = op_dtypes[1];

    Py_INCREF(op_dtypes[2]);
    new_op_dtypes[2] = op_dtypes[2];

    new_op_dtypes[3] = PyArray_DTypeFromTypeNum(NPY_INT64);

    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[4] = op_dtypes[0];
    return 0;
}


static NPY_CASTING
string_replace_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[3]),
        PyArray_Descr *given_descrs[5],
        PyArray_Descr *loop_descrs[5],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[4] == NULL) {
        PyErr_SetString(PyExc_ValueError, "out kwarg should be given");
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
    if (loop_descrs[1] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    loop_descrs[2] = NPY_DT_CALL_ensure_canonical(given_descrs[2]);
    if (loop_descrs[2] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    loop_descrs[3] = NPY_DT_CALL_ensure_canonical(given_descrs[3]);
    if (loop_descrs[3] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    loop_descrs[4] = NPY_DT_CALL_ensure_canonical(given_descrs[4]);
    if (loop_descrs[4] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    return NPY_NO_CASTING;
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
    new_op_dtypes[2] = NPY_DT_NewRef(&PyArray_Int64DType);
    new_op_dtypes[3] = NPY_DT_NewRef(&PyArray_Int64DType);
    new_op_dtypes[4] = NPY_DT_NewRef(&PyArray_BoolDType);
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


template<bool rstrip, ENCODING enc, COMP...>
struct add_loops;

template<bool rstrip, ENCODING enc>
struct add_loops<rstrip, enc> {
    int operator()(PyObject*, PyArrayMethod_Spec*) {
        return 0;
    }
};

template<bool rstrip, ENCODING enc, COMP comp, COMP... comps>
struct add_loops<rstrip, enc, comp, comps...> {
    int operator()(PyObject* umath, PyArrayMethod_Spec* spec) {
        PyArrayMethod_StridedLoop* loop = string_comparison_loop<rstrip, comp, enc>;

        if (add_loop(umath, comp_name(comp), spec, loop) < 0) {
            return -1;
        }
        else {
            return add_loops<rstrip, enc, comps...>()(umath, spec);
        }
    }
};


static int
init_comparison(PyObject *umath)
{
    int res = -1;
    PyArray_DTypeMeta *String = &PyArray_BytesDType;
    PyArray_DTypeMeta *Unicode = &PyArray_UnicodeDType;
    PyArray_DTypeMeta *Bool = &PyArray_BoolDType;

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
    using string_looper = add_loops<false, ENCODING::ASCII, COMP::EQ, COMP::NE, COMP::LT, COMP::LE, COMP::GT, COMP::GE>;
    if (string_looper()(umath, &spec) < 0) {
        goto finish;
    }

    /* All Unicode loops */
    using ucs_looper = add_loops<false, ENCODING::UTF32, COMP::EQ, COMP::NE, COMP::LT, COMP::LE, COMP::GT, COMP::GE>;
    dtypes[0] = Unicode;
    dtypes[1] = Unicode;
    if (ucs_looper()(umath, &spec) < 0) {
        goto finish;
    }

    res = 0;
  finish:
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
            dtypes[i] = NPY_DT_NewRef(&PyArray_UnicodeDType);
        }
        else if (typenums[i] == NPY_OBJECT && enc == ENCODING::ASCII) {
            dtypes[i] = NPY_DT_NewRef(&PyArray_BytesDType);
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

    dtypes[0] = dtypes[1] = dtypes[2] = NPY_OBJECT;
    dtypes[3] = NPY_INT64;
    dtypes[4] = NPY_OBJECT;
    if (init_ufunc<ENCODING::ASCII>(
            umath, "_replace", "templated_string_replace", 4, 1, dtypes,
            string_replace_loop<ENCODING::ASCII>,
            string_replace_resolve_descriptors) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "_replace", "templated_string_replace", 4, 1, dtypes,
            string_replace_loop<ENCODING::UTF32>,
            string_replace_resolve_descriptors) < 0) {
        return -1;
    }
    if (init_promoter(umath, "_replace", 4, 1, string_replace_promoter) < 0) {
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

    dtypes[0] = dtypes[1] = NPY_OBJECT;
    if (init_ufunc<ENCODING::ASCII>(
            umath, "_lstrip_whitespace", "templated_string_lstrip", 1, 1, dtypes,
            string_lstrip_whitespace_loop<ENCODING::ASCII>,
            string_strip_whitespace_resolve_descriptors) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "_lstrip_whitespace", "templated_string_lstrip", 1, 1, dtypes,
            string_lstrip_whitespace_loop<ENCODING::UTF32>,
            string_strip_whitespace_resolve_descriptors) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::ASCII>(
            umath, "_rstrip_whitespace", "templated_string_rstrip", 1, 1, dtypes,
            string_rstrip_whitespace_loop<ENCODING::ASCII>,
            string_strip_whitespace_resolve_descriptors) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "_rstrip_whitespace", "templated_string_rstrip", 1, 1, dtypes,
            string_rstrip_whitespace_loop<ENCODING::UTF32>,
            string_strip_whitespace_resolve_descriptors) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::ASCII>(
            umath, "_strip_whitespace", "templated_string_strip", 1, 1, dtypes,
            string_strip_whitespace_loop<ENCODING::ASCII>,
            string_strip_whitespace_resolve_descriptors) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "_strip_whitespace", "templated_string_strip", 1, 1, dtypes,
            string_strip_whitespace_loop<ENCODING::UTF32>,
            string_strip_whitespace_resolve_descriptors) < 0) {
        return -1;
    }

    dtypes[0] = dtypes[1] = dtypes[2] = NPY_OBJECT;
    if (init_ufunc<ENCODING::ASCII>(
            umath, "_lstrip_chars", "templated_string_lstrip", 2, 1, dtypes,
            string_lstrip_chars_loop<ENCODING::ASCII>,
            string_strip_chars_resolve_descriptors) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "_lstrip_chars", "templated_string_lstrip", 2, 1, dtypes,
            string_lstrip_chars_loop<ENCODING::UTF32>,
            string_strip_chars_resolve_descriptors) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::ASCII>(
            umath, "_rstrip_chars", "templated_string_rstrip", 2, 1, dtypes,
            string_rstrip_chars_loop<ENCODING::ASCII>,
            string_strip_chars_resolve_descriptors) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "_rstrip_chars", "templated_string_rstrip", 2, 1, dtypes,
            string_rstrip_chars_loop<ENCODING::UTF32>,
            string_strip_chars_resolve_descriptors) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::ASCII>(
            umath, "_strip_chars", "templated_string_strip", 2, 1, dtypes,
            string_strip_chars_loop<ENCODING::ASCII>,
            string_strip_chars_resolve_descriptors) < 0) {
        return -1;
    }
    if (init_ufunc<ENCODING::UTF32>(
            umath, "_strip_chars", "templated_string_strip", 2, 1, dtypes,
            string_strip_chars_loop<ENCODING::UTF32>,
            string_strip_chars_resolve_descriptors) < 0) {
        return -1;
    }

    return 0;
}


template <bool rstrip, ENCODING enc>
static PyArrayMethod_StridedLoop *
get_strided_loop(int comp)
{
    switch (comp) {
        case Py_EQ:
            return string_comparison_loop<rstrip, COMP::EQ, enc>;
        case Py_NE:
            return string_comparison_loop<rstrip, COMP::NE, enc>;
        case Py_LT:
            return string_comparison_loop<rstrip, COMP::LT, enc>;
        case Py_LE:
            return string_comparison_loop<rstrip, COMP::LE, enc>;
        case Py_GT:
            return string_comparison_loop<rstrip, COMP::GT, enc>;
        case Py_GE:
            return string_comparison_loop<rstrip, COMP::GE, enc>;
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
            strided_loop = get_strided_loop<false, ENCODING::ASCII>(cmp_op);
        }
        else {
            strided_loop = get_strided_loop<false, ENCODING::UTF32>(cmp_op);
        }
    }
    else {
        if (descrs[0]->type_num != NPY_UNICODE) {
            strided_loop = get_strided_loop<true, ENCODING::ASCII>(cmp_op);
        }
        else {
            strided_loop = get_strided_loop<true, ENCODING::UTF32>(cmp_op);
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
