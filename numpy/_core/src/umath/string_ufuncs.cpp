#include <Python.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"

#include "numpyos.h"
#include "dispatching.h"
#include "dtypemeta.h"
#include "convert_datatype.h"
#include "gil_utils.h"
#include "templ_common.h" /* for npy_mul_size_with_overflow_size_t */

#include "string_ufuncs.h"
#include "string_fastsearch.h"
#include "string_buffer.h"


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
string_str_len_loop(PyArrayMethod_Context *context,
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
using buffer_method = bool (Buffer<enc>::*)();

template <ENCODING enc>
static int
string_unary_loop(PyArrayMethod_Context *context,
                  char *const data[], npy_intp const dimensions[],
                  npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    buffer_method<enc> is_it = *(buffer_method<enc> *)(context->method->static_data);
    int elsize = context->descriptors[0]->elsize;

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in, elsize);
        *(npy_bool *)out = (buf.*is_it)();

        in += strides[0];
        out += strides[1];
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


template <ENCODING enc>
static inline int
string_multiply(Buffer<enc> buf1, npy_int64 reps, Buffer<enc> out)
{
    size_t len1 = buf1.num_codepoints();
    if (reps < 1 || len1 == 0) {
        out.buffer_fill_with_zeros_after_index(0);
        return 0;
    }

    if (len1 == 1) {
        out.buffer_memset(*buf1, reps);
        out.buffer_fill_with_zeros_after_index(reps);
        return 0;
    }

    size_t newlen;
    if (NPY_UNLIKELY(npy_mul_with_overflow_size_t(&newlen, reps, len1) != 0) || newlen > PY_SSIZE_T_MAX) {
        return -1;
    }

    size_t pad = 0;
    size_t width = out.buffer_width();
    if (width < newlen) {
        reps = width / len1;
        pad = width % len1;
    }

    for (npy_int64 i = 0; i < reps; i++) {
        buf1.buffer_memcpy(out, len1);
        out += len1;
    }

    buf1.buffer_memcpy(out, pad);
    out += pad;

    out.buffer_fill_with_zeros_after_index(0);

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
string_multiply_strint_loop(PyArrayMethod_Context *context,
                char *const data[], npy_intp const dimensions[],
                npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;
    int outsize = context->descriptors[2]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in1, elsize);
        Buffer<enc> outbuf(out, outsize);
        if (NPY_UNLIKELY(string_multiply<enc>(buf, *(npy_int64 *)in2, outbuf) < 0)) {
            npy_gil_error(PyExc_OverflowError, "Overflow detected in string multiply");
        }

        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    return 0;
}


template <ENCODING enc>
static int
string_multiply_intstr_loop(PyArrayMethod_Context *context,
                char *const data[], npy_intp const dimensions[],
                npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[1]->elsize;
    int outsize = context->descriptors[2]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in2, elsize);
        Buffer<enc> outbuf(out, outsize);
        if (NPY_UNLIKELY(string_multiply<enc>(buf, *(npy_int64 *)in1, outbuf) < 0)) {
            npy_gil_error(PyExc_OverflowError, "Overflow detected in string multiply");
        }

        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    return 0;
}


template <ENCODING enc>
using findlike_function = npy_intp (*)(Buffer<enc>, Buffer<enc>,
                                       npy_int64, npy_int64);

template <ENCODING enc>
static int
string_findlike_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    findlike_function<enc> function = *(findlike_function<enc>)(context->method->static_data);
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
        npy_intp idx = function(buf1, buf2, *(npy_int64 *)in3, *(npy_int64 *)in4);
        if (idx == -2) {
            return -1;
        }
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
string_startswith_endswith_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    STARTPOSITION startposition = *(STARTPOSITION *)(context->method->static_data);
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
                                        startposition);
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
string_lrstrip_whitespace_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    STRIPTYPE striptype = *(STRIPTYPE *)(context->method->static_data);
    int elsize = context->descriptors[0]->elsize;
    int outsize = context->descriptors[1]->elsize;

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in, elsize);
        Buffer<enc> outbuf(out, outsize);
        string_lrstrip_whitespace(buf, outbuf, striptype);

        in += strides[0];
        out += strides[1];
    }

    return 0;
}


template <ENCODING enc>
static int
string_lrstrip_chars_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    STRIPTYPE striptype = *(STRIPTYPE *)(context->method->static_data);
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
        string_lrstrip_chars(buf1, buf2, outbuf, striptype);

        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    return 0;
}


template <ENCODING enc>
static int
string_expandtabs_length_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in1, elsize);
        *(npy_intp *)out = string_expandtabs_length(buf, *(npy_int64 *)in2);

        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    return 0;
}


template <ENCODING enc>
static int
string_expandtabs_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;
    int outsize = context->descriptors[2]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in1, elsize);
        Buffer<enc> outbuf(out, outsize);
        npy_intp new_len = string_expandtabs(buf, *(npy_int64 *)in2, outbuf);
        outbuf.buffer_fill_with_zeros_after_index(new_len);

        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    return 0;
}


template <ENCODING bufferenc, ENCODING fillenc>
static int
string_center_ljust_rjust_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    JUSTPOSITION pos = *(JUSTPOSITION *)(context->method->static_data);
    int elsize1 = context->descriptors[0]->elsize;
    int elsize3 = context->descriptors[2]->elsize;
    int outsize = context->descriptors[3]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *out = data[3];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<bufferenc> buf(in1, elsize1);
        Buffer<fillenc> fill(in3, elsize3);
        Buffer<bufferenc> outbuf(out, outsize);
        if (bufferenc == ENCODING::ASCII && fillenc == ENCODING::UTF32 && *fill > 0x7F) {
            npy_gil_error(PyExc_ValueError, "non-ascii fill character is not allowed when buffer is ascii");
            return -1;
        }
        npy_intp len = string_pad(buf, *(npy_int64 *)in2, *fill, pos, outbuf);
        if (len < 0) {
            return -1;
        }
        outbuf.buffer_fill_with_zeros_after_index(len);

        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        out += strides[3];
    }

    return 0;
}


template <ENCODING enc>
static int
string_zfill_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;
    int outsize = context->descriptors[2]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf(in1, elsize);
        Buffer<enc> outbuf(out, outsize);
        npy_intp newlen = string_zfill(buf, *(npy_int64 *)in2, outbuf);
        if (newlen < 0) {
            return -1;
        }
        outbuf.buffer_fill_with_zeros_after_index(newlen);

        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    return 0;
}


template <ENCODING enc>
static int
string_partition_index_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    STARTPOSITION startposition = *(STARTPOSITION *)(context->method->static_data);
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;
    int outsize1 = context->descriptors[3]->elsize;
    int outsize2 = context->descriptors[4]->elsize;
    int outsize3 = context->descriptors[5]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *out1 = data[3];
    char *out2 = data[4];
    char *out3 = data[5];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        Buffer<enc> outbuf1(out1, outsize1);
        Buffer<enc> outbuf2(out2, outsize2);
        Buffer<enc> outbuf3(out3, outsize3);

        npy_intp final_len1, final_len2, final_len3;
        string_partition(buf1, buf2, *(npy_int64 *)in3, outbuf1, outbuf2, outbuf3,
                         &final_len1, &final_len2, &final_len3, startposition);
        if (final_len1 < 0 || final_len2 < 0 || final_len3 < 0) {
            return -1;
        }
        outbuf1.buffer_fill_with_zeros_after_index(final_len1);
        outbuf2.buffer_fill_with_zeros_after_index(final_len2);
        outbuf3.buffer_fill_with_zeros_after_index(final_len3);

        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        out1 += strides[3];
        out2 += strides[4];
        out3 += strides[5];
    }

    return 0;
}


template <ENCODING enc>
static int
string_slice_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int insize = context->descriptors[0]->elsize;
    int outsize = context->descriptors[4]->elsize;

    char *in_ptr = data[0];
    char *start_ptr = data[1];
    char *stop_ptr = data[2];
    char *step_ptr = data[3];
    char *out_ptr = data[4];

    npy_intp N = dimensions[0];

    while (N--) {
        Buffer<enc> inbuf(in_ptr, insize);
        Buffer<enc> outbuf(out_ptr, outsize);

        // get the slice
        npy_intp start = *(npy_intp*)start_ptr;
        npy_intp stop = *(npy_intp*)stop_ptr;
        npy_intp step = *(npy_intp*)step_ptr;

        // adjust slice to string length in codepoints
        // and handle negative indices
        size_t num_codepoints = inbuf.num_codepoints();
        npy_intp slice_length = PySlice_AdjustIndices(num_codepoints, &start, &stop, step);

        // iterate over slice and copy each character of the string
        inbuf.advance_chars_or_bytes(start);
        for (npy_intp i = 0; i < slice_length; i++) {
            // copy one codepoint
            inbuf.buffer_memcpy(outbuf, 1);

            // Move in inbuf by step.
            inbuf += step;

            // Move in outbuf by the number of chars or bytes written
            outbuf.advance_chars_or_bytes(1);
        }

        // fill remaining outbuf with zero bytes
        for (char *tmp = outbuf.buf; tmp < outbuf.after; tmp++) {
            *tmp = 0;
        }

        // Go to the next array element
        in_ptr += strides[0];
        start_ptr += strides[1];
        stop_ptr += strides[2];
        step_ptr += strides[3];
        out_ptr += strides[4];
    }

    return 0;
}


/* Resolve descriptors & promoter functions */

static NPY_CASTING
string_addition_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    npy_intp result_itemsize = given_descrs[0]->elsize + given_descrs[1]->elsize;

    /* NOTE: elsize can fit more than MAX_INT, but some code may still use ints */
    if (result_itemsize > NPY_MAX_INT || result_itemsize < 0) {
            npy_intp length = result_itemsize;
            if (given_descrs[0]->type == NPY_UNICODE) {
                length /= 4;
            }
            PyErr_Format(PyExc_TypeError,
                    "addition result string of length %zd is too large to store inside array.",
                    length);
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
    if (loop_descrs[1] == NULL) {
        Py_DECREF(loop_descrs[0]);
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    loop_descrs[2] = PyArray_DescrNew(loop_descrs[0]);
    if (loop_descrs[2] == NULL) {
        Py_DECREF(loop_descrs[0]);
        Py_DECREF(loop_descrs[1]);
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    loop_descrs[2]->elsize += loop_descrs[1]->elsize;

    return NPY_NO_CASTING;
}


static NPY_CASTING
string_multiply_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[2] == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "The 'out' kwarg is necessary when using the string multiply ufunc "
            "directly. Use numpy.strings.multiply to multiply strings without "
            "specifying 'out'.");
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

    return NPY_NO_CASTING;
}


static NPY_CASTING
string_strip_whitespace_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
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
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3],
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
string_findlike_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
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
string_replace_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
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
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[5]),
        PyArray_Descr *const given_descrs[5],
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
string_startswith_endswith_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
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


static int
string_expandtabs_length_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[0] = op_dtypes[0];
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_Int64DType);
    new_op_dtypes[2] = PyArray_DTypeFromTypeNum(NPY_DEFAULT_INT);
    return 0;
}


static int
string_expandtabs_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[0] = op_dtypes[0];
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_Int64DType);
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[2] = op_dtypes[0];
    return 0;
}


static NPY_CASTING
string_expandtabs_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[2] == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "The 'out' kwarg is necessary. Use numpy.strings.expandtabs without it.");
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

    return NPY_NO_CASTING;
}


static int
string_center_ljust_rjust_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[0] = op_dtypes[0];
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_Int64DType);
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[2] = op_dtypes[0];
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[3] = op_dtypes[0];
    return 0;
}


static NPY_CASTING
string_center_ljust_rjust_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[5],
        PyArray_Descr *loop_descrs[5],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[3] == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "The 'out' kwarg is necessary. Use the version in numpy.strings without it.");
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

    return NPY_NO_CASTING;
}


static int
string_zfill_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[0] = op_dtypes[0];
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_Int64DType);
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[2] = op_dtypes[0];
    return 0;
}


static NPY_CASTING
string_zfill_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[2] == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "The 'out' kwarg is necessary. Use numpy.strings.zfill without it.");
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

    return NPY_NO_CASTING;
}


static int
string_partition_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[0] = op_dtypes[0];
    Py_INCREF(op_dtypes[1]);
    new_op_dtypes[1] = op_dtypes[1];

    new_op_dtypes[2] = NPY_DT_NewRef(&PyArray_Int64DType);

    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[3] = op_dtypes[0];
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[4] = op_dtypes[0];
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[5] = op_dtypes[0];
    return 0;
}


static NPY_CASTING
string_partition_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (!given_descrs[3] || !given_descrs[4] || !given_descrs[5]) {
        PyErr_Format(PyExc_TypeError,
            "The '%s' ufunc requires the 'out' keyword to be set. The "
            "python wrapper in numpy.strings can be used without the "
            "out keyword.", self->name);
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    for (int i = 0; i < 6; i++) {
        loop_descrs[i] = NPY_DT_CALL_ensure_canonical(given_descrs[i]);
        if (!loop_descrs[i]) {
            return _NPY_ERROR_OCCURRED_IN_CAST;
        }
    }

    return NPY_NO_CASTING;
}


static int
string_slice_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[0] = op_dtypes[0];
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_IntpDType);
    new_op_dtypes[2] = NPY_DT_NewRef(&PyArray_IntpDType);
    new_op_dtypes[3] = NPY_DT_NewRef(&PyArray_IntpDType);
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[4] = op_dtypes[0];
    return 0;
}

static NPY_CASTING
string_slice_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[5]),
        PyArray_Descr *const given_descrs[5],
        PyArray_Descr *loop_descrs[5],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[4]) {
        PyErr_Format(PyExc_TypeError,
                     "The '%s' ufunc does not "
                     "currently support the 'out' keyword",
                     self->name);
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    for (int i = 0; i < 4; i++) {
        loop_descrs[i] = NPY_DT_CALL_ensure_canonical(given_descrs[i]);
        if (loop_descrs[i] == NULL) {
            return _NPY_ERROR_OCCURRED_IN_CAST;
        }
    }

    loop_descrs[4] = PyArray_DescrNew(loop_descrs[0]);
    if (loop_descrs[4] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    loop_descrs[4]->elsize = loop_descrs[0]->elsize;

    return NPY_NO_CASTING;
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
              PyArrayMethod_PromoterFunction promoter)
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


static int
init_ufunc(PyObject *umath, const char *name, int nin, int nout,
           NPY_TYPES *typenums, ENCODING enc, PyArrayMethod_StridedLoop loop,
           PyArrayMethod_ResolveDescriptors resolve_descriptors,
           void *static_data)
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

    PyType_Slot slots[4];
    slots[0] = {NPY_METH_strided_loop, nullptr};
    slots[1] = {_NPY_METH_static_data, static_data};
    slots[3] = {0, nullptr};
    if (resolve_descriptors != NULL) {
        slots[2] = {NPY_METH_resolve_descriptors, (void *) resolve_descriptors};
    }
    else {
        slots[2] = {0, nullptr};
    }

    char loop_name[256] = {0};
    snprintf(loop_name, sizeof(loop_name), "templated_string_%s", name);

    PyArrayMethod_Spec spec = {};
    spec.name = loop_name;
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


/*
 * This is a variant of init_ufunc that allows for mixed string dtypes
 * in its parameters. Instead of having NPY_OBJECT be a sentinel for a
 * fixed dtype, here the typenums are always the correct ones.
 */
static int
init_mixed_type_ufunc(PyObject *umath, const char *name, int nin, int nout,
           NPY_TYPES *typenums, PyArrayMethod_StridedLoop loop,
           PyArrayMethod_ResolveDescriptors resolve_descriptors,
           void *static_data)
{
    int res = -1;

    PyArray_DTypeMeta **dtypes = (PyArray_DTypeMeta **) PyMem_Malloc(
        (nin + nout) * sizeof(PyArray_DTypeMeta *));
    if (dtypes == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    for (int i = 0; i < nin+nout; i++) {
        dtypes[i] = PyArray_DTypeFromTypeNum(typenums[i]);
    }

    PyType_Slot slots[4];
    slots[0] = {NPY_METH_strided_loop, nullptr};
    slots[1] = {_NPY_METH_static_data, static_data};
    slots[3] = {0, nullptr};
    if (resolve_descriptors != NULL) {
        slots[2] = {NPY_METH_resolve_descriptors, (void *) resolve_descriptors};
    }
    else {
        slots[2] = {0, nullptr};
    }

    char loop_name[256] = {0};
    snprintf(loop_name, sizeof(loop_name), "templated_string_%s", name);

    PyArrayMethod_Spec spec = {};
    spec.name = loop_name;
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
    NPY_TYPES dtypes[] = {NPY_STRING, NPY_STRING, NPY_STRING, NPY_STRING, NPY_STRING, NPY_STRING};

    if (init_comparison(umath) < 0) {
        return -1;
    }

    // We use NPY_OBJECT as a sentinel value here, and this will be replaced by the
    // corresponding string dtype (either NPY_STRING or NPY_UNICODE).
    dtypes[0] = dtypes[1] = dtypes[2] = NPY_OBJECT;
    if (init_ufunc(
            umath, "add", 2, 1, dtypes, ENCODING::ASCII,
            string_add_loop<ENCODING::ASCII>, string_addition_resolve_descriptors,
            NULL) < 0) {
        return -1;
    }
    if (init_ufunc(
            umath, "add", 2, 1, dtypes, ENCODING::UTF32,
            string_add_loop<ENCODING::UTF32>, string_addition_resolve_descriptors,
            NULL) < 0) {
        return -1;
    }

    dtypes[0] = dtypes[2] = NPY_OBJECT;
    dtypes[1] = NPY_INT64;
    if (init_ufunc(
            umath, "multiply", 2, 1, dtypes, ENCODING::ASCII,
            string_multiply_strint_loop<ENCODING::ASCII>, string_multiply_resolve_descriptors,
            NULL) < 0) {
        return -1;
    }
    if (init_ufunc(
            umath, "multiply", 2, 1, dtypes, ENCODING::UTF32,
            string_multiply_strint_loop<ENCODING::UTF32>, string_multiply_resolve_descriptors,
            NULL) < 0) {
        return -1;
    }

    dtypes[1] = dtypes[2] = NPY_OBJECT;
    dtypes[0] = NPY_INT64;
    if (init_ufunc(
            umath, "multiply", 2, 1, dtypes, ENCODING::ASCII,
            string_multiply_intstr_loop<ENCODING::ASCII>, string_multiply_resolve_descriptors,
            NULL) < 0) {
        return -1;
    }
    if (init_ufunc(
            umath, "multiply", 2, 1, dtypes, ENCODING::UTF32,
            string_multiply_intstr_loop<ENCODING::UTF32>, string_multiply_resolve_descriptors,
            NULL) < 0) {
        return -1;
    }

    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_DEFAULT_INT;
    if (init_ufunc(
            umath, "str_len", 1, 1, dtypes, ENCODING::ASCII,
            string_str_len_loop<ENCODING::ASCII>, NULL, NULL) < 0) {
        return -1;
    }
    if (init_ufunc(
            umath, "str_len", 1, 1, dtypes, ENCODING::UTF32,
            string_str_len_loop<ENCODING::UTF32>, NULL, NULL) < 0) {
        return -1;
    }

    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_BOOL;

    const char *const unary_buffer_method_names[] = {
        "isalpha", "isalnum", "isdigit", "isspace", "islower",
        "isupper", "istitle", "isdecimal", "isnumeric",
    };

    static buffer_method<ENCODING::ASCII> unary_buffer_ascii_methods[] = {
        &Buffer<ENCODING::ASCII>::isalpha,
        &Buffer<ENCODING::ASCII>::isalnum,
        &Buffer<ENCODING::ASCII>::isdigit,
        &Buffer<ENCODING::ASCII>::isspace,
        &Buffer<ENCODING::ASCII>::islower,
        &Buffer<ENCODING::ASCII>::isupper,
        &Buffer<ENCODING::ASCII>::istitle,
    };

    static buffer_method<ENCODING::UTF32> unary_buffer_utf32_methods[] = {
        &Buffer<ENCODING::UTF32>::isalpha,
        &Buffer<ENCODING::UTF32>::isalnum,
        &Buffer<ENCODING::UTF32>::isdigit,
        &Buffer<ENCODING::UTF32>::isspace,
        &Buffer<ENCODING::UTF32>::islower,
        &Buffer<ENCODING::UTF32>::isupper,
        &Buffer<ENCODING::UTF32>::istitle,
        &Buffer<ENCODING::UTF32>::isdecimal,
        &Buffer<ENCODING::UTF32>::isnumeric,
    };

    for (int i = 0; i < 9; i++) {
        if (i < 7) { // isdecimal & isnumeric do not support ASCII
            if (init_ufunc(
                    umath, unary_buffer_method_names[i], 1, 1, dtypes, ENCODING::ASCII,
                    string_unary_loop<ENCODING::ASCII>, NULL,
                    &unary_buffer_ascii_methods[i]) < 0) {
                return -1;
            }
        }

        if (init_ufunc(
                umath, unary_buffer_method_names[i], 1, 1, dtypes, ENCODING::UTF32,
                string_unary_loop<ENCODING::UTF32>, NULL,
                &unary_buffer_utf32_methods[i]) < 0) {
            return -1;
        }
    }

    dtypes[0] = dtypes[1] = NPY_OBJECT;
    dtypes[2] = dtypes[3] = NPY_INT64;
    dtypes[4] = NPY_DEFAULT_INT;

    const char* findlike_names[] = {
        "find", "rfind", "index", "rindex", "count",
    };

    findlike_function<ENCODING::ASCII> findlike_ascii_functions[] = {
        string_find<ENCODING::ASCII>,
        string_rfind<ENCODING::ASCII>,
        string_index<ENCODING::ASCII>,
        string_rindex<ENCODING::ASCII>,
        string_count<ENCODING::ASCII>,
    };

    findlike_function<ENCODING::UTF32> findlike_utf32_functions[] = {
        string_find<ENCODING::UTF32>,
        string_rfind<ENCODING::UTF32>,
        string_index<ENCODING::UTF32>,
        string_rindex<ENCODING::UTF32>,
        string_count<ENCODING::UTF32>,
    };

    for (int j = 0; j < 5; j++) {

        if (init_ufunc(
                umath, findlike_names[j], 4, 1, dtypes, ENCODING::ASCII,
                string_findlike_loop<ENCODING::ASCII>, NULL,
                (void *) findlike_ascii_functions[j]) < 0) {
            return -1;
        }

        if (init_ufunc(
                umath, findlike_names[j], 4, 1, dtypes, ENCODING::UTF32,
                string_findlike_loop<ENCODING::UTF32>, NULL,
                (void *) findlike_utf32_functions[j]) < 0) {
            return -1;
        }

        if (init_promoter(umath, findlike_names[j], 4, 1,
                string_findlike_promoter) < 0) {
            return -1;
        }
    }

    dtypes[0] = dtypes[1] = dtypes[2] = NPY_OBJECT;
    dtypes[3] = NPY_INT64;
    dtypes[4] = NPY_OBJECT;
    if (init_ufunc(
            umath, "_replace", 4, 1, dtypes, ENCODING::ASCII,
            string_replace_loop<ENCODING::ASCII>,
            string_replace_resolve_descriptors, NULL) < 0) {
        return -1;
    }
    if (init_ufunc(
            umath, "_replace", 4, 1, dtypes, ENCODING::UTF32,
            string_replace_loop<ENCODING::UTF32>,
            string_replace_resolve_descriptors, NULL) < 0) {
        return -1;
    }
    if (init_promoter(umath, "_replace", 4, 1, string_replace_promoter) < 0) {
        return -1;
    }

    dtypes[0] = dtypes[1] = NPY_OBJECT;
    dtypes[2] = dtypes[3] = NPY_INT64;
    dtypes[4] = NPY_BOOL;

    const char *const startswith_endswith_names[] = {
        "startswith", "endswith"
    };

    static STARTPOSITION startpositions[] = {
        STARTPOSITION::FRONT, STARTPOSITION::BACK
    };

    for (int i = 0; i < 2; i++) {
        if (init_ufunc(
                umath, startswith_endswith_names[i], 4, 1, dtypes, ENCODING::ASCII,
                string_startswith_endswith_loop<ENCODING::ASCII>,
                NULL, &startpositions[i]) < 0) {
            return -1;
        }
        if (init_ufunc(
                umath, startswith_endswith_names[i], 4, 1, dtypes, ENCODING::UTF32,
                string_startswith_endswith_loop<ENCODING::UTF32>,
                NULL, &startpositions[i]) < 0) {
            return -1;
        }
        if (init_promoter(umath, startswith_endswith_names[i], 4, 1,
                string_startswith_endswith_promoter) < 0) {
            return -1;
        }
    }

    dtypes[0] = dtypes[1] = NPY_OBJECT;

    const char *const strip_whitespace_names[] = {
        "_lstrip_whitespace", "_rstrip_whitespace", "_strip_whitespace"
    };

    static STRIPTYPE striptypes[] = {
        STRIPTYPE::LEFTSTRIP, STRIPTYPE::RIGHTSTRIP, STRIPTYPE::BOTHSTRIP
    };

    for (int i = 0; i < 3; i++) {
        if (init_ufunc(
                umath, strip_whitespace_names[i], 1, 1, dtypes, ENCODING::ASCII,
                string_lrstrip_whitespace_loop<ENCODING::ASCII>,
                string_strip_whitespace_resolve_descriptors,
                &striptypes[i]) < 0) {
            return -1;
        }
        if (init_ufunc(
                umath, strip_whitespace_names[i], 1, 1, dtypes, ENCODING::UTF32,
                string_lrstrip_whitespace_loop<ENCODING::UTF32>,
                string_strip_whitespace_resolve_descriptors,
                &striptypes[i]) < 0) {
            return -1;
        }
    }

    dtypes[0] = dtypes[1] = dtypes[2] = NPY_OBJECT;

    const char *const strip_chars_names[] = {
        "_lstrip_chars", "_rstrip_chars", "_strip_chars"
    };

    for (int i = 0; i < 3; i++) {
        if (init_ufunc(
                umath, strip_chars_names[i], 2, 1, dtypes, ENCODING::ASCII,
                string_lrstrip_chars_loop<ENCODING::ASCII>,
                string_strip_chars_resolve_descriptors,
                &striptypes[i]) < 0) {
            return -1;
        }
        if (init_ufunc(
                umath, strip_chars_names[i], 2, 1, dtypes, ENCODING::UTF32,
                string_lrstrip_chars_loop<ENCODING::UTF32>,
                string_strip_chars_resolve_descriptors,
                &striptypes[i]) < 0) {
            return -1;
        }
    }

    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_INT64;
    dtypes[2] = NPY_DEFAULT_INT;
    if (init_ufunc(
            umath, "_expandtabs_length", 2, 1, dtypes, ENCODING::ASCII,
            string_expandtabs_length_loop<ENCODING::ASCII>, NULL, NULL) < 0) {
        return -1;
    }
    if (init_ufunc(
            umath, "_expandtabs_length", 2, 1, dtypes, ENCODING::UTF32,
            string_expandtabs_length_loop<ENCODING::UTF32>, NULL, NULL) < 0) {
        return -1;
    }
    if (init_promoter(umath, "_expandtabs_length", 2, 1, string_expandtabs_length_promoter) < 0) {
        return -1;
    }

    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_INT64;
    dtypes[2] = NPY_OBJECT;
    if (init_ufunc(
            umath, "_expandtabs", 2, 1, dtypes, ENCODING::ASCII,
            string_expandtabs_loop<ENCODING::ASCII>,
            string_expandtabs_resolve_descriptors, NULL) < 0) {
        return -1;
    }
    if (init_ufunc(
            umath, "_expandtabs", 2, 1, dtypes, ENCODING::UTF32,
            string_expandtabs_loop<ENCODING::UTF32>,
            string_expandtabs_resolve_descriptors, NULL) < 0) {
        return -1;
    }
    if (init_promoter(umath, "_expandtabs", 2, 1, string_expandtabs_promoter) < 0) {
        return -1;
    }

    dtypes[1] = NPY_INT64;

    const char *const center_ljust_rjust_names[] = {
        "_center", "_ljust", "_rjust"
    };

    static JUSTPOSITION padpositions[] = {
        JUSTPOSITION::CENTER, JUSTPOSITION::LEFT, JUSTPOSITION::RIGHT
    };

    for (int i = 0; i < 3; i++) {
        dtypes[0] = NPY_STRING;
        dtypes[2] = NPY_STRING;
        dtypes[3] = NPY_STRING;
        if (init_mixed_type_ufunc(
                umath, center_ljust_rjust_names[i], 3, 1, dtypes,
                string_center_ljust_rjust_loop<ENCODING::ASCII, ENCODING::ASCII>,
                string_center_ljust_rjust_resolve_descriptors,
                &padpositions[i]) < 0) {
            return -1;
        }
        dtypes[0] = NPY_STRING;
        dtypes[2] = NPY_UNICODE;
        dtypes[3] = NPY_STRING;
        if (init_mixed_type_ufunc(
                umath, center_ljust_rjust_names[i], 3, 1, dtypes,
                string_center_ljust_rjust_loop<ENCODING::ASCII, ENCODING::UTF32>,
                string_center_ljust_rjust_resolve_descriptors,
                &padpositions[i]) < 0) {
            return -1;
        }
        dtypes[0] = NPY_UNICODE;
        dtypes[2] = NPY_UNICODE;
        dtypes[3] = NPY_UNICODE;
        if (init_mixed_type_ufunc(
                umath, center_ljust_rjust_names[i], 3, 1, dtypes,
                string_center_ljust_rjust_loop<ENCODING::UTF32, ENCODING::UTF32>,
                string_center_ljust_rjust_resolve_descriptors,
                &padpositions[i]) < 0) {
            return -1;
        }
        dtypes[0] = NPY_UNICODE;
        dtypes[2] = NPY_STRING;
        dtypes[3] = NPY_UNICODE;
        if (init_mixed_type_ufunc(
                umath, center_ljust_rjust_names[i], 3, 1, dtypes,
                string_center_ljust_rjust_loop<ENCODING::UTF32, ENCODING::ASCII>,
                string_center_ljust_rjust_resolve_descriptors,
                &padpositions[i]) < 0) {
            return -1;
        }
        if (init_promoter(umath, center_ljust_rjust_names[i], 3, 1,
                string_center_ljust_rjust_promoter) < 0) {
            return -1;
        }
    }

    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_INT64;
    dtypes[2] = NPY_OBJECT;
    if (init_ufunc(
            umath, "_zfill", 2, 1, dtypes, ENCODING::ASCII,
            string_zfill_loop<ENCODING::ASCII>,
            string_zfill_resolve_descriptors, NULL) < 0) {
        return -1;
    }
    if (init_ufunc(
            umath, "_zfill", 2, 1, dtypes, ENCODING::UTF32,
            string_zfill_loop<ENCODING::UTF32>,
            string_zfill_resolve_descriptors, NULL) < 0) {
        return -1;
    }
    if (init_promoter(umath, "_zfill", 2, 1, string_zfill_promoter) < 0) {
        return -1;
    }

    dtypes[0] = dtypes[1] = dtypes[3] = dtypes[4] = dtypes[5] = NPY_OBJECT;
    dtypes[2] = NPY_INT64;

    const char *const partition_names[] = {"_partition_index", "_rpartition_index"};

    static STARTPOSITION partition_startpositions[] = {
        STARTPOSITION::FRONT, STARTPOSITION::BACK
    };

    for (int i = 0; i < 2; i++) {
        if (init_ufunc(
                umath, partition_names[i], 3, 3, dtypes, ENCODING::ASCII,
                string_partition_index_loop<ENCODING::ASCII>,
                string_partition_resolve_descriptors, &partition_startpositions[i]) < 0) {
            return -1;
        }
        if (init_ufunc(
                umath, partition_names[i], 3, 3, dtypes, ENCODING::UTF32,
                string_partition_index_loop<ENCODING::UTF32>,
                string_partition_resolve_descriptors, &partition_startpositions[i]) < 0) {
            return -1;
        }
        if (init_promoter(umath, partition_names[i], 3, 3,
                string_partition_promoter) < 0) {
            return -1;
        }
    }

    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_INTP;
    dtypes[2] = NPY_INTP;
    dtypes[3] = NPY_INTP;
    dtypes[4] = NPY_OBJECT;
    if (init_ufunc(
            umath, "_slice", 4, 1, dtypes, ENCODING::ASCII,
            string_slice_loop<ENCODING::ASCII>,
            string_slice_resolve_descriptors, NULL) < 0) {
        return -1;
    }
    if (init_ufunc(
            umath, "_slice", 4, 1, dtypes, ENCODING::UTF32,
            string_slice_loop<ENCODING::UTF32>,
            string_slice_resolve_descriptors, NULL) < 0) {
        return -1;
    }
    if (init_promoter(umath, "_slice", 4, 1,
            string_slice_promoter) < 0) {
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
