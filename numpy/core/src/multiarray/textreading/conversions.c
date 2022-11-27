
#include <Python.h>

#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "lowlevel_strided_loops.h"

#include "conversions.h"
#include "str_to_int.h"

#include "array_coercion.h"


/*
 * Coercion to boolean is done via integer right now.
 */
NPY_NO_EXPORT int
npy_to_bool(PyArray_Descr *NPY_UNUSED(descr),
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *NPY_UNUSED(pconfig))
{
    int64_t res;
    if (str_to_int64(str, end, INT64_MIN, INT64_MAX, &res) < 0) {
        return -1;
    }
    *dataptr = (char)(res != 0);
    return 0;
}


/*
 * In order to not pack a whole copy of a floating point parser, we copy the
 * result into ascii and call the Python one.  Float parsing isn't super quick
 * so this is not terrible, but avoiding it would speed up things.
 *
 * Also note that parsing the first float of a complex will copy the whole
 * string to ascii rather than just the first part.
 * TODO: A tweak of the break might be a simple mitigation there.
 *
 * @param str The UCS4 string to parse
 * @param end Pointer to the end of the string
 * @param skip_trailing_whitespace If false does not skip trailing whitespace
 *        (used by the complex parser).
 * @param result Output stored as double value.
 */
static inline int
double_from_ucs4(
        const Py_UCS4 *str, const Py_UCS4 *end,
        bool strip_whitespace, double *result, const Py_UCS4 **p_end)
{
    /* skip leading whitespace */
    if (strip_whitespace) {
        while (Py_UNICODE_ISSPACE(*str)) {
            str++;
        }
    }
    if (str == end) {
        return -1;  /* empty or only whitespace: not a floating point number */
    }

    /* We convert to ASCII for the Python parser, use stack if small: */
    char stack_buf[128];
    char *heap_buf = NULL;
    char *ascii = stack_buf;

    size_t str_len = end - str + 1;
    if (str_len > 128) {
        heap_buf = PyMem_MALLOC(str_len);
        if (heap_buf == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        ascii = heap_buf;
    }
    char *c = ascii;
    for (; str < end; str++, c++) {
        if (NPY_UNLIKELY(*str >= 128)) {
            /* Character cannot be used, ignore for end calculation and stop */
            end = str;
            break;
        }
        *c = (char)(*str);
    }
    *c = '\0';

    char *end_parsed;
    *result = PyOS_string_to_double(ascii, &end_parsed, NULL);
    /* Rewind `end` to the first UCS4 character not parsed: */
    end = end - (c - end_parsed);

    PyMem_FREE(heap_buf);

    if (*result == -1. && PyErr_Occurred()) {
        return -1;
    }

    if (strip_whitespace) {
        /* and then skip any remaining whitespace: */
        while (Py_UNICODE_ISSPACE(*end)) {
            end++;
        }
    }
    *p_end = end;
    return 0;
}


NPY_NO_EXPORT int
npy_to_float(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *NPY_UNUSED(pconfig))
{
    double double_val;
    const Py_UCS4 *p_end;
    if (double_from_ucs4(str, end, true, &double_val, &p_end) < 0) {
        return -1;
    }
    if (p_end != end) {
        return -1;
    }

    float val = (float)double_val;
    memcpy(dataptr, &val, sizeof(float));
    if (!PyArray_ISNBO(descr->byteorder)) {
        npy_bswap4_unaligned(dataptr);
    }
    return 0;
}


NPY_NO_EXPORT int
npy_to_double(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *NPY_UNUSED(pconfig))
{
    double val;
    const Py_UCS4 *p_end;
    if (double_from_ucs4(str, end, true, &val, &p_end) < 0) {
        return -1;
    }
    if (p_end != end) {
        return -1;
    }

    memcpy(dataptr, &val, sizeof(double));
    if (!PyArray_ISNBO(descr->byteorder)) {
        npy_bswap8_unaligned(dataptr);
    }
    return 0;
}


static bool
to_complex_int(
        const Py_UCS4 *item, const Py_UCS4 *token_end,
        double *p_real, double *p_imag,
        Py_UCS4 imaginary_unit, bool allow_parens)
{
    const Py_UCS4 *p_end;
    bool unmatched_opening_paren = false;

    /* Remove whitespace before the possibly leading '(' */
    while (Py_UNICODE_ISSPACE(*item)) {
        ++item;
    }
    if (allow_parens && (*item == '(')) {
        unmatched_opening_paren = true;
        ++item;
        /* Allow whitespace within the parentheses: "( 1j)" */
        while (Py_UNICODE_ISSPACE(*item)) {
            ++item;
        }
    }
    if (double_from_ucs4(item, token_end, false, p_real, &p_end) < 0) {
        return false;
    }
    if (p_end == token_end) {
        // No imaginary part in the string (e.g. "3.5")
        *p_imag = 0.0;
        return !unmatched_opening_paren;
    }
    if (*p_end == imaginary_unit) {
        /* Only an imaginary part (e.g "1.5j") */
        *p_imag = *p_real;
        *p_real = 0.0;
        ++p_end;
    }
    else if (*p_end == '+' || *p_end == '-') {
        /* Imaginary part still to parse */
        if (*p_end == '+') {
            ++p_end;  /* Advance to support +- (and ++) */
        }
        if (double_from_ucs4(p_end, token_end, false, p_imag, &p_end) < 0) {
            return false;
        }
        if (*p_end != imaginary_unit) {
            return false;
        }
        ++p_end;
    }
    else {
        *p_imag = 0;
    }

    if (unmatched_opening_paren) {
        /* Allow whitespace inside brackets as in "(1+2j )" or "( 1j )" */
        while (Py_UNICODE_ISSPACE(*p_end)) {
            ++p_end;
        }
        if (*p_end == ')') {
            ++p_end;
        }
        else {
            /* parentheses was not closed */
            return false;
        }
    }

    while (Py_UNICODE_ISSPACE(*p_end)) {
        ++p_end;
    }
    return p_end == token_end;
}


NPY_NO_EXPORT int
npy_to_cfloat(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig)
{
    double real;
    double imag;

    bool success = to_complex_int(
            str, end, &real, &imag,
            pconfig->imaginary_unit, true);

    if (!success) {
        return -1;
    }
    npy_complex64 val = {(float)real, (float)imag};
    memcpy(dataptr, &val, sizeof(npy_complex64));
    if (!PyArray_ISNBO(descr->byteorder)) {
        npy_bswap4_unaligned(dataptr);
        npy_bswap4_unaligned(dataptr + 4);
    }
    return 0;
}


NPY_NO_EXPORT int
npy_to_cdouble(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig)
{
    double real;
    double imag;

    bool success = to_complex_int(
            str, end, &real, &imag, pconfig->imaginary_unit, true);

    if (!success) {
        return -1;
    }
    npy_complex128 val = {real, imag};
    memcpy(dataptr, &val, sizeof(npy_complex128));
    if (!PyArray_ISNBO(descr->byteorder)) {
        npy_bswap8_unaligned(dataptr);
        npy_bswap8_unaligned(dataptr + 8);
    }
    return 0;
}


/*
 * String and unicode conversion functions.
 */
NPY_NO_EXPORT int
npy_to_string(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *NPY_UNUSED(unused))
{
    const Py_UCS4* c = str;
    size_t length = descr->elsize;

    for (size_t i = 0; i < length; i++) {
        if (c < end) {
            /*
             * loadtxt assumed latin1, which is compatible with UCS1 (first
             * 256 unicode characters).
             */
            if (NPY_UNLIKELY(*c > 255)) {
                /* TODO: Was UnicodeDecodeError, is unspecific error good? */
                return -1;
            }
            dataptr[i] = (Py_UCS1)(*c);
            c++;
        }
        else {
            dataptr[i] = '\0';
        }
    }
    return 0;
}


NPY_NO_EXPORT int
npy_to_unicode(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *NPY_UNUSED(unused))
{
    int length = descr->elsize / 4;

    if (length <= end - str) {
        memcpy(dataptr, str, length * 4);
    }
    else {
        size_t given_len = end - str;
        memcpy(dataptr, str, given_len * 4);
        memset(dataptr + given_len * 4, '\0', (length - given_len) * 4);
    }

    if (!PyArray_ISNBO(descr->byteorder)) {
        for (int i = 0; i < length; i++) {
            npy_bswap4_unaligned(dataptr);
            dataptr += 4;
        }
    }
    return 0;
}



/*
 * Convert functions helper for the generic converter.
 */
static PyObject *
call_converter_function(
        PyObject *func, const Py_UCS4 *str, size_t length, bool byte_converters)
{
    PyObject *s = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, str, length);
    if (s == NULL) {
        return s;
    }
    if (byte_converters) {
        Py_SETREF(s, PyUnicode_AsEncodedString(s, "latin1", NULL));
        if (s == NULL) {
            return NULL;
        }
    }
    if (func == NULL) {
        return s;
    }
    PyObject *result = PyObject_CallFunctionObjArgs(func, s, NULL);
    Py_DECREF(s);
    return result;
}


NPY_NO_EXPORT int
npy_to_generic_with_converter(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *config, PyObject *func)
{
    bool use_byte_converter;
    if (func == NULL) {
        use_byte_converter = config->c_byte_converters;
    }
    else {
        use_byte_converter = config->python_byte_converters;
    }
    /* Converts to unicode and calls custom converter (if set) */
    PyObject *converted = call_converter_function(
            func, str, (size_t)(end - str), use_byte_converter);
    if (converted == NULL) {
        return -1;
    }

    int res = PyArray_Pack(descr, dataptr, converted);
    Py_DECREF(converted);
    return res;
}


NPY_NO_EXPORT int
npy_to_generic(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *config)
{
    return npy_to_generic_with_converter(descr, str, end, dataptr, config, NULL);
}
