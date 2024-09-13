#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STR_TO_INT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STR_TO_INT_H_

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"

#include "textreading/parser_config.h"


/*
 * The following two string conversion functions are largely equivalent
 * in Pandas.  They are in the header file here, to ensure they can be easily
 * inline in the other function.
 * Unlike pandas, pass in end-pointer (do not rely on \0) and return 0 or -1.
 *
 * The actual functions are defined using macro templating below.
 */
NPY_FINLINE int
str_to_int64(
        const Py_UCS4 *p_item, const Py_UCS4 *p_end,
        int64_t int_min, int64_t int_max, int64_t *result)
{
    const Py_UCS4 *p = (const Py_UCS4 *)p_item;
    bool isneg = 0;
    int64_t number = 0;

    // Skip leading spaces.
    while (Py_UNICODE_ISSPACE(*p)) {
        ++p;
    }

    // Handle sign.
    if (*p == '-') {
        isneg = true;
        ++p;
    }
    else if (*p == '+') {
        p++;
    }

    // Check that there is a first digit.
    if (!isdigit(*p)) {
        return -1;
    }

    if (isneg) {
        // If number is greater than pre_min, at least one more digit
        // can be processed without overflowing.
        int dig_pre_min = -(int_min % 10);
        int64_t pre_min = int_min / 10;

        // Process the digits.
        int d = *p;
        while (isdigit(d)) {
            if ((number > pre_min) || ((number == pre_min) && (d - '0' <= dig_pre_min))) {
                number = number * 10 - (d - '0');
                d = *++p;
            }
            else {
                return -1;
            }
        }
    }
    else {
        // If number is less than pre_max, at least one more digit
        // can be processed without overflowing.
        int64_t pre_max = int_max / 10;
        int dig_pre_max = int_max % 10;

        // Process the digits.
        int d = *p;
        while (isdigit(d)) {
            if ((number < pre_max) || ((number == pre_max) && (d - '0' <= dig_pre_max))) {
                number = number * 10 + (d - '0');
                d = *++p;
            }
            else {
                return -1;
            }
        }
    }

    // Skip trailing spaces.
    while (Py_UNICODE_ISSPACE(*p)) {
        ++p;
    }

    // Did we use up all the characters?
    if (p != p_end) {
        return -1;
    }

    *result = number;
    return 0;
}


NPY_FINLINE int
str_to_uint64(
        const Py_UCS4 *p_item, const Py_UCS4 *p_end,
        uint64_t uint_max, uint64_t *result)
{
    const Py_UCS4 *p = (const Py_UCS4 *)p_item;
    uint64_t number = 0;
    int d;

    // Skip leading spaces.
    while (Py_UNICODE_ISSPACE(*p)) {
        ++p;
    }

    // Handle sign.
    if (*p == '-') {
        return -1;
    }
    if (*p == '+') {
        p++;
    }

    // Check that there is a first digit.
    if (!isdigit(*p)) {
        return -1;
    }

    // If number is less than pre_max, at least one more digit
    // can be processed without overflowing.
    uint64_t pre_max = uint_max / 10;
    int dig_pre_max = uint_max % 10;

    // Process the digits.
    d = *p;
    while (isdigit(d)) {
        if ((number < pre_max) || ((number == pre_max) && (d - '0' <= dig_pre_max))) {
            number = number * 10 + (d - '0');
            d = *++p;
        }
        else {
            return -1;
        }
    }

    // Skip trailing spaces.
    while (Py_UNICODE_ISSPACE(*p)) {
        ++p;
    }

    // Did we use up all the characters?
    if (p != p_end) {
        return -1;
    }

    *result = number;
    return 0;
}


#define DECLARE_TO_INT_PROTOTYPE(intw)                                  \
    NPY_NO_EXPORT int                                                   \
    npy_to_##intw(PyArray_Descr *descr,                                     \
            const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,      \
            parser_config *pconfig);

DECLARE_TO_INT_PROTOTYPE(int8)
DECLARE_TO_INT_PROTOTYPE(int16)
DECLARE_TO_INT_PROTOTYPE(int32)
DECLARE_TO_INT_PROTOTYPE(int64)

DECLARE_TO_INT_PROTOTYPE(uint8)
DECLARE_TO_INT_PROTOTYPE(uint16)
DECLARE_TO_INT_PROTOTYPE(uint32)
DECLARE_TO_INT_PROTOTYPE(uint64)

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STR_TO_INT_H_ */
