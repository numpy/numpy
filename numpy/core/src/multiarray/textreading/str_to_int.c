
#include <Python.h>

#include <string.h>
#include "textreading/str_to_int.h"
#include "textreading/conversions.h"
#include "textreading/parser_config.h"


NPY_NO_EXPORT PyArray_Descr *double_descr = NULL;


#define DECLARE_TO_INT(intw, INT_MIN, INT_MAX)                              \
    int                                                                     \
    to_##intw(PyArray_Descr *descr,                                         \
            const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,          \
            parser_config *pconfig)                                         \
    {                                                                       \
        int64_t parsed;                                                     \
        intw##_t x;                                                         \
                                                                            \
        if (str_to_int64(str, end, INT_MIN, INT_MAX, &parsed) < 0) {        \
            return -1;                                                      \
        }                                                                   \
        else {                                                              \
            x = (intw##_t)parsed;                                           \
        }                                                                   \
        memcpy(dataptr, &x, sizeof(x));                                     \
        if (!PyArray_ISNBO(descr->byteorder)) {                             \
            descr->f->copyswap(dataptr, dataptr, 1, NULL);                  \
        }                                                                   \
        return 0;                                                           \
    }

#define DECLARE_TO_UINT(uintw, UINT_MAX)                                    \
    int                                                                     \
    to_##uintw(PyArray_Descr *descr,                                        \
            const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,          \
            parser_config *pconfig)                                         \
    {                                                                       \
        uint64_t parsed;                                                    \
        uintw##_t x;                                                        \
                                                                            \
        if (str_to_uint64(str, end, UINT_MAX, &parsed) < 0) {               \
            return -1;                                                      \
        }                                                                   \
        else {                                                              \
            x = (uintw##_t)parsed;                                          \
        }                                                                   \
        memcpy(dataptr, &x, sizeof(x));                                     \
        if (!PyArray_ISNBO(descr->byteorder)) {                             \
            descr->f->copyswap(dataptr, dataptr, 1, NULL);                  \
        }                                                                   \
        return 0;                                                           \
    }

DECLARE_TO_INT(int8, INT8_MIN, INT8_MAX)
DECLARE_TO_INT(int16, INT16_MIN, INT16_MAX)
DECLARE_TO_INT(int32, INT32_MIN, INT32_MAX)
DECLARE_TO_INT(int64, INT64_MIN, INT64_MAX)

DECLARE_TO_UINT(uint8, UINT8_MAX)
DECLARE_TO_UINT(uint16, UINT16_MAX)
DECLARE_TO_UINT(uint32, UINT32_MAX)
DECLARE_TO_UINT(uint64, UINT64_MAX)
