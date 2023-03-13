
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "lowlevel_strided_loops.h"

#include <string.h>
#include "textreading/str_to_int.h"
#include "textreading/parser_config.h"
#include "conversions.h"  /* For the deprecated parse-via-float path */


const char *deprecation_msg = (
        "loadtxt(): Parsing an integer via a float is deprecated.  To avoid "
        "this warning, you can:\n"
        "    * make sure the original data is stored as integers.\n"
        "    * use the `converters=` keyword argument.  If you only use\n"
        "      NumPy 1.23 or later, `converters=float` will normally work.\n"
        "    * Use `np.loadtxt(...).astype(np.int64)` parsing the file as\n"
        "      floating point and then convert it.  (On all NumPy versions.)\n"
        "  (Deprecated NumPy 1.23)");

#define DECLARE_TO_INT(intw, INT_MIN, INT_MAX, byteswap_unaligned)          \
    NPY_NO_EXPORT int                                                       \
    npy_to_##intw(PyArray_Descr *descr,                                     \
            const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,          \
            parser_config *pconfig)                                         \
    {                                                                       \
        int64_t parsed;                                                     \
        intw##_t x;                                                         \
                                                                            \
        if (NPY_UNLIKELY(                                                   \
                str_to_int64(str, end, INT_MIN, INT_MAX, &parsed) < 0)) {   \
            /* DEPRECATED 2022-07-03, NumPy 1.23 */                         \
            double fval;                                                    \
            PyArray_Descr *d_descr = PyArray_DescrFromType(NPY_DOUBLE);     \
            Py_DECREF(d_descr);  /* borrowed */                             \
            if (npy_to_double(d_descr, str, end, (char *)&fval, pconfig) < 0) { \
                return -1;                                                  \
            }                                                               \
            if (!pconfig->gave_int_via_float_warning) {                     \
                pconfig->gave_int_via_float_warning = true;                 \
                if (PyErr_WarnEx(PyExc_DeprecationWarning,                  \
                        deprecation_msg, 3) < 0) {                          \
                    return -1;                                              \
                }                                                           \
            }                                                               \
            pconfig->gave_int_via_float_warning = true;                     \
            x = (intw##_t)fval;                                             \
        }                                                                   \
        else {                                                              \
            x = (intw##_t)parsed;                                           \
        }                                                                   \
        memcpy(dataptr, &x, sizeof(x));                                     \
        if (!PyArray_ISNBO(descr->byteorder)) {                             \
            byteswap_unaligned(dataptr);                                    \
        }                                                                   \
        return 0;                                                           \
    }

#define DECLARE_TO_UINT(uintw, UINT_MAX, byteswap_unaligned)                \
    NPY_NO_EXPORT int                                                       \
    npy_to_##uintw(PyArray_Descr *descr,                                    \
            const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,          \
            parser_config *pconfig)                                         \
    {                                                                       \
        uint64_t parsed;                                                    \
        uintw##_t x;                                                        \
                                                                            \
        if (NPY_UNLIKELY(                                                   \
                str_to_uint64(str, end, UINT_MAX, &parsed) < 0)) {          \
            /* DEPRECATED 2022-07-03, NumPy 1.23 */                         \
            double fval;                                                    \
            PyArray_Descr *d_descr = PyArray_DescrFromType(NPY_DOUBLE);     \
            Py_DECREF(d_descr);  /* borrowed */                             \
            if (npy_to_double(d_descr, str, end, (char *)&fval, pconfig) < 0) { \
                return -1;                                                  \
            }                                                               \
            if (!pconfig->gave_int_via_float_warning) {                     \
                pconfig->gave_int_via_float_warning = true;                 \
                if (PyErr_WarnEx(PyExc_DeprecationWarning,                  \
                        deprecation_msg, 3) < 0) {                          \
                    return -1;                                              \
                }                                                           \
            }                                                               \
            pconfig->gave_int_via_float_warning = true;                     \
            x = (uintw##_t)fval;                                            \
        }                                                                   \
        else {                                                              \
            x = (uintw##_t)parsed;                                          \
        }                                                                   \
        memcpy(dataptr, &x, sizeof(x));                                     \
        if (!PyArray_ISNBO(descr->byteorder)) {                             \
            byteswap_unaligned(dataptr);                                    \
        }                                                                   \
        return 0;                                                           \
    }

#define byteswap_nothing(ptr)

DECLARE_TO_INT(int8, INT8_MIN, INT8_MAX, byteswap_nothing)
DECLARE_TO_INT(int16, INT16_MIN, INT16_MAX, npy_bswap2_unaligned)
DECLARE_TO_INT(int32, INT32_MIN, INT32_MAX, npy_bswap4_unaligned)
DECLARE_TO_INT(int64, INT64_MIN, INT64_MAX, npy_bswap8_unaligned)

DECLARE_TO_UINT(uint8, UINT8_MAX, byteswap_nothing)
DECLARE_TO_UINT(uint16, UINT16_MAX, npy_bswap2_unaligned)
DECLARE_TO_UINT(uint32, UINT32_MAX, npy_bswap4_unaligned)
DECLARE_TO_UINT(uint64, UINT64_MAX, npy_bswap8_unaligned)
