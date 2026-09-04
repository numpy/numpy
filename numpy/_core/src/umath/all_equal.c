/*
 * Internal gufuncs ``(n),(n)->()`` implementing a fused "all elements
 * equal" reduction with an early exit, used by np.array_equal:
 *
 *   _all_equal:     all(a == b) with the dtype's regular equality
 *                   (NaN compares unequal, -0.0 == 0.0, bool by truth value).
 *   _all_equal_nan: like _all_equal for inexact dtypes, but NaNs compare
 *                   equal when they appear in the same positions (for
 *                   complex, a value counts as NaN if either component is).
 *
 * Loops process contiguous data in blocks with branch-free accumulation
 * (so compilers can vectorize) and exit early after a failing block.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_math.h"
#include "numpy/halffloat.h"

#include "all_equal.h"

#define ALL_EQUAL_BLOCK 1024


#define OUTER_LOOP_HEAD \
    npy_intp N_ = dimensions[0]; \
    npy_intp n_ = dimensions[1]; \
    char *ap_ = args[0], *bp_ = args[1], *op_ = args[2]; \
    npy_intp as_ = steps[0], bs_ = steps[1], os_ = steps[2]; \
    npy_intp ais_ = steps[3], bis_ = steps[4]; \
    for (npy_intp i_ = 0; i_ < N_; i_++, ap_ += as_, bp_ += bs_, op_ += os_) { \
        char *a_ = ap_, *b_ = bp_; \
        npy_bool res_ = 1;

#define OUTER_LOOP_TAIL \
        *(npy_bool *)op_ = res_; \
    }

/* Branch-free blocked comparison of contiguous data; PRED uses ta/tb/k_. */
#define BLOCKED_CONTIG(type, count, PRED) \
    do { \
        const type *ta = (const type *)a_; \
        const type *tb = (const type *)b_; \
        npy_intp k_ = 0; \
        while (k_ < (count) && res_) { \
            npy_intp end_ = k_ + ALL_EQUAL_BLOCK; \
            if (end_ > (count)) { \
                end_ = (count); \
            } \
            npy_bool ok_ = 1; \
            for (; k_ < end_; k_++) { \
                ok_ &= (npy_bool)(PRED); \
            } \
            res_ = ok_; \
        } \
    } while (0)

/* Integer loops: memcmp for unit strides, scalar early exit otherwise. */
#define INT_ALL_EQUAL(TYPE, type) \
static void \
TYPE##_all_equal(char **args, npy_intp const *dimensions, \
                 npy_intp const *steps, void *NPY_UNUSED(func)) \
{ \
    OUTER_LOOP_HEAD \
    if (ais_ == sizeof(type) && bis_ == sizeof(type)) { \
        res_ = memcmp(a_, b_, n_ * sizeof(type)) == 0; \
    } \
    else { \
        for (npy_intp k_ = 0; k_ < n_; k_++, a_ += ais_, b_ += bis_) { \
            if (*(type *)a_ != *(type *)b_) { \
                res_ = 0; \
                break; \
            } \
        } \
    } \
    OUTER_LOOP_TAIL \
}

INT_ALL_EQUAL(BYTE, npy_byte)
INT_ALL_EQUAL(UBYTE, npy_ubyte)
INT_ALL_EQUAL(SHORT, npy_short)
INT_ALL_EQUAL(USHORT, npy_ushort)
INT_ALL_EQUAL(INT, npy_int)
INT_ALL_EQUAL(UINT, npy_uint)
INT_ALL_EQUAL(LONG, npy_long)
INT_ALL_EQUAL(ULONG, npy_ulong)
INT_ALL_EQUAL(LONGLONG, npy_longlong)
INT_ALL_EQUAL(ULONGLONG, npy_ulonglong)

/* Bools compare by truth value; identical bytes always means equal. */
static NPY_GCC_OPT_3 void
BOOL_all_equal(char **args, npy_intp const *dimensions,
               npy_intp const *steps, void *NPY_UNUSED(func))
{
    OUTER_LOOP_HEAD
    if (ais_ == 1 && bis_ == 1) {
        if (memcmp(a_, b_, n_) != 0) {
            BLOCKED_CONTIG(npy_bool, n_, (ta[k_] != 0) == (tb[k_] != 0));
        }
    }
    else {
        for (npy_intp k_ = 0; k_ < n_; k_++, a_ += ais_, b_ += bis_) {
            if ((*(npy_bool *)a_ != 0) != (*(npy_bool *)b_ != 0)) {
                res_ = 0;
                break;
            }
        }
    }
    OUTER_LOOP_TAIL
}

/*
 * Float loops: regular equality (NaN unequal, -0.0 == 0.0).  A contiguous
 * complex array compares equal iff its 2*n interleaved components do.
 */
#define FLOAT_ALL_EQUAL(TYPE, type, NCOMP) \
static NPY_GCC_OPT_3 void \
TYPE##_all_equal(char **args, npy_intp const *dimensions, \
                 npy_intp const *steps, void *NPY_UNUSED(func)) \
{ \
    OUTER_LOOP_HEAD \
    if (ais_ == (NCOMP) * sizeof(type) && bis_ == (NCOMP) * sizeof(type)) { \
        BLOCKED_CONTIG(type, (NCOMP) * n_, ta[k_] == tb[k_]); \
    } \
    else { \
        for (npy_intp k_ = 0; k_ < n_; k_++, a_ += ais_, b_ += bis_) { \
            const type *ca_ = (const type *)a_; \
            const type *cb_ = (const type *)b_; \
            npy_bool eq_ = 1; \
            for (int c_ = 0; c_ < (NCOMP); c_++) { \
                eq_ &= (npy_bool)(ca_[c_] == cb_[c_]); \
            } \
            if (!eq_) { \
                res_ = 0; \
                break; \
            } \
        } \
    } \
    OUTER_LOOP_TAIL \
}

FLOAT_ALL_EQUAL(FLOAT, npy_float, 1)
FLOAT_ALL_EQUAL(DOUBLE, npy_double, 1)
FLOAT_ALL_EQUAL(LONGDOUBLE, npy_longdouble, 1)
FLOAT_ALL_EQUAL(CFLOAT, npy_float, 2)
FLOAT_ALL_EQUAL(CDOUBLE, npy_double, 2)
FLOAT_ALL_EQUAL(CLONGDOUBLE, npy_longdouble, 2)

static void
HALF_all_equal(char **args, npy_intp const *dimensions,
               npy_intp const *steps, void *NPY_UNUSED(func))
{
    OUTER_LOOP_HEAD
    for (npy_intp k_ = 0; k_ < n_; k_++, a_ += ais_, b_ += bis_) {
        if (!npy_half_eq(*(npy_half *)a_, *(npy_half *)b_)) {
            res_ = 0;
            break;
        }
    }
    OUTER_LOOP_TAIL
}

/* equal_nan variants: NaNs must appear in the same positions. */
#define FLOAT_ALL_EQUAL_NAN(TYPE, type) \
static NPY_GCC_OPT_3 void \
TYPE##_all_equal_nan(char **args, npy_intp const *dimensions, \
                     npy_intp const *steps, void *NPY_UNUSED(func)) \
{ \
    OUTER_LOOP_HEAD \
    if (ais_ == sizeof(type) && bis_ == sizeof(type)) { \
        BLOCKED_CONTIG(type, n_, \
            ((ta[k_] != ta[k_]) == (tb[k_] != tb[k_])) \
            & ((ta[k_] != ta[k_]) | (ta[k_] == tb[k_]))); \
    } \
    else { \
        for (npy_intp k_ = 0; k_ < n_; k_++, a_ += ais_, b_ += bis_) { \
            type av_ = *(type *)a_, bv_ = *(type *)b_; \
            int an_ = av_ != av_, bn_ = bv_ != bv_; \
            if (an_ != bn_ || (!an_ && av_ != bv_)) { \
                res_ = 0; \
                break; \
            } \
        } \
    } \
    OUTER_LOOP_TAIL \
}

FLOAT_ALL_EQUAL_NAN(FLOAT, npy_float)
FLOAT_ALL_EQUAL_NAN(DOUBLE, npy_double)
FLOAT_ALL_EQUAL_NAN(LONGDOUBLE, npy_longdouble)

static void
HALF_all_equal_nan(char **args, npy_intp const *dimensions,
                   npy_intp const *steps, void *NPY_UNUSED(func))
{
    OUTER_LOOP_HEAD
    for (npy_intp k_ = 0; k_ < n_; k_++, a_ += ais_, b_ += bis_) {
        npy_half av_ = *(npy_half *)a_, bv_ = *(npy_half *)b_;
        int an_ = npy_half_isnan(av_), bn_ = npy_half_isnan(bv_);
        if (an_ != bn_ || (!an_ && !npy_half_eq_nonan(av_, bv_))) {
            res_ = 0;
            break;
        }
    }
    OUTER_LOOP_TAIL
}

/* A complex value counts as NaN if either component is NaN. */
#define CFLOAT_ALL_EQUAL_NAN(TYPE, type) \
static void \
TYPE##_all_equal_nan(char **args, npy_intp const *dimensions, \
                     npy_intp const *steps, void *NPY_UNUSED(func)) \
{ \
    OUTER_LOOP_HEAD \
    for (npy_intp k_ = 0; k_ < n_; k_++, a_ += ais_, b_ += bis_) { \
        const type *ca_ = (const type *)a_; \
        const type *cb_ = (const type *)b_; \
        int an_ = ca_[0] != ca_[0] || ca_[1] != ca_[1]; \
        int bn_ = cb_[0] != cb_[0] || cb_[1] != cb_[1]; \
        if (an_ != bn_ \
                || (!an_ && (ca_[0] != cb_[0] || ca_[1] != cb_[1]))) { \
            res_ = 0; \
            break; \
        } \
    } \
    OUTER_LOOP_TAIL \
}

CFLOAT_ALL_EQUAL_NAN(CFLOAT, npy_float)
CFLOAT_ALL_EQUAL_NAN(CDOUBLE, npy_double)
CFLOAT_ALL_EQUAL_NAN(CLONGDOUBLE, npy_longdouble)


static PyUFuncGenericFunction all_equal_functions[] = {
    BOOL_all_equal,
    BYTE_all_equal, UBYTE_all_equal,
    SHORT_all_equal, USHORT_all_equal,
    INT_all_equal, UINT_all_equal,
    LONG_all_equal, ULONG_all_equal,
    LONGLONG_all_equal, ULONGLONG_all_equal,
    HALF_all_equal,
    FLOAT_all_equal, DOUBLE_all_equal, LONGDOUBLE_all_equal,
    CFLOAT_all_equal, CDOUBLE_all_equal, CLONGDOUBLE_all_equal,
};

static const char all_equal_types[] = {
    NPY_BOOL, NPY_BOOL, NPY_BOOL,
    NPY_BYTE, NPY_BYTE, NPY_BOOL,
    NPY_UBYTE, NPY_UBYTE, NPY_BOOL,
    NPY_SHORT, NPY_SHORT, NPY_BOOL,
    NPY_USHORT, NPY_USHORT, NPY_BOOL,
    NPY_INT, NPY_INT, NPY_BOOL,
    NPY_UINT, NPY_UINT, NPY_BOOL,
    NPY_LONG, NPY_LONG, NPY_BOOL,
    NPY_ULONG, NPY_ULONG, NPY_BOOL,
    NPY_LONGLONG, NPY_LONGLONG, NPY_BOOL,
    NPY_ULONGLONG, NPY_ULONGLONG, NPY_BOOL,
    NPY_HALF, NPY_HALF, NPY_BOOL,
    NPY_FLOAT, NPY_FLOAT, NPY_BOOL,
    NPY_DOUBLE, NPY_DOUBLE, NPY_BOOL,
    NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_BOOL,
    NPY_CFLOAT, NPY_CFLOAT, NPY_BOOL,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_BOOL,
    NPY_CLONGDOUBLE, NPY_CLONGDOUBLE, NPY_BOOL,
};

static void *all_equal_data[18];

static PyUFuncGenericFunction all_equal_nan_functions[] = {
    HALF_all_equal_nan,
    FLOAT_all_equal_nan, DOUBLE_all_equal_nan, LONGDOUBLE_all_equal_nan,
    CFLOAT_all_equal_nan, CDOUBLE_all_equal_nan, CLONGDOUBLE_all_equal_nan,
};

static const char all_equal_nan_types[] = {
    NPY_HALF, NPY_HALF, NPY_BOOL,
    NPY_FLOAT, NPY_FLOAT, NPY_BOOL,
    NPY_DOUBLE, NPY_DOUBLE, NPY_BOOL,
    NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_BOOL,
    NPY_CFLOAT, NPY_CFLOAT, NPY_BOOL,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_BOOL,
    NPY_CLONGDOUBLE, NPY_CLONGDOUBLE, NPY_BOOL,
};

static void *all_equal_nan_data[7];


NPY_NO_EXPORT int
init_all_equal(PyObject *d)
{
    PyObject *f = PyUFunc_FromFuncAndDataAndSignature(
            all_equal_functions, all_equal_data, (char *)all_equal_types, 18,
            2, 1, PyUFunc_None, "_all_equal",
            "all(a == b) over the core dimension, with an early exit",
            0, "(n),(n)->()");
    if (f == NULL) {
        return -1;
    }
    int res = PyDict_SetItemString(d, "_all_equal", f);
    Py_DECREF(f);
    if (res < 0) {
        return -1;
    }

    f = PyUFunc_FromFuncAndDataAndSignature(
            all_equal_nan_functions, all_equal_nan_data,
            (char *)all_equal_nan_types, 7,
            2, 1, PyUFunc_None, "_all_equal_nan",
            "like _all_equal, but NaNs in matching positions compare equal",
            0, "(n),(n)->()");
    if (f == NULL) {
        return -1;
    }
    res = PyDict_SetItemString(d, "_all_equal_nan", f);
    Py_DECREF(f);
    if (res < 0) {
        return -1;
    }
    return 0;
}
