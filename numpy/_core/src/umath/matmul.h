#ifndef _NPY_CORE_SRC_UMATH_MATMUL_H_
#define _NPY_CORE_SRC_UMATH_MATMUL_H_

#ifdef __cplusplus

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_config.h"
#include "numpy/npy_common.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_math.h"
#include "numpy/halffloat.h"
#include "lowlevel_strided_loops.h"

#include "blas_utils.h"
#include "npy_cblas.h"
#include "arraytypes.h"

#include <assert.h>
#include <complex>

template<typename T>
void
_gemv(T *ip1, npy_intp is1_m, npy_intp is1_n,
    T *ip2, npy_intp is2_n,
    T *op, npy_intp op_m,
    npy_intp m, npy_intp n)
{
    // TODO
}


template<typename T>
void
_dot(T *ip1, npy_intp is1, T *ip2, npy_intp is2, T *op,
        npy_intp n, void *NPY_UNUSED(ignore))
{

}

/***********************************************************************************
** Defining dot functions
***********************************************************************************/
#define DEFINE_DOT_FUNCTION(NAME, TYPE)                                    \
template<>                                                         \
void _dot<TYPE>(TYPE *ip1, npy_intp is1, TYPE *ip2,               \
                npy_intp is2, TYPE *op, npy_intp n,                \
                void *NPY_UNUSED(ignore))                          \
{                                                                  \
    NAME##_dot((char *)ip1, is1, (char *)ip2, is2,                 \
               (char *)op, n, nullptr);                            \
}

DEFINE_DOT_FUNCTION(FLOAT, npy_float)
DEFINE_DOT_FUNCTION(DOUBLE, npy_double)
DEFINE_DOT_FUNCTION(CFLOAT, std::complex<float>) // TODO: need to fix handling complex
DEFINE_DOT_FUNCTION(CDOUBLE, std::complex<double>)
DEFINE_DOT_FUNCTION(BOOL, npy_bool)
DEFINE_DOT_FUNCTION(BYTE, npy_byte)
// DEFINE_DOT_FUNCTION(UBYTE, npy_ubyte)
DEFINE_DOT_FUNCTION(SHORT, npy_short)
DEFINE_DOT_FUNCTION(USHORT, npy_ushort)
DEFINE_DOT_FUNCTION(INT, npy_int)
DEFINE_DOT_FUNCTION(UINT, npy_uint)
DEFINE_DOT_FUNCTION(LONG, npy_long)
DEFINE_DOT_FUNCTION(ULONG, npy_ulong)
DEFINE_DOT_FUNCTION(LONGLONG, npy_longlong)
DEFINE_DOT_FUNCTION(ULONGLONG, npy_ulonglong)
DEFINE_DOT_FUNCTION(LONGDOUBLE, npy_longdouble)
// DEFINE_DOT_FUNCTION(TIMEDELTA, npy_timedelta)
// DEFINE_DOT_FUNCTION(HALF, npy_half) // TODO: fix handling of this and some other types
DEFINE_DOT_FUNCTION(CLONGDOUBLE, npy_clongdouble)
DEFINE_DOT_FUNCTION(OBJECT, PyObject*)

#undef DEFINE_DOT_FUNCTION


template <typename T>
void
_matmul_inner_noblas(T *ip1, npy_intp is1_m, npy_intp is1_n,
                        T *ip2, npy_intp is2_n, npy_intp is2_p,
                        T *op, npy_intp os_m, npy_intp os_p,
                        npy_intp dm, npy_intp dn, npy_intp dp)
{
    npy_intp m, n, p;
    npy_intp ib1_n, ib2_n, ib2_p, ob_p;

    ib1_n = is1_n * dn;
    ib2_n = is2_n * dn;
    ib2_p = is2_p * dp;
    ob_p = os_p * dp;

    for(m = 0; m < dm; m++) {
        for(p = 0; p < dp; p++) {
            *op = 0;
            for(n = 0; n < dn; n++) {
                T val1 = *ip1;
                T val2 = *ip2;
                *op += val1 * val2;

                ip2 += is2_n;
                ip1 += is1_n;
            }
            // TODO: Need to fix npy_half

            ip1 -= ib1_n;
            ip2 -= ib2_n;
            op += os_p;
            ip2 += is2_p;
        }
        op -= ob_p;
        ip2 -= ib2_p;
        ip1 += is1_m;
        op += os_m;
    }
}

template<>
void
_matmul_inner_noblas<npy_bool>(npy_bool *ip1, npy_intp is1_m, npy_intp is1_n,
                        npy_bool *ip2, npy_intp is2_n, npy_intp is2_p,
                        npy_bool *op, npy_intp os_m, npy_intp os_p,
                        npy_intp dm, npy_intp dn, npy_intp dp)
{

}

template<>
void
_matmul_inner_noblas<PyObject*>(PyObject **ip1, npy_intp is1_m, npy_intp is1_n,
                        PyObject **ip2, npy_intp is2_n, npy_intp is2_p,
                        PyObject **op, npy_intp os_m, npy_intp os_p,
                        npy_intp dm, npy_intp dn, npy_intp dp)
{

}

template <typename T>
NPY_NO_EXPORT void
matmul(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    npy_intp d_outer = *dimensions++;
    npy_intp i_outer;
    npy_intp s0 = steps[0]; // Steps for batches
    npy_intp s1 = steps[1];
    npy_intp s2 = steps[2];
    npy_intp dm = dimensions[0];
    npy_intp dn = dimensions[1];
    npy_intp dp = dimensions[2];

    npy_intp is1_m = steps[3]; // (m, n) @ (n, p) -> (m, p)
    npy_intp is1_n = steps[4];
    npy_intp is2_n = steps[5];
    npy_intp is2_p = steps[6];
    npy_intp os_m = steps[7];
    npy_intp os_p = steps[8];
#if defined(HAVE_CBLAS)
    if constexpr(std::is_same_v<T, npy_float> || std::is_same_v<T, npy_double> ||
                std::is_same_v<T, std::complex<npy_float>> || std::is_same_v<T, std::complex<npy_double>>) {
        // TODO
    }
#endif

    for(i_outer = 0; i_outer < d_outer; i_outer++,
                                args[0] += s0, args[1] += s1, args[2] += s2) {
        T *ip1 = reinterpret_cast<T *>(args[0]);
        T *ip2 = reinterpret_cast<T *>(args[1]);
        T *op = reinterpret_cast<T *>(args[2]);
#if defined(HAVE_CBLAS)
        if constexpr(std::is_same_v<T, npy_float> || std::is_same_v<T, npy_double> ||
                std::is_same_v<T, std::complex<npy_float>> || std::is_same_v<T, std::complex<npy_double>>) {
            // TODO
        }
#else
        _matmul_inner_noblas<T>(ip1, is1_m, is1_n,
                            ip2, is2_n, is2_p,
                            op, os_m, os_p, dm, dn, dp);
#endif
    }
#if defined(HAVE_CBLAS)
    if constexpr(std::is_same_v<T, npy_float> || std::is_same_v<T, npy_double> ||
                std::is_same_v<T, std::complex<npy_float>> || std::is_same_v<T, std::complex<npy_double>>) {
#if NPY_BLAS_CHECK_FPE_SUPPORT
        if(!npy_blas_supports_fpe()) {
            npy_clear_npy_floatstatus_barrier((char*) args);
        }
#endif
        // if (allocate_buffer) free(tmp_ip12op);
    }
#endif
}

template<typename T>
NPY_NO_EXPORT void
vecdot(char **args, npy_intp const * dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    npy_intp n_outer = dimensions[0];
    npy_intp s0 = steps[0];
    npy_intp s1 = steps[1];
    npy_intp s2 = steps[2];
    npy_intp n_inner = dimensions[1];
    npy_intp is1 = steps[3];
    npy_intp is2 = steps[4];

    for(npy_intp i = 0; i < n_outer; i++, args[0] += s0, args[1] += s1, args[2] += s2) {
        /*
         * TODO: use new API to select inner loop with get_loop and
         * improve error treatment.
         */
        T *ip1 = reinterpret_cast<T *>(args[0]);
        T *ip2 = reinterpret_cast<T *>(args[1]);
        T *op = reinterpret_cast<T *>(args[2]);
        _dot<T>(ip1, is1, ip2, is2, op, n_inner, nullptr);

        if constexpr(std::is_same_v<T, PyObject*>) {
            if(PyErr_Occurred()) {
                return;
            }
        }
    }
#if NPY_BLAS_CHECK_FPE_SUPPORT
    if constexpr(std::is_same_v<T, npy_float> || std::is_same_v<T, npy_double> ||
            std::is_same_v<T, std::complex<npy_float>> || std::is_same_v<T, std::complex<npy_double>>) {
        if(!npy_blas_supports_fpe()) {
            npy_clear_floatstatus_barrier((char*) args);
        }
    }
#endif
}

template<typename T>
void _vecmat_via_gemm(T *ip1, npy_intp is1_n,
                    T *ip2, npy_intp is2_n, npy_intp is2_m,
                    T *op, npy_intp os_m,
                    npy_intp n, npy_intp m)
{
    // TODO
}


template<typename T>
NPY_NO_EXPORT void
matvec(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    npy_intp n_outer = dimensions[0];
    npy_intp s0 = steps[0];
    npy_intp s1 = steps[1];
    npy_intp s2 = steps[2];
    npy_intp dm = dimensions[1];
    npy_intp dn = dimensions[2];

    npy_intp is1_m = steps[3]; // (m, n) @ (n) -> (m)
    npy_intp is1_n = steps[4];
    npy_intp is2_n = steps[5];
    npy_intp os_m = steps[6];

#if defined(HAVE_CBLAS)
    if constexpr(std::is_same_v<T, npy_float> || std::is_same_v<T, npy_double> ||
            std::is_same_v<T, std::complex<npy_float>> || std::is_same_v<T, std::complex<npy_double>>) {
        // TODO
    }
#endif
    for(npy_intp i = 0; i < n_outer; i++,
            args[0] += s0, args[1] += s1, args[2] += s2) {
        T *ip1 = reinterpret_cast<T *>(args[0]);
        T *ip2 = reinterpret_cast<T *>(args[1]);
        T *op = reinterpret_cast<T *>(args[2]);
#if defined(HAVE_CBLAS)
        if constexpr(std::is_same_v<T, npy_float> || std::is_same_v<T, npy_double> ||
            std::is_same_v<T, std::complex<npy_float>> || std::is_same_v<T, std::complex<npy_double>>) {
            // TODO
        }
#endif
        for(npy_intp j = 0; j < dm; j++, ip1 += is1_m, op += os_m) {
            _dot<T>(ip1, is1_n, ip2, is2_n, op, dn, nullptr);

            if constexpr(std::is_same_v<T, PyObject*>) {
                if(PyErr_Occurred()) {
                    return;
                }
            }
        }
    }

#if NPY_BLAS_CHECK_FPE_SUPPORT
    if constexpr(std::is_same_v<T, npy_float> || std::is_same_v<T, npy_double> ||
        std::is_same_v<T, std::complex<npy_float>> || std::is_same_v<T, std::complex<npy_double>>) {
        if(!npy_blas_supports_fpe()) {
            npy_clear_floatstatus_barrier((char*) args);
        }
    }
#endif
}

template<typename T>
NPY_NO_EXPORT void
vecmat(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    npy_intp n_outer = dimensions[0];
    npy_intp s0 = steps[0];
    npy_intp s1 = steps[1];
    npy_intp s2 = steps[2];
    npy_intp dn = dimensions[1];
    npy_intp dm = dimensions[2];

    npy_intp is1_n = steps[3]; // (n) * (n, m) -> (m)
    npy_intp is2_n = steps[4];
    npy_intp is2_m = steps[5];
    npy_intp os_m = steps[6];

#if defined(HAVE_CBLAS)
    if constexpr(std::is_same_v<T, npy_float> || std::is_same_v<T, npy_double> ||
            std::is_same_v<T, std::complex<npy_float>> || std::is_same_v<T, std::complex<npy_double>>) {
        // TODO
    }
#endif

    for(npy_intp i = 0; i < n_outer; i++, args[0] += s0, args[1] += s1, args[2] += s2) {
        T *ip1 = reinterpret_cast<T *>(args[0]);
        T *ip2 = reinterpret_cast<T *>(args[1]);
        T *op = reinterpret_cast<T *>(args[2]);
#if defined(HAVE_CBLAS)
       if constexpr(std::is_same_v<T, npy_float> || std::is_same_v<T, npy_double> ||
            std::is_same_v<T, std::complex<npy_float>> || std::is_same_v<T, std::complex<npy_double>>) {
            // if(blasable) {
            //     if constexpr(std::is_same_v<T, std::complex<npy_float>> || std::is_same_v<T, std::complex<npy_double>>) {
            //         _vecmat_via_gemm<T>(ip1, is1_n, ip2, is2_n, is2_m, op, os_m, dn, dm);
            //     } else {
            //         _gemv<T>(ip2, is2_m, is2_n, ip1, is1_n, op, os_m, dm, dn);
            //     }
            //     continue;
            // }
        }
#endif
        for(npy_intp j = 0; j < dm; j++, ip2 += is2_m, op += os_m) {
            _dot<T>(ip1, is1_n, ip2, is2_n, op, dn, nullptr);
            if constexpr(std::is_same_v<T, PyObject*>) {
                if(PyErr_Occurred()) {
                    return;
                }
            }
        }
    }

#if NPY_BLAS_CHECK_FPE_SUPPORT
    if constexpr(std::is_same_v<T, npy_float> || std::is_same_v<T, npy_double> ||
            std::is_same_v<T, std::complex<npy_float>> || std::is_same_v<T, std::complex<npy_double>>) {
        if(!npy_blas_supports_fpe()) {
            npy_clear_floatstatus_barrier((char*) args);
        }
    }
#endif

}

#endif // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

/***********************************************************************************
** Declaring matmul functions
***********************************************************************************/
#define DECLARE_MATMUL_FUNCTION(TYPE)                        \
NPY_NO_EXPORT void                                           \
TYPE##_matmul(char **args,                                   \
              npy_intp const *dimensions,                    \
              npy_intp const *steps,                         \
              void *NPY_UNUSED(func));

DECLARE_MATMUL_FUNCTION(FLOAT)
DECLARE_MATMUL_FUNCTION(DOUBLE)
DECLARE_MATMUL_FUNCTION(LONGDOUBLE)
DECLARE_MATMUL_FUNCTION(HALF)
DECLARE_MATMUL_FUNCTION(CFLOAT)
DECLARE_MATMUL_FUNCTION(CDOUBLE)
DECLARE_MATMUL_FUNCTION(CLONGDOUBLE)
DECLARE_MATMUL_FUNCTION(UBYTE)
DECLARE_MATMUL_FUNCTION(USHORT)
DECLARE_MATMUL_FUNCTION(UINT)
DECLARE_MATMUL_FUNCTION(ULONG)
DECLARE_MATMUL_FUNCTION(ULONGLONG)
DECLARE_MATMUL_FUNCTION(BYTE)
DECLARE_MATMUL_FUNCTION(SHORT)
DECLARE_MATMUL_FUNCTION(INT)
DECLARE_MATMUL_FUNCTION(LONG)
DECLARE_MATMUL_FUNCTION(LONGLONG)
DECLARE_MATMUL_FUNCTION(BOOL)
DECLARE_MATMUL_FUNCTION(OBJECT)

#undef DECLARE_MATMUL_FUNCTION

/***********************************************************************************
** Declaring vecdot functions
***********************************************************************************/
#define DECLARE_VECDOT_FUNCTION(TYPE)                        \
NPY_NO_EXPORT void                                           \
TYPE##_vecdot(char **args,                                   \
              npy_intp const *dimensions,                    \
              npy_intp const *steps,                         \
              void *NPY_UNUSED(func));

DECLARE_VECDOT_FUNCTION(FLOAT)
DECLARE_VECDOT_FUNCTION(DOUBLE)
DECLARE_VECDOT_FUNCTION(LONGDOUBLE)
DECLARE_VECDOT_FUNCTION(HALF)
DECLARE_VECDOT_FUNCTION(CFLOAT)
DECLARE_VECDOT_FUNCTION(CDOUBLE)
DECLARE_VECDOT_FUNCTION(CLONGDOUBLE)
DECLARE_VECDOT_FUNCTION(UBYTE)
DECLARE_VECDOT_FUNCTION(USHORT)
DECLARE_VECDOT_FUNCTION(UINT)
DECLARE_VECDOT_FUNCTION(ULONG)
DECLARE_VECDOT_FUNCTION(ULONGLONG)
DECLARE_VECDOT_FUNCTION(BYTE)
DECLARE_VECDOT_FUNCTION(SHORT)
DECLARE_VECDOT_FUNCTION(INT)
DECLARE_VECDOT_FUNCTION(LONG)
DECLARE_VECDOT_FUNCTION(LONGLONG)
DECLARE_VECDOT_FUNCTION(BOOL)
DECLARE_VECDOT_FUNCTION(OBJECT)

#undef DECLARE_VECDOT_FUNCTION

/***********************************************************************************
** Declaring matvec functions
***********************************************************************************/
#define DECLARE_MATVEC_FUNCTION(TYPE)                        \
NPY_NO_EXPORT void                                           \
TYPE##_matvec(char **args,                                   \
              npy_intp const *dimensions,                    \
              npy_intp const *steps,                         \
              void *NPY_UNUSED(func));

DECLARE_MATVEC_FUNCTION(FLOAT)
DECLARE_MATVEC_FUNCTION(DOUBLE)
DECLARE_MATVEC_FUNCTION(LONGDOUBLE)
DECLARE_MATVEC_FUNCTION(HALF)
DECLARE_MATVEC_FUNCTION(CFLOAT)
DECLARE_MATVEC_FUNCTION(CDOUBLE)
DECLARE_MATVEC_FUNCTION(CLONGDOUBLE)
DECLARE_MATVEC_FUNCTION(UBYTE)
DECLARE_MATVEC_FUNCTION(USHORT)
DECLARE_MATVEC_FUNCTION(UINT)
DECLARE_MATVEC_FUNCTION(ULONG)
DECLARE_MATVEC_FUNCTION(ULONGLONG)
DECLARE_MATVEC_FUNCTION(BYTE)
DECLARE_MATVEC_FUNCTION(SHORT)
DECLARE_MATVEC_FUNCTION(INT)
DECLARE_MATVEC_FUNCTION(LONG)
DECLARE_MATVEC_FUNCTION(LONGLONG)
DECLARE_MATVEC_FUNCTION(BOOL)
DECLARE_MATVEC_FUNCTION(OBJECT)

#undef DECLARE_MATVEC_FUNCTION

/***********************************************************************************
** Declaring vecmat functions
***********************************************************************************/
#define DECLARE_VECMAT_FUNCTION(TYPE)                        \
NPY_NO_EXPORT void                                           \
TYPE##_vecmat(char **args,                                   \
              npy_intp const *dimensions,                    \
              npy_intp const *steps,                         \
              void *NPY_UNUSED(func));

DECLARE_VECMAT_FUNCTION(FLOAT)
DECLARE_VECMAT_FUNCTION(DOUBLE)
DECLARE_VECMAT_FUNCTION(LONGDOUBLE)
DECLARE_VECMAT_FUNCTION(HALF)
DECLARE_VECMAT_FUNCTION(CFLOAT)
DECLARE_VECMAT_FUNCTION(CDOUBLE)
DECLARE_VECMAT_FUNCTION(CLONGDOUBLE)
DECLARE_VECMAT_FUNCTION(UBYTE)
DECLARE_VECMAT_FUNCTION(USHORT)
DECLARE_VECMAT_FUNCTION(UINT)
DECLARE_VECMAT_FUNCTION(ULONG)
DECLARE_VECMAT_FUNCTION(ULONGLONG)
DECLARE_VECMAT_FUNCTION(BYTE)
DECLARE_VECMAT_FUNCTION(SHORT)
DECLARE_VECMAT_FUNCTION(INT)
DECLARE_VECMAT_FUNCTION(LONG)
DECLARE_VECMAT_FUNCTION(LONGLONG)
DECLARE_VECMAT_FUNCTION(BOOL)
DECLARE_VECMAT_FUNCTION(OBJECT)

#undef DECLARE_VECMAT_FUNCTION

#ifdef __cplusplus
} // extern "C"
#endif

#endif
