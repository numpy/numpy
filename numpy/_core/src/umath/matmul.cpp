#include "matmul.h"


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
DEFINE_DOT_FUNCTION(CFLOAT, npy_cfloat) // TODO: need to fix handling complex
DEFINE_DOT_FUNCTION(CDOUBLE, npy_cdouble)
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
    npy_intp ib1_n, ib2_n, ib2_p, ob_p;

    ib1_n = is1_n * dn;
    ib2_n = is2_n * dn;
    ib2_p = is2_p * dp;
    ob_p = os_p * dp;

    for(npy_intp m = 0; m < dm; m++) {
        for(npy_intp p = 0; p < dp; p++) {
            if constexpr(std::is_same_v<T, npy_cfloat>) {
                npy_csetrealf(op, 0.0);
                npy_csetimagf(op, 0.0);
            } else if constexpr(std::is_same_v<T, npy_cdouble>) {
                npy_csetreal(op, 0.0);
                npy_csetimag(op, 0.0);
            } else if constexpr(std::is_same_v<T, npy_clongdouble>) {
                npy_csetreall(op, 0.0);
                npy_csetimagl(op, 0.0);
            } else {
                *op = 0;
            }

            for(npy_intp n = 0; n < dn; n++) {
                T val1 = *ip1;
                T val2 = *ip2;
                if constexpr(std::is_same_v<T, npy_cfloat>) {
                    auto a = npy_crealf(val1);
                    auto c = npy_crealf(val2);
                    auto b = npy_cimagf(val1);
                    auto d = npy_cimagf(val2);
                    npy_csetrealf(op, npy_crealf(*op) + a * c - b * d);
                } else if constexpr(std::is_same_v<T, npy_cdouble>) {
                    auto a = npy_creal(val1);
                    auto c = npy_creal(val2);
                    auto b = npy_cimag(val1);
                    auto d = npy_cimag(val2);
                    npy_csetreal(op, npy_creal(*op) + a * c - b * d);
                } else if constexpr(std::is_same_v<T, npy_clongdouble>) {
                    auto a = npy_creall(val1);
                    auto c = npy_creall(val2);
                    auto b = npy_cimagl(val1);
                    auto d = npy_cimagl(val2);
                    npy_csetreall(op, npy_creall(*op) + a * c - b * d);
                } else {
                    *op += val1 * val2;
                }

                ip2 += is2_n / sizeof(T);
                ip1 += is1_n / sizeof(T);
            }
            // TODO: Need to fix npy_half

            ip1 -= ib1_n / sizeof(T); // rewind for next column
            ip2 -= ib2_n / sizeof(T);
            op += os_p / sizeof(T);
            ip2 += is2_p / sizeof(T);
        }
        op -= ob_p / sizeof(T);
        ip2 -= ib2_p / sizeof(T);
        ip1 += is1_m / sizeof(T);
        op += os_m / sizeof(T);
    }
}

static void
BOOL_matmul_inner_noblas(npy_bool *ip1, npy_intp is1_m, npy_intp is1_n,
                                    npy_bool *ip2, npy_intp is2_n, npy_intp is2_p,
                                    npy_bool *op, npy_intp os_m, npy_intp os_p,
                                    npy_intp dm, npy_intp dn, npy_intp dp)
{
    npy_intp ib2_p = is2_p * dp;
    npy_intp ob_p = os_p * dp;

    for(npy_intp m = 0; m < dm; m++) {
        for(npy_intp p = 0; p < dp; p++) {
            npy_bool *ip1tmp = ip1;
            npy_bool *ip2tmp = ip2;
            *op = NPY_FALSE;

            for(npy_intp n = 0; n < dn; n++) {
                npy_bool val1 = *ip1tmp;
                npy_bool val2 = *ip2tmp;

                if(val1 != 0 && val2 != 0) {
                    *op = NPY_TRUE;
                    break;
                }
                ip2tmp += is2_n / sizeof(npy_bool);
                ip1tmp += is1_n / sizeof(npy_bool);
            }
            op += os_p / sizeof(npy_bool);
            ip2 += is2_p / sizeof(npy_bool);
        }
        op -= ob_p / sizeof(npy_bool);
        ip2 -= ib2_p / sizeof(npy_bool);
        ip1 += is1_m / sizeof(npy_bool);
        op += os_m / sizeof(npy_bool);
    }
}

static void
OBJECT_matmul_inner_noblas(PyObject **ip1, npy_intp is1_m, npy_intp is1_n,
                            PyObject **ip2, npy_intp is2_n, npy_intp is2_p,
                            PyObject **op, npy_intp os_m, npy_intp os_p,
                            npy_intp dm, npy_intp dn, npy_intp dp)
{
    npy_intp ib1_n = is1_n * dn;
    npy_intp ib2_n = is2_n * dn;
    npy_intp ib2_p = is2_p * dp;
    npy_intp ob_p = os_p * dp;

    PyObject *product = nullptr;
    PyObject *sum_of_products = nullptr;

    for(npy_intp m = 0; m < dm; m++) {
        for(npy_intp p = 0; p < dp; p++) {
            if(dn == 0) {
                sum_of_products = PyLong_FromLong(0);
                if(sum_of_products == nullptr) {
                    return;
                }
            }

            for(npy_intp n = 0; n < dn; n++) {
                PyObject *obj1 = *ip1;
                PyObject *obj2 = *ip2;

                if(obj1 == nullptr) {
                    obj1 = Py_None;
                }
                if(obj2 == nullptr) {
                    obj2 = Py_None;
                }

                product = PyNumber_Multiply(obj1, obj2);
                if(product == nullptr) {
                    Py_XDECREF(sum_of_products);
                    return;
                }

                if(n == 0) {
                    sum_of_products = product;
                } else {
                    Py_SETREF(sum_of_products, PyNumber_Add(sum_of_products, product));
                    Py_DECREF(product);
                    if(sum_of_products == nullptr) {
                        return;
                    }
                }

                ip2 += is2_n / sizeof(PyObject*);
                ip1 += is1_n / sizeof(PyObject*);
            }

            *op = sum_of_products;
            ip1 -= ib1_n / sizeof(PyObject*);
            ip2 -= ib2_n / sizeof(PyObject*);
            op += os_p / sizeof(PyObject*);
            ip2 += is2_p / sizeof(PyObject*);
        }
        op -= ob_p / sizeof(PyObject*);
        ip2 -= ib2_p / sizeof(PyObject*);
        ip1 += is1_m / sizeof(PyObject*);
        op += os_m / sizeof(PyObject*);
    }
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
        _matmul_inner_noblas<T>(ip1, is1_m, is1_n, ip2, is2_n, is2_p, op, os_m, os_p, dm, dn, dp);
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
        for(npy_intp j = 0; j < dm; j++, ip1 += is1_m / sizeof(T), op += os_m / sizeof(T)) {
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
        for(npy_intp j = 0; j < dm; j++, ip2 += is2_m / sizeof(T), op += os_m / sizeof(T)) {
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



/***********************************************************************************
** Defining matmul functions
***********************************************************************************/
#define DEFINE_MATMUL_FUNCTION(NAME, TYPE)                      \
NPY_NO_EXPORT void                                              \
NAME##_matmul(char **args,                                      \
                npy_intp const *dimensions,                     \
                npy_intp const *steps,                          \
                void *NPY_UNUSED(func))                         \
{                                                               \
    matmul<TYPE>(args, dimensions, steps, nullptr);             \
}

DEFINE_MATMUL_FUNCTION(FLOAT, npy_float)
DEFINE_MATMUL_FUNCTION(DOUBLE, npy_double)
DEFINE_MATMUL_FUNCTION(LONGDOUBLE, npy_longdouble)
DEFINE_MATMUL_FUNCTION(HALF, npy_half)
DEFINE_MATMUL_FUNCTION(CFLOAT, npy_cfloat)
DEFINE_MATMUL_FUNCTION(CDOUBLE, npy_cdouble)
DEFINE_MATMUL_FUNCTION(CLONGDOUBLE, npy_clongdouble)
DEFINE_MATMUL_FUNCTION(UBYTE, npy_ubyte)
DEFINE_MATMUL_FUNCTION(USHORT, npy_ushort)
DEFINE_MATMUL_FUNCTION(UINT, npy_uint)
DEFINE_MATMUL_FUNCTION(ULONG, npy_ulong)
DEFINE_MATMUL_FUNCTION(ULONGLONG, npy_ulonglong)
DEFINE_MATMUL_FUNCTION(BYTE, npy_byte)
DEFINE_MATMUL_FUNCTION(SHORT, npy_short)
DEFINE_MATMUL_FUNCTION(INT, npy_int)
DEFINE_MATMUL_FUNCTION(LONG, npy_long)
DEFINE_MATMUL_FUNCTION(LONGLONG, npy_longlong)

#undef DEFINE_MATMUL_FUNCTION

NPY_NO_EXPORT void
BOOL_matmul(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    npy_intp d_outer = *dimensions++;
    npy_intp i_outer;
    npy_intp s0 = steps[0];
    npy_intp s1 = steps[1];
    npy_intp s2 = steps[2];
    npy_intp dm = dimensions[0];
    npy_intp dn = dimensions[1];
    npy_intp dp = dimensions[2];

    npy_intp is1_m = steps[3];
    npy_intp is1_n = steps[4];
    npy_intp is2_n = steps[5];
    npy_intp is2_p = steps[6];
    npy_intp os_m = steps[7];
    npy_intp os_p = steps[8];

    for(i_outer = 0; i_outer < d_outer; i_outer++,
            args[0] += s0, args[1] += s1, args[2] += s2) {
        npy_bool *ip1 = (npy_bool*) args[0];
        npy_bool *ip2 = (npy_bool*) args[1];
        npy_bool *op = (npy_bool*) args[2];
        BOOL_matmul_inner_noblas(ip1, is1_m, is1_n,
                                ip2, is2_n, is2_p,
                                op, os_m, os_p, dm, dn, dp);
    }
}

NPY_NO_EXPORT void
OBJECT_matmul(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    npy_intp d_outer = *dimensions++;
    npy_intp i_outer;
    npy_intp s0 = steps[0];
    npy_intp s1 = steps[1];
    npy_intp s2 = steps[2];
    npy_intp dm = dimensions[0];
    npy_intp dn = dimensions[1];
    npy_intp dp = dimensions[2];

    npy_intp is1_m = steps[3];
    npy_intp is1_n = steps[4];
    npy_intp is2_n = steps[5];
    npy_intp is2_p = steps[6];
    npy_intp os_m = steps[7];
    npy_intp os_p = steps[8];

    for(i_outer = 0; i_outer < d_outer; i_outer++,
            args[0] += s0, args[1] += s1, args[2] += s2) {
        PyObject **ip1 = (PyObject**) args[0];
        PyObject **ip2 = (PyObject**) args[1];
        PyObject **op = (PyObject**) args[2];
        OBJECT_matmul_inner_noblas(ip1, is1_m, is1_n,
                                ip2, is2_n, is2_p,
                                op, os_m, os_p, dm, dn, dp);
    }
}


/***********************************************************************************
** Defining vecdot functions
***********************************************************************************/
#define DEFINE_VECDOT_FUNCTION(NAME, TYPE)                      \
NPY_NO_EXPORT void                                              \
NAME##_vecdot(char **args,                                      \
                   npy_intp const *dimensions,                  \
                   npy_intp const *steps,                       \
                   void *NPY_UNUSED(func))                      \
{                                                               \
    vecdot<TYPE>(args, dimensions, steps, nullptr);             \
}

DEFINE_VECDOT_FUNCTION(FLOAT, npy_float)
DEFINE_VECDOT_FUNCTION(DOUBLE, npy_double)
DEFINE_VECDOT_FUNCTION(LONGDOUBLE, npy_longdouble)
DEFINE_VECDOT_FUNCTION(HALF, npy_half)
DEFINE_VECDOT_FUNCTION(CFLOAT, npy_cfloat)
DEFINE_VECDOT_FUNCTION(CDOUBLE, npy_cdouble)
DEFINE_VECDOT_FUNCTION(CLONGDOUBLE, npy_clongdouble)
DEFINE_VECDOT_FUNCTION(UBYTE, npy_ubyte)
DEFINE_VECDOT_FUNCTION(USHORT, npy_ushort)
DEFINE_VECDOT_FUNCTION(UINT, npy_uint)
DEFINE_VECDOT_FUNCTION(ULONG, npy_ulong)
DEFINE_VECDOT_FUNCTION(ULONGLONG, npy_ulonglong)
DEFINE_VECDOT_FUNCTION(BYTE, npy_byte)
DEFINE_VECDOT_FUNCTION(SHORT, npy_short)
DEFINE_VECDOT_FUNCTION(INT, npy_int)
DEFINE_VECDOT_FUNCTION(LONG, npy_long)
DEFINE_VECDOT_FUNCTION(LONGLONG, npy_longlong)
DEFINE_VECDOT_FUNCTION(BOOL, npy_bool)
DEFINE_VECDOT_FUNCTION(OBJECT, PyObject*)

#undef DEFINE_VECDOT_FUNCTION

/***********************************************************************************
** Defining matvec functions
***********************************************************************************/
#define DEFINE_MATVEC_FUNCTION(NAME, TYPE)                      \
NPY_NO_EXPORT void                                              \
NAME##_matvec(char **args,                                      \
                   npy_intp const *dimensions,                  \
                   npy_intp const *steps,                       \
                   void *NPY_UNUSED(func))                      \
{                                                               \
    matvec<TYPE>(args, dimensions, steps, nullptr);             \
}

DEFINE_MATVEC_FUNCTION(FLOAT, npy_float)
DEFINE_MATVEC_FUNCTION(DOUBLE, npy_double)
DEFINE_MATVEC_FUNCTION(LONGDOUBLE, npy_longdouble)
DEFINE_MATVEC_FUNCTION(HALF, npy_half)
DEFINE_MATVEC_FUNCTION(CFLOAT, npy_cfloat)
DEFINE_MATVEC_FUNCTION(CDOUBLE, npy_cdouble)
DEFINE_MATVEC_FUNCTION(CLONGDOUBLE, npy_clongdouble)
DEFINE_MATVEC_FUNCTION(UBYTE, npy_ubyte)
DEFINE_MATVEC_FUNCTION(USHORT, npy_ushort)
DEFINE_MATVEC_FUNCTION(UINT, npy_uint)
DEFINE_MATVEC_FUNCTION(ULONG, npy_ulong)
DEFINE_MATVEC_FUNCTION(ULONGLONG, npy_ulonglong)
DEFINE_MATVEC_FUNCTION(BYTE, npy_byte)
DEFINE_MATVEC_FUNCTION(SHORT, npy_short)
DEFINE_MATVEC_FUNCTION(INT, npy_int)
DEFINE_MATVEC_FUNCTION(LONG, npy_long)
DEFINE_MATVEC_FUNCTION(LONGLONG, npy_longlong)
DEFINE_MATVEC_FUNCTION(BOOL, npy_bool)
DEFINE_MATVEC_FUNCTION(OBJECT, PyObject*)

#undef DEFINE_MATVEC_FUNCTION

/***********************************************************************************
** Defining vecmat functions
***********************************************************************************/
#define DEFINE_VECMAT_FUNCTION(NAME, TYPE)                      \
NPY_NO_EXPORT void                                              \
NAME##_vecmat(char **args,                                      \
                   npy_intp const *dimensions,                  \
                   npy_intp const *steps,                       \
                   void *NPY_UNUSED(func))                      \
{                                                               \
    vecmat<TYPE>(args, dimensions, steps, nullptr);             \
}

DEFINE_VECMAT_FUNCTION(FLOAT, npy_float)
DEFINE_VECMAT_FUNCTION(DOUBLE, npy_double)
DEFINE_VECMAT_FUNCTION(LONGDOUBLE, npy_longdouble)
DEFINE_VECMAT_FUNCTION(HALF, npy_half)
DEFINE_VECMAT_FUNCTION(CFLOAT, npy_cfloat)
DEFINE_VECMAT_FUNCTION(CDOUBLE, npy_cdouble)
DEFINE_VECMAT_FUNCTION(CLONGDOUBLE, npy_clongdouble)
DEFINE_VECMAT_FUNCTION(UBYTE, npy_ubyte)
DEFINE_VECMAT_FUNCTION(USHORT, npy_ushort)
DEFINE_VECMAT_FUNCTION(UINT, npy_uint)
DEFINE_VECMAT_FUNCTION(ULONG, npy_ulong)
DEFINE_VECMAT_FUNCTION(ULONGLONG, npy_ulonglong)
DEFINE_VECMAT_FUNCTION(BYTE, npy_byte)
DEFINE_VECMAT_FUNCTION(SHORT, npy_short)
DEFINE_VECMAT_FUNCTION(INT, npy_int)
DEFINE_VECMAT_FUNCTION(LONG, npy_long)
DEFINE_VECMAT_FUNCTION(LONGLONG, npy_longlong)
DEFINE_VECMAT_FUNCTION(BOOL, npy_bool)
DEFINE_VECMAT_FUNCTION(OBJECT, PyObject*)

#undef DEFINE_VECMAT_FUNCTION
