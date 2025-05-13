#ifndef LOG1P_COMPLEX_H
#define LOG1P_COMPLEX_H

#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"

#include <cmath>
#include <complex>
#include <limits>

// For memcpy
#include <cstring>


//
// Trivial C++ wrappers for several npy_* functions.
//

#define CPP_WRAP1(name) \
inline float name(const float x)                \
{                                               \
    return ::npy_ ## name ## f(x);              \
}                                               \
inline double name(const double x)              \
{                                               \
    return ::npy_ ## name(x);                   \
}                                               \
inline long double name(const long double x)    \
{                                               \
    return ::npy_ ## name ## l(x);              \
}                                               \

#define CPP_WRAP2(name) \
inline float name(const float x,                \
                  const float y)                \
{                                               \
    return ::npy_ ## name ## f(x, y);           \
}                                               \
inline double name(const double x,              \
                   const double y)              \
{                                               \
    return ::npy_ ## name(x, y);                \
}                                               \
inline long double name(const long double x,    \
                        const long double y)    \
{                                               \
    return ::npy_ ## name ## l(x, y);           \
}                                               \


namespace npy {

CPP_WRAP1(fabs)
CPP_WRAP1(log)
CPP_WRAP1(log1p)
CPP_WRAP2(atan2)
CPP_WRAP2(hypot)

}

namespace log1p_complex
{

template<typename T>
struct doubled_t {
    T upper;
    T lower;
};

//
// There are three functions below where it is crucial that the
// expressions are not optimized. E.g. `t - (t - x)` must not be
// simplified by the compiler to just `x`. The NO_OPT macro defines
// an attribute that should turn off optimization for the function.
//
// The inclusion of `gnu::target("fpmath=sse")` when __GNUC__ and
// __i386 are defined also turns off the use of the floating-point
// unit '387'.  It is important that when the type is, for example,
// `double`, these functions compute their results with 64 bit
// precision, and not with 80 bit extended precision.
//

#if defined(__clang__)
#define NO_OPT [[clang::optnone]]
#elif defined(__GNUC__)
    #if defined(__i386)
    #define NO_OPT [[gnu::optimize(0),gnu::target("fpmath=sse")]]
    #else
    #define NO_OPT [[gnu::optimize(0)]]
    #endif
#else
#define NO_OPT
#endif

//
// Dekker splitting.  See, for example, Theorem 1 of
//
//   Seppa Linnainmaa, Software for Double-Precision Floating-Point
//   Computations, ACM Transactions on Mathematical Software, Vol 7, No 3,
//   September 1981, pages 272-283.
//
// or Theorem 17 of
//
//   J. R. Shewchuk, Adaptive Precision Floating-Point Arithmetic and
//   Fast Robust Geometric Predicates, CMU-CS-96-140R, from Discrete &
//   Computational Geometry 18(3):305-363, October 1997.
//
template<typename T>
NO_OPT inline void
split(T x, doubled_t<T>& out)
{
    if (std::numeric_limits<T>::digits == 106) {
        // Special case: IBM double-double format.  The value is already
        // split in memory, so there is no need for any calculations.
        std::memcpy(&out, &x, sizeof(out));
    }
    else {
        constexpr int halfprec = (std::numeric_limits<T>::digits + 1)/2;
        T t = ((1ull << halfprec) + 1)*x;
        // The compiler must not be allowed to simplify this expression:
        out.upper = t - (t - x);
        out.lower = x - out.upper;
    }
}

template<typename T>
NO_OPT inline void
two_sum_quick(T x, T y, doubled_t<T>& out)
{
    T r = x + y;
    T e = y - (r - x);
    out.upper = r;
    out.lower = e;
}

template<typename T>
NO_OPT inline void
two_sum(T x, T y, doubled_t<T>& out)
{
    T s = x + y;
    T v = s - x;
    T e = (x - (s - v)) + (y - v);
    out.upper = s;
    out.lower = e;
}

template<typename T>
inline void
double_sum(const doubled_t<T>& x, const doubled_t<T>& y,
           doubled_t<T>& out)
{
    two_sum<T>(x.upper, y.upper, out);
    out.lower += x.lower + y.lower;
    two_sum_quick<T>(out.upper, out.lower, out);
}

template<typename T>
inline void
square(T x, doubled_t<T>& out)
{
    doubled_t<T> xsplit;
    out.upper = x*x;
    split(x, xsplit);
    out.lower = xsplit.lower*xsplit.lower
                - ((out.upper - xsplit.upper*xsplit.upper)
                   - 2*xsplit.lower*xsplit.upper);
}

//
// As the name makes clear, this function computes x**2 + 2*x + y**2.
// It uses doubled_t<T> for the intermediate calculations.
// (That is, we give the floating point type T an upgrayedd, spelled with
// two d's for a double dose of precision.)
//
// The function is used in log1p_complex() to avoid the loss of
// precision that can occur in the expression when x**2 + y**2 ≈ -2*x.
//
template<typename T>
inline T
xsquared_plus_2x_plus_ysquared_dd(T x, T y)
{
    doubled_t<T> x2, y2, twox, sum1, sum2;

    square<T>(x, x2);               // x2 = x**2
    square<T>(y, y2);               // y2 = y**2
    twox.upper = 2*x;               // twox = 2*x
    twox.lower = 0.0;
    double_sum<T>(x2, twox, sum1);  // sum1 = x**2 + 2*x
    double_sum<T>(sum1, y2, sum2);  // sum2 = x**2 + 2*x + y**2
    return sum2.upper;
}

//
// For the float type, the intermediate calculation is done
// with the double type.  We don't need to use doubled_t<float>.
//
inline float
xsquared_plus_2x_plus_ysquared(float x, float y)
{
    double xd = x;
    double yd = y;
    return xd*(2.0 + xd) + yd*yd;
}

//
// For double, we used doubled_t<double> if long double doesn't have
// at least 106 bits of precision.
//
inline double
xsquared_plus_2x_plus_ysquared(double x, double y)
{
    if (std::numeric_limits<long double>::digits >= 106) {
        // Cast to long double for the calculation.
        long double xd = x;
        long double yd = y;
        return xd*(2.0L + xd) + yd*yd;
    }
    else {
        // Use doubled_t<double> for the calculation.
        return xsquared_plus_2x_plus_ysquared_dd<double>(x, y);
    }
}

//
// For long double, we always use doubled_t<long double> for the
// calculation.
//
inline long double
xsquared_plus_2x_plus_ysquared(long double x, long double y)
{
    return xsquared_plus_2x_plus_ysquared_dd<long double>(x, y);
}

//
// Implement log1p(z) for complex inputs that are near the unit circle
// centered at -1+0j.
//
// The function assumes that neither component of z is nan.
//
template<typename T>
inline std::complex<T>
log1p_complex(std::complex<T> z)
{
    T x = z.real();
    T y = z.imag();
    // The input is close to the unit circle centered at -1+0j.
    // Compute x**2 + 2*x + y**2 with higher precision than T.
    // The calculation here is equivalent to log(hypot(x+1, y)),
    // since
    //    log(hypot(x+1, y)) = 0.5*log(x**2 + 2*x + 1 + y**2)
    //                       = 0.5*log1p(x**2 + 2*x + y**2)
    T t = xsquared_plus_2x_plus_ysquared(x, y);
    T lnr = 0.5*npy::log1p(t);
    return std::complex<T>(lnr, npy::atan2(y, x + static_cast<T>(1)));
}

} // namespace log1p_complex

#endif
