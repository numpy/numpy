/*
 * Functions to set the floating point status word.
 * keep in sync with NO_FLOATING_POINT_SUPPORT in ufuncobject.h
 */

/* This include is wrapped so that these functions can also be called when
 * doing config tests.
 */
#ifndef CONFIG_TESTS
#include "npy_math_common.h"
#endif

#if (defined(__unix__) || defined(unix)) && !defined(USG)
#include <sys/param.h>
#endif

/* Solaris --------------------------------------------------------*/
/* --------ignoring SunOS ieee_flags approach, someone else can
**         deal with that! */
#if defined(sun) || defined(__BSD__) || defined(__OpenBSD__) || \
    (defined(__FreeBSD__) && (__FreeBSD_version < 502114)) || \
    defined(__NetBSD__)
#include <ieeefp.h>

int
npy_get_floatstatus(void)
{
    int fpstatus = fpgetsticky();
    return ((FP_X_DZ  & fpstatus) ? NPY_FPE_DIVIDEBYZERO : 0) |
           ((FP_X_OFL & fpstatus) ? NPY_FPE_OVERFLOW : 0) |
           ((FP_X_UFL & fpstatus) ? NPY_FPE_UNDERFLOW : 0) |
           ((FP_X_INV & fpstatus) ? NPY_FPE_INVALID : 0);
}

int
npy_clear_floatstatus(void)
{
    int fpstatus = npy_get_floatstatus();
    fpsetsticky(0);

    return fpstatus;
}

void
npy_set_floatstatus_divbyzero(void)
{
    fpsetsticky(FP_X_DZ);
}

void
npy_set_floatstatus_overflow(void)
{
    fpsetsticky(FP_X_OFL);
}

void
npy_set_floatstatus_underflow(void)
{
    fpsetsticky(FP_X_UFL);
}

void
npy_set_floatstatus_invalid(void)
{
    fpsetsticky(FP_X_INV);
}


#elif defined(__GLIBC__) || defined(__APPLE__) || \
      defined(__CYGWIN__) || defined(__MINGW32__) || \
      (defined(__FreeBSD__) && (__FreeBSD_version >= 502114))

# if defined(__GLIBC__) || defined(__APPLE__) || \
     defined(__MINGW32__) || defined(__FreeBSD__)
#  include <fenv.h>
# elif defined(__CYGWIN__)
#  include "numpy/fenv/fenv.h"
# endif

int
npy_get_floatstatus(void)
{
    int fpstatus = fetestexcept(FE_DIVBYZERO | FE_OVERFLOW |
                                FE_UNDERFLOW | FE_INVALID);

    return ((FE_DIVBYZERO  & fpstatus) ? NPY_FPE_DIVIDEBYZERO : 0) |
           ((FE_OVERFLOW   & fpstatus) ? NPY_FPE_OVERFLOW : 0) |
           ((FE_UNDERFLOW  & fpstatus) ? NPY_FPE_UNDERFLOW : 0) |
           ((FE_INVALID    & fpstatus) ? NPY_FPE_INVALID : 0);
}

int
npy_clear_floatstatus(void)
{
    /* testing float status is 50-100 times faster than clearing on x86 */
    int fpstatus = npy_get_floatstatus();
    if (fpstatus != 0) {
        feclearexcept(FE_DIVBYZERO | FE_OVERFLOW |
                      FE_UNDERFLOW | FE_INVALID);
    }

    return fpstatus;
}


void
npy_set_floatstatus_divbyzero(void)
{
    feraiseexcept(FE_DIVBYZERO);
}

void
npy_set_floatstatus_overflow(void)
{
    feraiseexcept(FE_OVERFLOW);
}

void
npy_set_floatstatus_underflow(void)
{
    feraiseexcept(FE_UNDERFLOW);
}

void
npy_set_floatstatus_invalid(void)
{
    feraiseexcept(FE_INVALID);
}

#elif defined(_AIX)
#include <float.h>
#include <fpxcp.h>

int
npy_get_floatstatus(void)
{
    int fpstatus = fp_read_flag();
    return ((FP_DIV_BY_ZERO & fpstatus) ? NPY_FPE_DIVIDEBYZERO : 0) |
           ((FP_OVERFLOW & fpstatus) ? NPY_FPE_OVERFLOW : 0) |
           ((FP_UNDERFLOW & fpstatus) ? NPY_FPE_UNDERFLOW : 0) |
           ((FP_INVALID & fpstatus) ? NPY_FPE_INVALID : 0);
}

int
npy_clear_floatstatus(void)
{
    int fpstatus = npy_get_floatstatus();
    fp_swap_flag(0);

    return fpstatus;
}

void
npy_set_floatstatus_divbyzero(void)
{
    fp_raise_xcp(FP_DIV_BY_ZERO);
}

void
npy_set_floatstatus_overflow(void)
{
    fp_raise_xcp(FP_OVERFLOW);
}

void
npy_set_floatstatus_underflow(void)
{
    fp_raise_xcp(FP_UNDERFLOW);
}

void npy_set_floatstatus_invalid(void)
{
    fp_raise_xcp(FP_INVALID);
}

#else

/* MS Windows -----------------------------------------------------*/
#if defined(_MSC_VER)

#include <float.h>


int
npy_get_floatstatus(void)
{
#if defined(_WIN64)
    int fpstatus = _statusfp();
#else
    /* windows enables sse on 32 bit, so check both flags */
    int fpstatus, fpstatus2;
    _statusfp2(&fpstatus, &fpstatus2);
    fpstatus |= fpstatus2;
#endif
    return ((SW_ZERODIVIDE & fpstatus) ? NPY_FPE_DIVIDEBYZERO : 0) |
           ((SW_OVERFLOW & fpstatus) ? NPY_FPE_OVERFLOW : 0) |
           ((SW_UNDERFLOW & fpstatus) ? NPY_FPE_UNDERFLOW : 0) |
           ((SW_INVALID & fpstatus) ? NPY_FPE_INVALID : 0);
}

int
npy_clear_floatstatus(void)
{
    int fpstatus = npy_get_floatstatus();
    _clearfp();

    return fpstatus;
}

/*  OSF/Alpha (Tru64)  ---------------------------------------------*/
#elif defined(__osf__) && defined(__alpha)

#include <machine/fpu.h>

int
npy_get_floatstatus(void)
{
    unsigned long fpstatus = ieee_get_fp_control();

    return  ((IEEE_STATUS_DZE & fpstatus) ? NPY_FPE_DIVIDEBYZERO : 0) |
            ((IEEE_STATUS_OVF & fpstatus) ? NPY_FPE_OVERFLOW : 0) |
            ((IEEE_STATUS_UNF & fpstatus) ? NPY_FPE_UNDERFLOW : 0) |
            ((IEEE_STATUS_INV & fpstatus) ? NPY_FPE_INVALID : 0);
}

int
npy_clear_floatstatus(void)
{
    long fpstatus = npy_get_floatstatus();
    /* clear status bits as well as disable exception mode if on */
    ieee_set_fp_control(0);

    return fpstatus;
}

#else

int
npy_get_floatstatus(void)
{
    return 0;
}

int
npy_clear_floatstatus(void)
{
    return 0;
}

#endif

/*
 * By using a volatile floating point value,
 * the compiler is forced to actually do the requested
 * operations because of potential concurrency.
 *
 * We shouldn't write multiple values to a single
 * global here, because that would cause
 * a race condition.
 */
static volatile double _npy_floatstatus_x;
static volatile double _npy_floatstatus_zero = 0.0;
static volatile double _npy_floatstatus_big = 1e300;
static volatile double _npy_floatstatus_small = 1e-300;
static volatile double _npy_floatstatus_inf;

void
npy_set_floatstatus_divbyzero(void)
{
    _npy_floatstatus_x = 1.0 / _npy_floatstatus_zero;
}

void
npy_set_floatstatus_overflow(void)
{
    _npy_floatstatus_x = _npy_floatstatus_big * 1e300;
}

void
npy_set_floatstatus_underflow(void)
{
    _npy_floatstatus_x = _npy_floatstatus_small * 1e-300;
}

void
npy_set_floatstatus_invalid(void)
{
    _npy_floatstatus_inf = NPY_INFINITY;
    _npy_floatstatus_x = _npy_floatstatus_inf - NPY_INFINITY;
}

#endif
