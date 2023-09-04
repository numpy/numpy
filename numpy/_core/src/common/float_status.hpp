#ifndef NUMPY_CORE_SRC_COMMON_FLOAT_STATUS_HPP
#define NUMPY_CORE_SRC_COMMON_FLOAT_STATUS_HPP

#include "npstd.hpp"

#include <fenv.h>

namespace np {

/// @addtogroup cpp_core_utility
/// @{
/**
 * Class wraps floating-point environment operations,
 * provides lazy access to its functionality.
 */
class FloatStatus {
 public:
/*
 * According to the C99 standard FE_DIVBYZERO, etc. may not be provided when
 * unsupported.  In such cases NumPy will not report these correctly, but we
 * should still allow compiling (whether tests pass or not).
 * By defining them as 0 locally, we make them no-ops.  Unlike these defines,
 * for example `musl` still defines all of the functions (as no-ops):
 *     https://git.musl-libc.org/cgit/musl/tree/src/fenv/fenv.c
 * and does similar replacement in its tests:
 * http://nsz.repo.hu/git/?p=libc-test;a=blob;f=src/common/mtest.h;h=706c1ba23ea8989b17a2f72ed1a919e187c06b6a;hb=HEAD#l30
 */
#ifdef FE_DIVBYZERO
    static constexpr int kDivideByZero = FE_DIVBYZERO;
#else
    static constexpr int kDivideByZero = 0;
#endif
#ifdef FE_INVALID
    static constexpr int kInvalid = FE_INVALID;
#else
    static constexpr int kInvalid = 0;
#endif
#ifdef FE_INEXACT
    static constexpr int kInexact = FE_INEXACT;
#else
    static constexpr int kInexact = 0;
#endif
#ifdef FE_OVERFLOW
    static constexpr int kOverflow = FE_OVERFLOW;
#else
    static constexpr int kOverflow = 0;
#endif
#ifdef FE_UNDERFLOW
    static constexpr int kUnderflow = FE_UNDERFLOW;
#else
    static constexpr int kUnderflow = 0;
#endif
    static constexpr int kAllExcept = (kDivideByZero | kInvalid | kInexact |
                                       kOverflow | kUnderflow);

    FloatStatus(bool clear_on_dst=true)
        : clear_on_dst_(clear_on_dst)
    {
        if constexpr (kAllExcept != 0) {
            fpstatus_ = fetestexcept(kAllExcept);
        }
        else {
            fpstatus_ = 0;
        }
    }
    ~FloatStatus()
    {
        if constexpr (kAllExcept != 0) {
            if (fpstatus_ != 0 && clear_on_dst_) {
                feclearexcept(kAllExcept);
            }
        }
    }
    constexpr bool IsDivideByZero() const
    {
        return (fpstatus_ & kDivideByZero) != 0;
    }
    constexpr bool IsInexact() const
    {
        return (fpstatus_ & kInexact) != 0;
    }
    constexpr bool IsInvalid() const
    {
        return (fpstatus_ & kInvalid) != 0;
    }
    constexpr bool IsOverFlow() const
    {
        return (fpstatus_ & kOverflow) != 0;
    }
    constexpr bool IsUnderFlow() const
    {
        return (fpstatus_ & kUnderflow) != 0;
    }
    static void RaiseDivideByZero()
    {
        if constexpr (kDivideByZero != 0) {
            feraiseexcept(kDivideByZero);
        }
    }
    static void RaiseInexact()
    {
        if constexpr (kInexact != 0) {
            feraiseexcept(kInexact);
        }
    }
    static void RaiseInvalid()
    {
        if constexpr (kInvalid != 0) {
            feraiseexcept(kInvalid);
        }
    }
    static void RaiseOverflow()
    {
        if constexpr (kOverflow != 0) {
            feraiseexcept(kOverflow);
        }
    }
    static void RaiseUnderflow()
    {
        if constexpr (kUnderflow != 0) {
            feraiseexcept(kUnderflow);
        }
    }

  private:
    bool clear_on_dst_;
    int fpstatus_;
};

/// @} cpp_core_utility
} // namespace np

#endif // NUMPY_CORE_SRC_COMMON_FLOAT_STATUS_HPP

