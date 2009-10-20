/*-
 * Copyright (c) 2004 David Schultz <das@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * $FreeBSD$
 */

#ifndef _FENV_H_
#define _FENV_H_

#include <sys/cdefs.h>
#include <sys/types.h>

typedef struct {
        __uint32_t      __control;
        __uint32_t      __status;
        __uint32_t      __tag;
        char            __other[16];
} fenv_t;

typedef __uint16_t      fexcept_t;

/* Exception flags */
#define FE_INVALID      0x01
#define FE_DENORMAL     0x02
#define FE_DIVBYZERO    0x04
#define FE_OVERFLOW     0x08
#define FE_UNDERFLOW    0x10
#define FE_INEXACT      0x20
#define FE_ALL_EXCEPT   (FE_DIVBYZERO | FE_DENORMAL | FE_INEXACT | \
                         FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW)

/* Rounding modes */
#define FE_TONEAREST    0x0000
#define FE_DOWNWARD     0x0400
#define FE_UPWARD       0x0800
#define FE_TOWARDZERO   0x0c00
#define _ROUND_MASK     (FE_TONEAREST | FE_DOWNWARD | \
                         FE_UPWARD | FE_TOWARDZERO)

__BEGIN_DECLS

/* Default floating-point environment */
extern const fenv_t     npy__fe_dfl_env;
#define FE_DFL_ENV      (&npy__fe_dfl_env)

#define __fldcw(__cw)           __asm __volatile("fldcw %0" : : "m" (__cw))
#define __fldenv(__env)         __asm __volatile("fldenv %0" : : "m" (__env))
#define __fnclex()              __asm __volatile("fnclex")
#define __fnstenv(__env)        __asm __volatile("fnstenv %0" : "=m" (*(__env)))
#define __fnstcw(__cw)          __asm __volatile("fnstcw %0" : "=m" (*(__cw)))
#define __fnstsw(__sw)          __asm __volatile("fnstsw %0" : "=am" (*(__sw)))
#define __fwait()               __asm __volatile("fwait")

static __inline int
feclearexcept(int __excepts)
{
        fenv_t __env;

        if (__excepts == FE_ALL_EXCEPT) {
                __fnclex();
        } else {
                __fnstenv(&__env);
                __env.__status &= ~__excepts;
                __fldenv(__env);
        }
        return (0);
}

static __inline int
fegetexceptflag(fexcept_t *__flagp, int __excepts)
{
        __uint16_t __status;

        __fnstsw(&__status);
        *__flagp = __status & __excepts;
        return (0);
}

static __inline int
fesetexceptflag(const fexcept_t *__flagp, int __excepts)
{
        fenv_t __env;

        __fnstenv(&__env);
        __env.__status &= ~__excepts;
        __env.__status |= *__flagp & __excepts;
        __fldenv(__env);
        return (0);
}

static __inline int
feraiseexcept(int __excepts)
{
        fexcept_t __ex = __excepts;

        fesetexceptflag(&__ex, __excepts);
        __fwait();
        return (0);
}

static __inline int
fetestexcept(int __excepts)
{
        __uint16_t __status;

        __fnstsw(&__status);
        return (__status & __excepts);
}

static __inline int
fegetround(void)
{
        int __control;

        __fnstcw(&__control);
        return (__control & _ROUND_MASK);
}

static __inline int
fesetround(int __round)
{
        int __control;

        if (__round & ~_ROUND_MASK)
                return (-1);
        __fnstcw(&__control);
        __control &= ~_ROUND_MASK;
        __control |= __round;
        __fldcw(__control);
        return (0);
}

static __inline int
fegetenv(fenv_t *__envp)
{
        int __control;

        /*
         * fnstenv masks all exceptions, so we need to save and
         * restore the control word to avoid this side effect.
         */
        __fnstcw(&__control);
        __fnstenv(__envp);
        __fldcw(__control);
        return (0);
}

static __inline int
feholdexcept(fenv_t *__envp)
{

        __fnstenv(__envp);
        __fnclex();
        return (0);
}

static __inline int
fesetenv(const fenv_t *__envp)
{

        __fldenv(*__envp);
        return (0);
}

static __inline int
feupdateenv(const fenv_t *__envp)
{
        __uint16_t __status;

        __fnstsw(&__status);
        __fldenv(*__envp);
        feraiseexcept(__status & FE_ALL_EXCEPT);
        return (0);
}

#if __BSD_VISIBLE

static __inline int
fesetmask(int __mask)
{
        int __control;

        __fnstcw(&__control);
        __mask = (__control | FE_ALL_EXCEPT) & ~__mask;
        __fldcw(__mask);
        return (~__control & FE_ALL_EXCEPT);
}

static __inline int
fegetmask(void)
{
        int __control;

        __fnstcw(&__control);
        return (~__control & FE_ALL_EXCEPT);
}

#endif /* __BSD_VISIBLE */

__END_DECLS

#endif  /* !_FENV_H_ */
