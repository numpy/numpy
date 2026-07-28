#ifndef NUMPY_CORE_SRC_COMMON_NPY_FENV_H_
#define NUMPY_CORE_SRC_COMMON_NPY_FENV_H_

#include <fenv.h>

#ifdef __EMSCRIPTEN__

#ifdef __cplusplus
extern "C" {
#endif

#undef FE_INVALID
#undef FE_DIVBYZERO
#undef FE_OVERFLOW
#undef FE_UNDERFLOW
#undef FE_INEXACT
#undef FE_ALL_EXCEPT

#define FE_INVALID   0x01
#define FE_DIVBYZERO 0x02
#define FE_OVERFLOW  0x04
#define FE_UNDERFLOW 0x08
#define FE_INEXACT   0x10
#define FE_ALL_EXCEPT \
    (FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT)

extern __attribute__((visibility("default"))) int npy_wasm_fenv_flags;

static inline int
npy_wasm_fetestexcept(int excepts)
{
    return npy_wasm_fenv_flags & excepts & FE_ALL_EXCEPT;
}

static inline int
npy_wasm_feclearexcept(int excepts)
{
    npy_wasm_fenv_flags &= ~(excepts & FE_ALL_EXCEPT);
    return 0;
}

static inline int
npy_wasm_feraiseexcept(int excepts)
{
    npy_wasm_fenv_flags |= excepts & FE_ALL_EXCEPT;
    return 0;
}

static inline int
npy_wasm_fegetexceptflag(fexcept_t *flagp, int excepts)
{
    *flagp = (fexcept_t)(npy_wasm_fenv_flags & excepts & FE_ALL_EXCEPT);
    return 0;
}

static inline int
npy_wasm_fesetexceptflag(const fexcept_t *flagp, int excepts)
{
    excepts &= FE_ALL_EXCEPT;
    npy_wasm_fenv_flags = (npy_wasm_fenv_flags & ~excepts) |
                          ((int)*flagp & excepts);
    return 0;
}

#define fetestexcept(excepts)          npy_wasm_fetestexcept(excepts)
#define feclearexcept(excepts)         npy_wasm_feclearexcept(excepts)
#define feraiseexcept(excepts)         npy_wasm_feraiseexcept(excepts)
#define fegetexceptflag(flagp, excpts) npy_wasm_fegetexceptflag(flagp, excpts)
#define fesetexceptflag(flagp, excpts) npy_wasm_fesetexceptflag(flagp, excpts)

#ifdef __cplusplus
}
#endif

#endif  /* __EMSCRIPTEN__ */

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
#ifndef FE_DIVBYZERO
    #define FE_DIVBYZERO 0
#endif
#ifndef FE_OVERFLOW
    #define FE_OVERFLOW 0
#endif
#ifndef FE_UNDERFLOW
    #define FE_UNDERFLOW 0
#endif
#ifndef FE_INVALID
    #define FE_INVALID 0
#endif

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_FENV_H_ */
