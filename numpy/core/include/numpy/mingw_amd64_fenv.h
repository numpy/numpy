/*
 * Those are mostly copies from BSD mlib
 */
#include <inttypes.h>
typedef struct {
	struct {
		uint32_t	__control;
		uint32_t	__status;
		uint32_t	__tag;
		char		__other[16];
	} __x87;
	uint32_t		__mxcsr;
} npy_fenv_t;

typedef	uint16_t	npy_fexcept_t;

/* Exception flags */
#define	NPY_FE_INVALID		0x01
#define	NPY_FE_DENORMAL		0x02
#define	NPY_FE_DIVBYZERO	0x04
#define	NPY_FE_OVERFLOW		0x08
#define	NPY_FE_UNDERFLOW	0x10
#define	NPY_FE_INEXACT		0x20
#define	NPY_FE_ALL_EXCEPT	(NPY_FE_DIVBYZERO | NPY_FE_DENORMAL | \
				NPY_FE_INEXACT | NPY_FE_INVALID | \
				NPY_FE_OVERFLOW | NPY_FE_UNDERFLOW)

/* Assembly macros */
#define	__fldcw(__cw)		__asm __volatile("fldcw %0" : : "m" (__cw))
#define	__fldenv(__env)		__asm __volatile("fldenv %0" : : "m" (__env))
#define	__fldenvx(__env)	__asm __volatile("fldenv %0" : : "m" (__env)  \
				: "st", "st(1)", "st(2)", "st(3)", "st(4)",   \
				"st(5)", "st(6)", "st(7)")
#define	__fnclex()		__asm __volatile("fnclex")
#define	__fnstenv(__env)	__asm __volatile("fnstenv %0" : "=m" (*(__env)))
#define	__fnstcw(__cw)		__asm __volatile("fnstcw %0" : "=m" (*(__cw)))
#define	__fnstsw(__sw)		__asm __volatile("fnstsw %0" : "=am" (*(__sw)))
#define	__fwait()		__asm __volatile("fwait")
#define	__ldmxcsr(__csr)	__asm __volatile("ldmxcsr %0" : : "m" (__csr))
#define	__stmxcsr(__csr)	__asm __volatile("stmxcsr %0" : "=m" (*(__csr)))

static __inline int npy_feclearexcept(int __excepts)
{
	npy_fenv_t __env;

	if (__excepts == NPY_FE_ALL_EXCEPT) {
		__fnclex();
	} else {
		__fnstenv(&__env.__x87);
		__env.__x87.__status &= ~__excepts;
		__fldenv(__env.__x87);
	}
	__stmxcsr(&__env.__mxcsr);
	__env.__mxcsr &= ~__excepts;
	__ldmxcsr(__env.__mxcsr);
	return (0);
}

static __inline int npy_fetestexcept(int __excepts)
{
	int __mxcsr, __status;

	__stmxcsr(&__mxcsr);
	__fnstsw(&__status);
	return ((__status | __mxcsr) & __excepts);
}

static __inline int
npy_fesetexceptflag(const npy_fexcept_t *flagp, int excepts)
{
	npy_fenv_t env;

	__fnstenv(&env.__x87);
	env.__x87.__status &= ~excepts;
	env.__x87.__status |= *flagp & excepts;
	__fldenv(env.__x87);

	__stmxcsr(&env.__mxcsr);
	env.__mxcsr &= ~excepts;
	env.__mxcsr |= *flagp & excepts;
	__ldmxcsr(env.__mxcsr);

	return (0);
}

int npy_feraiseexcept(int excepts)
{
	npy_fexcept_t ex = excepts;

	npy_fesetexceptflag(&ex, excepts);
	__fwait();
	return (0);
}

#undef __fldcw
#undef __fldenv
#undef __fldenvx
#undef	__fnclex
#undef	__fnstenv
#undef	__fnstcw
#undef	__fnstsw
#undef	__fwait
#undef	__ldmxcsr
#undef	__stmxcsr

