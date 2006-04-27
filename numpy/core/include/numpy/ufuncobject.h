#ifndef Py_UFUNCOBJECT_H
#define Py_UFUNCOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#define MAX_ARGS 40

typedef void (*PyUFuncGenericFunction) (char **, intp *, intp *, void *);

typedef struct {
	PyObject_HEAD
	int nin, nout, nargs;
	int identity;
	PyUFuncGenericFunction *functions;
	void **data;
	int ntypes;
	int check_return;
	char *name, *types;
	char *doc;
        void *ptr;
        PyObject *obj;
	PyObject *userloops;
} PyUFuncObject;

#include "arrayobject.h"

#ifdef PY_ARRAY_TYPES_PREFIX
#  define CAT2(x,y)   x ## y
#  define CAT(x,y)    CAT2(x,y)
#  define NS(name)    CAT(PY_ARRAY_TYPES_PREFIX, name)
#  define intp        NS(intp)
#endif

#define UFUNC_ERR_IGNORE 0
#define UFUNC_ERR_WARN   1
#define UFUNC_ERR_RAISE  2
#define UFUNC_ERR_CALL   3

	/* Python side integer mask */

#define UFUNC_MASK_DIVIDEBYZERO 0x03
#define UFUNC_MASK_OVERFLOW 0x0c
#define UFUNC_MASK_UNDERFLOW 0x30
#define UFUNC_MASK_INVALID 0xc0

#define UFUNC_SHIFT_DIVIDEBYZERO 0
#define UFUNC_SHIFT_OVERFLOW     2
#define UFUNC_SHIFT_UNDERFLOW    4
#define UFUNC_SHIFT_INVALID      6       


/* platform-dependent code translates floating point
   status to an integer sum of these values
*/
#define UFUNC_FPE_DIVIDEBYZERO  1
#define UFUNC_FPE_OVERFLOW      2
#define UFUNC_FPE_UNDERFLOW     4
#define UFUNC_FPE_INVALID       8
	
#define UFUNC_ERR_DEFAULT 0  /* Default error mode */

	/* Only internal -- not exported, yet*/
typedef struct {
	/* Multi-iterator portion --- needs to be present in this order 
	   to work with PyArray_Broadcast */
	PyObject_HEAD
	int  numiter;
	intp size;
	intp index;
	int nd;
	intp dimensions[MAX_DIMS];	
	PyArrayIterObject *iters[MAX_ARGS];
        /*  End of Multi-iterator portion */

	/* The ufunc */
	PyUFuncObject *ufunc;
	
	/* The error handling */
	int errormask;         /* Integer showing desired error handling */
	PyObject *errobj;      /* currently a tuple with 
				  (string, func or None)
			       */

	/* Specific function and data to use */
	PyUFuncGenericFunction function;
	void *funcdata;

	/* Loop method */
	int meth;

	/* Whether we need to copy to a buffer or not.*/
	int needbuffer[MAX_ARGS];
	int leftover;
	int ninnerloops;
	int lastdim;
	
	/* Whether or not to swap */
	int swap[MAX_ARGS];

	/* Buffers for the loop */
	char *buffer[MAX_ARGS];
	int bufsize;
	intp bufcnt;
	char *dptr[MAX_ARGS];

	/* For casting */
	char *castbuf[MAX_ARGS];
	PyArray_VectorUnaryFunc *cast[MAX_ARGS];

	/* usually points to buffer but when a cast is to be
	   done it switches for that argument to castbuf.
	*/
	char *bufptr[MAX_ARGS];  

	/* Steps filled in from iters or sizeof(item)
	   depending on loop method. 
	*/
	intp steps[MAX_ARGS];

        int obj;  /* This loop uses object arrays */
        int notimplemented; /* The loop caused notimplemented */	
        int objfunc; /* This loop calls object functions 
                        (an inner-loop function with argument types */
} PyUFuncLoopObject;

/* Could make this more clever someday */
#define UFUNC_MAXIDENTITY 32

typedef struct {
        PyObject_HEAD
        PyArrayIterObject *it;
        PyArrayObject *ret;
	PyArrayIterObject *rit;   /* Needed for Accumulate */
        int  outsize;
	intp  index;
	intp  size;
        char idptr[UFUNC_MAXIDENTITY];

	/* The ufunc */
	PyUFuncObject *ufunc;

	/* The error handling */
	int errormask;
	PyObject *errobj;

        PyUFuncGenericFunction function;
        void *funcdata;
        int meth;
        int swap;

        char *buffer;
        int bufsize;

        char *castbuf;
        PyArray_VectorUnaryFunc *cast;

        char *bufptr[3];
        intp steps[3];

        intp N;
        int  instrides;
        int  insize;
        char *inptr;

        /* For copying small arrays */
        PyObject *decref;

        int obj;

} PyUFuncReduceObject;


#if defined(ALLOW_THREADS)
#define LOOP_BEGIN_THREADS if (!(loop->obj)) {_save = PyEval_SaveThread();}
#define LOOP_END_THREADS   if (!(loop->obj)) {PyEval_RestoreThread(_save);}
#else
#define LOOP_BEGIN_THREADS
#define LOOP_END_THREADS
#endif

#define PyUFunc_One 1
#define PyUFunc_Zero 0
#define PyUFunc_None -1

#define UFUNC_REDUCE 0
#define UFUNC_ACCUMULATE 1
#define UFUNC_REDUCEAT 2
#define UFUNC_OUTER 3


typedef struct {
        int nin;
        int nout;
        PyObject *callable;
} PyUFunc_PyFuncData;


#include "__ufunc_api.h"

#define UFUNC_PYVALS_NAME "UFUNC_PYVALS"

#define UFUNC_CHECK_ERROR(arg)                                         \
	if (((arg)->obj && PyErr_Occurred()) ||                        \
            ((arg)->errormask &&                                       \
             PyUFunc_checkfperr((arg)->errormask,                      \
                                (arg)->errobj)))                       \
		goto fail

/* This code checks the IEEE status flags in a platform-dependent way */
/* Adapted from Numarray  */

/*  OSF/Alpha (Tru64)  ---------------------------------------------*/
#if defined(__osf__) && defined(__alpha)

#include <machine/fpu.h>

#define UFUNC_CHECK_STATUS(ret) {		\
	unsigned long fpstatus;		        \
						\
	fpstatus = ieee_get_fp_control();				\
	/* clear status bits as well as disable exception mode if on */ \
	ieee_set_fp_control( 0 );					\
	ret = ((IEEE_STATUS_DZE & fpstatus) ? UFUNC_FPE_DIVIDEBYZERO : 0) \
		| ((IEEE_STATUS_OVF & fpstatus) ? UFUNC_FPE_OVERFLOW : 0) \
		| ((IEEE_STATUS_UNF & fpstatus) ? UFUNC_FPE_UNDERFLOW : 0) \
		| ((IEEE_STATUS_INV & fpstatus) ? UFUNC_FPE_INVALID : 0); \
	}
	
/* MS Windows -----------------------------------------------------*/
#elif defined(_MSC_VER) 

#include <float.h>

#define UFUNC_CHECK_STATUS(ret) {		 \
	int fpstatus = (int) _clearfp();			\
									\
	ret = ((SW_ZERODIVIDE & fpstatus) ? UFUNC_FPE_DIVIDEBYZERO : 0)	\
		| ((SW_OVERFLOW & fpstatus) ? UFUNC_FPE_OVERFLOW : 0)	\
		| ((SW_UNDERFLOW & fpstatus) ? UFUNC_FPE_UNDERFLOW : 0)	\
		| ((SW_INVALID & fpstatus) ? UFUNC_FPE_INVALID : 0);	\
	}

#define isnan(x) (_isnan((double)(x)))
#define isinf(x) ((_fpclass((double)(x)) == _FPCLASS_PINF) ||	\
		  (_fpclass((double)(x)) == _FPCLASS_NINF))
#define isfinite(x) (_finite((double) x))
	
/* Solaris --------------------------------------------------------*/
/* --------ignoring SunOS ieee_flags approach, someone else can
**         deal with that! */
#elif defined(sun) || defined(__BSD__) 
#include <ieeefp.h>

#define UFUNC_CHECK_STATUS(ret) {				\
	int fpstatus;						\
								\
	fpstatus = (int) fpgetsticky();					\
	ret = ((FP_X_DZ  & fpstatus) ? UFUNC_FPE_DIVIDEBYZERO : 0)	\
		| ((FP_X_OFL & fpstatus) ? UFUNC_FPE_OVERFLOW : 0)	\
		| ((FP_X_UFL & fpstatus) ? UFUNC_FPE_UNDERFLOW : 0)	\
		| ((FP_X_INV & fpstatus) ? UFUNC_FPE_INVALID : 0);	\
	(void) fpsetsticky(0);						\
	}
	
#elif defined(linux) || defined(__APPLE__) || defined(__CYGWIN__) || defined(__MINGW32__)

#if defined(__GLIBC__) || defined(__APPLE__) || defined(__MINGW32__)
#include <fenv.h>
#elif defined(__CYGWIN__)
#include "fenv/fenv.c"
#endif

#define UFUNC_CHECK_STATUS(ret) {                                       \
	int fpstatus = (int) fetestexcept(FE_DIVBYZERO | FE_OVERFLOW |	\
					  FE_UNDERFLOW | FE_INVALID);	\
	ret = ((FE_DIVBYZERO  & fpstatus) ? UFUNC_FPE_DIVIDEBYZERO : 0) \
		| ((FE_OVERFLOW   & fpstatus) ? UFUNC_FPE_OVERFLOW : 0)	\
		| ((FE_UNDERFLOW  & fpstatus) ? UFUNC_FPE_UNDERFLOW : 0) \
		| ((FE_INVALID    & fpstatus) ? UFUNC_FPE_INVALID : 0);	\
	(void) feclearexcept(FE_DIVBYZERO | FE_OVERFLOW |		\
			     FE_UNDERFLOW | FE_INVALID);		\
}

#define generate_divbyzero_error() feraiseexcept(FE_DIVBYZERO)
#define generate_overflow_error() feraiseexcept(FE_OVERFLOW)
	
#elif defined(_AIX)

#include <float.h>
#include <fpxcp.h>

#define UFUNC_CHECK_STATUS(ret) { \
	fpflag_t fpstatus; \
                                                \
	fpstatus = fp_read_flag(); \
	ret = ((FP_DIV_BY_ZERO & fpstatus) ? UFUNC_FPE_DIVIDEBYZERO : 0) \
		| ((FP_OVERFLOW & fpstatus) ? UFUNC_FPE_OVERFLOW : 0)	\
		| ((FP_UNDERFLOW & fpstatus) ? UFUNC_FPE_UNDERFLOW : 0) \
		| ((FP_INVALID & fpstatus) ? UFUNC_FPE_INVALID : 0); \
	fp_clr_flag( FP_DIV_BY_ZERO | FP_OVERFLOW | FP_UNDERFLOW | FP_INVALID); \
}

#define generate_divbyzero_error() fp_raise_xcp(FP_DIV_BY_ZERO)
#define generate_overflow_error() fp_raise_xcp(FP_OVERFLOW)

#else

#define NO_FLOATING_POINT_SUPPORT 
#define UFUNC_CHECK_STATUS(ret) { \
    ret = 0;							     \
  }

#endif

/* These should really be altered to just set the corresponding bit
   in the floating point status flag.  Need to figure out how to do that
   on all the platforms...
*/

#if !defined(generate_divbyzero_error)
static int numeric_zero2 = 0;
static void generate_divbyzero_error(void) {
	double dummy;
	dummy = 1./numeric_zero2;
        if (dummy) /* to prevent optimizer from eliminating expression */
	   return;
	else /* should never be called */
	   numeric_zero2 += 1;
	return;	
}
#endif

#if !defined(generate_overflow_error)
static double numeric_two = 2.0;
static void generate_overflow_error(void) {
	double dummy;
	dummy = pow(numeric_two,1000);
        if (dummy)
           return;
        else
           numeric_two += 0.1;
        return;
	return;
}
#endif


#ifdef PY_ARRAY_TYPES_PREFIX
#  undef CAT
#  undef CAT2
#  undef NS
#  undef inpt
#endif


#ifdef __cplusplus
}
#endif
#endif /* !Py_UFUNCOBJECT_H */
