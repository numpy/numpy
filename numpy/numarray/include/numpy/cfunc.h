#if !defined(__cfunc__)
#define __cfunc__ 1

typedef PyObject *(*CFUNCasPyValue)(void *);
typedef int (*UFUNC)(long, long, long, void **, long*);
/* typedef void (*CFUNC_2ARG)(long, void *, void *); */
/* typedef void (*CFUNC_3ARG)(long, void *, void *, void *); */
typedef int (*CFUNCfromPyValue)(PyObject *, void *);
typedef int (*CFUNC_STRIDE_CONV_FUNC)(long, long, maybelong *,
	      void *, long, maybelong*, void *, long, maybelong *);

typedef int (*CFUNC_STRIDED_FUNC)(PyObject *, long, PyArrayObject **,
				  char **data);

#define MAXARRAYS 16

typedef enum {
	CFUNC_UFUNC,
	CFUNC_STRIDING,
	CFUNC_NSTRIDING,
	CFUNC_AS_PY_VALUE,
	CFUNC_FROM_PY_VALUE
} eCfuncType;

typedef struct {
	char *name;
        void *fptr;            /* Pointer to "un-wrapped" c function */
	eCfuncType type;       /* UFUNC, STRIDING, AsPyValue, FromPyValue */
	Bool chkself;          /* CFUNC does own alignment/bounds checking */
	Bool align;            /* CFUNC requires aligned buffer pointers */
	Int8 wantIn, wantOut;  /* required input/output arg counts. */
	Int8 sizes[MAXARRAYS]; /* array of align/itemsizes. */
	Int8 iters[MAXARRAYS]; /*  array of element counts. 0 --> niter. */
} CfuncDescriptor;

typedef struct {
    PyObject_HEAD
    CfuncDescriptor descr;
} CfuncObject;

#define SELF_CHECKED_CFUNC_DESCR(name, type)                                 \
   static CfuncDescriptor name##_descr = { #name, (void *) name, type, 1 }

#define CHECK_ALIGN 1

#define CFUNC_DESCR(name, type, align, iargs, oargs, s1, s2, s3, i1, i2, i3)  \
  static CfuncDescriptor name##_descr =                                       \
    { #name, (void *)name, type, 0, align, iargs, oargs, {s1, s2, s3}, {i1, i2, i3} }

#define UFUNC_DESCR1(name, s1)                                                \
   CFUNC_DESCR(name, CFUNC_UFUNC, CHECK_ALIGN, 0, 1, s1, 0, 0, 0, 0, 0)

#define UFUNC_DESCR2(name, s1, s2)                                            \
   CFUNC_DESCR(name, CFUNC_UFUNC, CHECK_ALIGN, 1, 1, s1, s2, 0, 0, 0, 0)

#define UFUNC_DESCR3(name, s1, s2, s3)                                        \
   CFUNC_DESCR(name, CFUNC_UFUNC, CHECK_ALIGN, 2, 1, s1, s2, s3, 0, 0, 0)

#define UFUNC_DESCR3sv(name, s1, s2, s3)                                      \
   CFUNC_DESCR(name, CFUNC_UFUNC, CHECK_ALIGN, 2, 1, s1, s2, s3, 1, 0, 0)

#define UFUNC_DESCR3vs(name, s1, s2, s3)                                      \
   CFUNC_DESCR(name, CFUNC_UFUNC, CHECK_ALIGN, 2, 1, s1, s2, s3, 0, 1, 0)

#define STRIDING_DESCR2(name, align, s1, s2)                                  \
   CFUNC_DESCR(name, CFUNC_STRIDING, align, 1, 1, s1, s2, 0, 0, 0, 0)

#define NSTRIDING_DESCR1(name)                                                \
   CFUNC_DESCR(name, CFUNC_NSTRIDING, 0, 0, 1, 0, 0, 0, 0, 0, 0)

#define NSTRIDING_DESCR2(name)                                                \
   CFUNC_DESCR(name, CFUNC_NSTRIDING, 0, 1, 1, 0, 0, 0, 0, 0, 0)

#define NSTRIDING_DESCR3(name)                                                \
   CFUNC_DESCR(name, CFUNC_NSTRIDING, 0, 2, 1, 0, 0, 0, 0, 0, 0)

#endif
