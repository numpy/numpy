/* struct module -- pack values into and (out of) strings */

/* New version supporting byte order, alignment and size options,
  character strings, and unsigned numbers */

#define PY_SSIZE_T_CLEAN

#include "Python.h"
#include "structseq.h"
#include "structmember.h"
#include <ctype.h>

static PyTypeObject PyStructType;
static PyTypeObject PyFieldTupleType;

/* If PY_STRUCT_OVERFLOW_MASKING is defined, the struct module will wrap all input
  numbers for explicit endians such that they fit in the given type, much
  like explicit casting in C. A warning will be raised if the number did
  not originally fit within the range of the requested type. If it is
  not defined, then all range errors and overflow will be struct.error
  exceptions. */

#define PY_STRUCT_OVERFLOW_MASKING 1

#ifdef PY_STRUCT_OVERFLOW_MASKING
static PyObject *pylong_ulong_mask = NULL;
static PyObject *pyint_zero = NULL;
#endif

/* If PY_STRUCT_FLOAT_COERCE is defined, the struct module will allow float
  arguments for integer formats with a warning for backwards
  compatibility. */

#define PY_STRUCT_FLOAT_COERCE 1

#ifdef PY_STRUCT_FLOAT_COERCE
#define FLOAT_COERCE "integer argument expected, got float"
#endif

/* Native is default
  If F_LE or F_BE is set, then alignment
  information should not be used.
*/
#define F_LE      0x0100
#define F_BE      0x0200
#define F_BIT     0x0400
#define F_PTR     0x0800
#define F_CMPLX   0x1000
#define F_STRUCT  0x2000
#define F_ARRAY   0x4000
#define F_FUNCPTR 0x8000
#define F_BITMASK 0x00FF  /* storage in FLAG for bit-offset */

/* Forward declarations */
typedef struct _formatdef formatdef;
typedef struct _formatcode formatcode;

#define F_IS_NATIVE(c) ((((const formatcode *)(c))->flags & (F_LE||F_BE))==0)
#define F_IS_LE(c) ((((const formatcode *)(c))->flags & F_LE) == F_LE)
#define F_IS_BE(c) ((((const formatcode *)(c))->flags & F_BE) == F_BE)

/* The translation function for each format character is table driven */
struct _formatdef {
       char format;
       Py_ssize_t size;
       Py_ssize_t stdsize;
       Py_ssize_t alignment;
       PyObject* (*unpack)(const char *,
                           const formatdef *,
                           const formatcode *);
       int (*pack)(char *, PyObject *,
                   const formatdef *,
                   const formatcode *);
};

struct _formatcode {
       const formatdef *fmtdef;  /* format information */
       Py_ssize_t offset;                /* offset into the data area
                                            relative to the start of
                                            this structure */
       Py_ssize_t size;                  /* size of the data implied */
       char *name;                       /* field name of this code */
       int flags;                        /* ISBIT, ISPTR, ISSTRUCT,
                                            ISCMPLX, ISFUNCPTR, ISARRAY */
       void *ptr;

              /* several possibilities:

               pointer to a format code     pointer, complex, array
               pointer to a StructObject    struct, function pointer

               in case of function pointer, the structure represents
               the function arguments (last one represents return type).
               */
};

/* This format code has extra fields for shape information  but it is
  binary compatible with the formatcode.
*/

typedef struct _arraycode {
       formatcode code;
       int ndims;
       Py_ssize_t *shape;
} arraycode;

/* Struct object interface */

typedef struct _structobject {
       PyObject_HEAD
       Py_ssize_t s_size;     /* The byte-size the data-string */
       Py_ssize_t s_len;      /* Number of codes */
       formatcode *s_codes;   /* Pointer to array of codes */
       PyObject *s_format;    /* Python format string */
       PyObject *weakreflist; /* List of weak references */
} PyStructObject;


typedef struct _fieldtupleobject {
       PyTupleObject tup;
       char **names;            /* array of strings with names of the fields */
} PyFieldTupleObject;

#define PyStruct_Check(op) PyObject_TypeCheck(op, &PyStructType)
#define PyStruct_CheckExact(op) ((op)->ob_type == &PyStructType)

#define PyFieldTuple_Check(op) PyObject_TypeCheck(op, &PyFieldTuple_Check)
#define PyFieldTuple_CheckExact(op) ((op)->ob_type == &PyFieldTuple_Check)


/* Exception */

static PyObject *StructError;


/* Define various structs to figure out the alignments of types */


typedef struct { char c; short x; } st_short;
typedef struct { char c; int x; } st_int;
typedef struct { char c; long x; } st_long;
typedef struct { char c; float x; } st_float;
typedef struct { char c; double x; } st_double;
typedef struct { char c; long double x;} st_longdouble;
typedef struct { char c; void *x; } st_void_ptr;

#define SHORT_ALIGN (sizeof(st_short) - sizeof(short))
#define INT_ALIGN (sizeof(st_int) - sizeof(int))
#define LONG_ALIGN (sizeof(st_long) - sizeof(long))
#define FLOAT_ALIGN (sizeof(st_float) - sizeof(float))
#define DOUBLE_ALIGN (sizeof(st_double) - sizeof(double))
#define VOID_PTR_ALIGN (sizeof(st_void_ptr) - sizeof(void *))

/* We can't support q and Q in native mode unless the compiler does;
  in std mode, they're 8 bytes on all platforms. */
#ifdef HAVE_LONG_LONG
typedef struct { char c; PY_LONG_LONG x; } s_long_long;
#define LONG_LONG_ALIGN (sizeof(s_long_long) - sizeof(PY_LONG_LONG))
#endif

/* Long double in native mode is platform dependent.
  in std mode they're 16 bytes on all platforms
 */
#ifdef HAVE_LONG_DOUBLE
typedef struct { char c; PY_LONG_LONG x; } s_long_double;
#define LONG_DOUBLE_ALIGN (sizeof(s_long_double) - sizeof(long double))
#endif

#ifdef HAVE_C99_BOOL
#define BOOL_TYPE _Bool
typedef struct { char c; _Bool x; } s_bool;
#define BOOL_ALIGN (sizeof(s_bool) - sizeof(BOOL_TYPE))
#else
#define BOOL_TYPE char
#define BOOL_ALIGN 0
#endif

#define STRINGIFY(x)    #x

#ifdef __powerc
#pragma options align=reset
#endif

/* Helper to get a PyLongObject by hook or by crook.  Caller should decref. */

static PyObject *
get_pylong(PyObject *v)
{
       PyNumberMethods *m;

       assert(v != NULL);
       if (PyLong_Check(v)) {
               Py_INCREF(v);
               return v;
       }
       m = v->ob_type->tp_as_number;
       if (m != NULL && m->nb_long != NULL) {
               v = m->nb_long(v);
               if (v == NULL)
                       return NULL;
               if (PyLong_Check(v))
                       return v;
               Py_DECREF(v);
       }
       PyErr_SetString(StructError,
                       "cannot convert argument to long");
       return NULL;
}

/* Helper routine to get a Python integer and raise the appropriate error
  if it isn't one */

static int
get_long(PyObject *v, long *p)
{
    long x = PyInt_AsLong(v);
    if (x == -1 && PyErr_Occurred()) {
#ifdef PY_STRUCT_FLOAT_COERCE
        if (PyFloat_Check(v)) {
            PyObject *o;
            int res;
            PyErr_Clear();
            if (PyErr_WarnEx(PyExc_DeprecationWarning, FLOAT_COERCE, 2) < 0)
                return -1;
            o = PyNumber_Int(v);
            if (o == NULL)
                return -1;
            res = get_long(o, p);
            Py_DECREF(o);
            return res;
        }
#endif
        if (PyErr_ExceptionMatches(PyExc_TypeError))
            PyErr_SetString(StructError, "required argument is not an integer");
        return -1;
    }
    *p = x;
    return 0;
}


/* Same, but handling unsigned long */

static int
get_ulong(PyObject *v, unsigned long *p)
{
       if (PyLong_Check(v)) {
               unsigned long x = PyLong_AsUnsignedLong(v);
               if (x == (unsigned long)(-1) && PyErr_Occurred())
                       return -1;
               *p = x;
               return 0;
       }
       if (get_long(v, (long *)p) < 0)
               return -1;
       if (((long)*p) < 0) {
               PyErr_SetString(StructError,
                               "unsigned argument is < 0");
               return -1;
       }
       return 0;
}

#ifdef HAVE_LONG_LONG

/* Same, but handling native long long. */

static int
get_longlong(PyObject *v, PY_LONG_LONG *p)
{
       PY_LONG_LONG x;

       v = get_pylong(v);
       if (v == NULL)
               return -1;
       assert(PyLong_Check(v));
       x = PyLong_AsLongLong(v);
       Py_DECREF(v);
       if (x == (PY_LONG_LONG)-1 && PyErr_Occurred())
               return -1;
       *p = x;
       return 0;
}

/* Same, but handling native unsigned long long. */

static int
get_ulonglong(PyObject *v, unsigned PY_LONG_LONG *p)
{
       unsigned PY_LONG_LONG x;

       v = get_pylong(v);
       if (v == NULL)
               return -1;
       assert(PyLong_Check(v));
       x = PyLong_AsUnsignedLongLong(v);
       Py_DECREF(v);
       if (x == (unsigned PY_LONG_LONG)-1 && PyErr_Occurred())
               return -1;
       *p = x;
       return 0;
}

#endif

#ifdef PY_STRUCT_OVERFLOW_MASKING

/* Helper routine to get a Python integer and raise the appropriate error
  if it isn't one */

#define INT_OVERFLOW "struct integer overflow masking is deprecated"

static int
get_wrapped_long(PyObject *v, long *p)
{
       if (get_long(v, p) < 0) {
               if (PyLong_Check(v) &&
                   PyErr_ExceptionMatches(PyExc_OverflowError)) {
                       PyObject *wrapped;
                       long x;
                       PyErr_Clear();
#ifdef PY_STRUCT_FLOAT_COERCE
                       if (PyFloat_Check(v)) {
                               PyObject *o;
                               int res;
                               PyErr_Clear();
                               if (PyErr_WarnEx(PyExc_DeprecationWarning, FLOAT_COERCE, 2) < 0)
                                       return -1;
                               o = PyNumber_Int(v);
                               if (o == NULL)
                                       return -1;
                               res = get_wrapped_long(o, p);
                               Py_DECREF(o);
                               return res;
                       }
#endif
                       if (PyErr_WarnEx(PyExc_DeprecationWarning, INT_OVERFLOW, 2) < 0)
                               return -1;
                       wrapped = PyNumber_And(v, pylong_ulong_mask);
                       if (wrapped == NULL)
                               return -1;
                       x = (long)PyLong_AsUnsignedLong(wrapped);
                       Py_DECREF(wrapped);
                       if (x == -1 && PyErr_Occurred())
                               return -1;
                       *p = x;
               } else {
                       return -1;
               }
       }
       return 0;
}

static int
get_wrapped_ulong(PyObject *v, unsigned long *p)
{
       long x = (long)PyLong_AsUnsignedLong(v);
       if (x == -1 && PyErr_Occurred()) {
               PyObject *wrapped;
               PyErr_Clear();
#ifdef PY_STRUCT_FLOAT_COERCE
               if (PyFloat_Check(v)) {
                       PyObject *o;
                       int res;
                       PyErr_Clear();
                       if (PyErr_WarnEx(PyExc_DeprecationWarning, FLOAT_COERCE, 2) < 0)
                               return -1;
                       o = PyNumber_Int(v);
                       if (o == NULL)
                               return -1;
                       res = get_wrapped_ulong(o, p);
                       Py_DECREF(o);
                       return res;
               }
#endif
               wrapped = PyNumber_And(v, pylong_ulong_mask);
               if (wrapped == NULL)
                       return -1;
               if (PyErr_WarnEx(PyExc_DeprecationWarning, INT_OVERFLOW, 2) < 0) {
                       Py_DECREF(wrapped);
                       return -1;
               }
               x = (long)PyLong_AsUnsignedLong(wrapped);
               Py_DECREF(wrapped);
               if (x == -1 && PyErr_Occurred())
                       return -1;
       }
       *p = (unsigned long)x;
       return 0;
}

#define RANGE_ERROR(x, f, flag, mask) \
       do { \
               if (_range_error(f, flag) < 0) \
                       return -1; \
               else \
                       (x) &= (mask); \
       } while (0)

#else

#define get_wrapped_long get_long
#define get_wrapped_ulong get_ulong
#define RANGE_ERROR(x, f, flag, mask) return _range_error(f, flag)

#endif

/* Floating point helpers */

static PyObject *
unpack_float_internal(const char *p,  /* start of 4-byte string */
            int le)         /* true for little-endian, false for big-endian */
{
       double x;

       x = _PyFloat_Unpack4((unsigned char *)p, le);
       if (x == -1.0 && PyErr_Occurred())
               return NULL;
       return PyFloat_FromDouble(x);
}

static PyObject *
unpack_double_internal(const char *p,  /* start of 8-byte string */
             int le)         /* true for little-endian, false for big-endian */
{
       double x;

       x = _PyFloat_Unpack8((unsigned char *)p, le);
       if (x == -1.0 && PyErr_Occurred())
               return NULL;
       return PyFloat_FromDouble(x);
}

/* Helper to format the range error exceptions */
static int
_range_error(const formatdef *f, int is_unsigned)
{
       /* ulargest is the largest unsigned value with f->size bytes.
        * Note that the simpler:
        *     ((size_t)1 << (f->size * 8)) - 1
        * doesn't work when f->size == sizeof(size_t) because C doesn't
        * define what happens when a left shift count is >= the number of
        * bits in the integer being shifted; e.g., on some boxes it doesn't
        * shift at all when they're equal.
        */
       const size_t ulargest = (size_t)-1 >> ((SIZEOF_SIZE_T - f->size)*8);
       assert(f->size >= 1 && f->size <= SIZEOF_SIZE_T);
       if (is_unsigned)
               PyErr_Format(StructError,
                       "'%c' format requires 0 <= number <= %zu",
                       f->format,
                       ulargest);
       else {
               const Py_ssize_t largest = (Py_ssize_t)(ulargest >> 1);
               PyErr_Format(StructError,
                       "'%c' format requires %zd <= number <= %zd",
                       f->format,
                       ~ largest,
                       largest);
       }
#ifdef PY_STRUCT_OVERFLOW_MASKING
       {
               PyObject *ptype, *pvalue, *ptraceback;
               PyObject *msg;
               int rval;
               PyErr_Fetch(&ptype, &pvalue, &ptraceback);
               assert(pvalue != NULL);
               msg = PyObject_Str(pvalue);
               Py_XDECREF(ptype);
               Py_XDECREF(pvalue);
               Py_XDECREF(ptraceback);
               if (msg == NULL)
                       return -1;
               rval = PyErr_WarnEx(PyExc_DeprecationWarning,
                                   PyString_AS_STRING(msg), 2);
               Py_DECREF(msg);
               if (rval == 0)
                       return 0;
       }
#endif
       return -1;
}

static PyObject *
unpack_bit(const char *p, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "unpack_bit is not implemented.");
    return NULL;
}

static int
pack_bit(char *p, PyObject *obj, const struct _formatdef *f,
     const struct _formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "pack_bit is not implemented.");
    return -1;
}


static PyObject *
unpack_byte(const char *p, const formatdef *f, const formatcode *c)
{
       return PyInt_FromLong((long) *(signed char *)p);
}

static int
pack_byte(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{
       long x;
       if (get_long(v, &x) < 0)
               return -1;
       if (x < -128 || x > 127){
               PyErr_SetString(StructError,
                               "byte format requires -128 <= number <= 127");
               return -1;
       }
       *p = (char)x;
       return 0;
}

static PyObject *
unpack_ubyte(const char *p, const formatdef *f, const formatcode *c)
{
       return PyInt_FromLong((long) *(unsigned char *)p);
}


static int
pack_ubyte(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{
       long x;
       if (get_long(v, &x) < 0)
               return -1;
       if (x < 0 || x > 255){
               PyErr_SetString(StructError,
                               "ubyte format requires 0 <= number <= 255");
               return -1;
       }
       *p = (char)x;
       return 0;
}


static PyObject *
unpack_char(const char *p, const formatdef *f, const formatcode *c)
{
       return PyString_FromStringAndSize(p, 1);
}


static int
pack_char(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{
       if (!PyString_Check(v) || PyString_Size(v) != 1) {
               PyErr_SetString(StructError,
                               "char format require string of length 1");
               return -1;
       }
       *p = *PyString_AsString(v);
       return 0;
}

/* XXX Function Stub */
static PyObject *
unpack_ucs2(const char *p, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "unpack_uc2 is not implemented.");
    return NULL;
}

/* XXX Function Stub */
static int
pack_ucs2(char *p, PyObject *obj, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "pack_uc2 is not implemented.");
    return -1;
}

/* XXX Function Stub */
static PyObject *
unpack_ucs4(const char *p, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError, 
                    "unpack_ucs4 is not implemented.");
    return NULL;
}

/* XXX Function Stub */
static int
pack_ucs4(char *p, PyObject *obj, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "pack_ucs4 is not implemented.");
    return -1;
}


/***************

HELPER ROUTINES for big-endian un-packing

******************/

static PyObject *
bu_int(const char *p, const formatdef *f)
{
       long x = 0;
       Py_ssize_t i = f->stdsize;
       const unsigned char *bytes = (const unsigned char *)p;
       do {
               x = (x<<8) | *bytes++;
       } while (--i > 0);
       /* Extend the sign bit. */
       if (SIZEOF_LONG > f->size)
               x |= -(x & (1L << ((8 * f->size) - 1)));
       return PyInt_FromLong(x);
}

static PyObject *
bu_uint(const char *p, const formatdef *f)
{
       unsigned long x = 0;
       Py_ssize_t i = f->stdsize;
       const unsigned char *bytes = (const unsigned char *)p;
       do {
               x = (x<<8) | *bytes++;
       } while (--i > 0);
       if (x <= LONG_MAX)
               return PyInt_FromLong((long)x);
       return PyLong_FromUnsignedLong(x);
}

static PyObject *
bu_longlong(const char *p, const formatdef *f)
{
#ifdef HAVE_LONG_LONG
       PY_LONG_LONG x = 0;
       Py_ssize_t i = f->stdsize;
       const unsigned char *bytes = (const unsigned char *)p;
       do {
               x = (x<<8) | *bytes++;
       } while (--i > 0);
       /* Extend the sign bit. */
       if (SIZEOF_LONG_LONG > f->size)
               x |= -(x & (1L << ((8 * f->size) - 1)));
       if (x >= LONG_MIN && x <= LONG_MAX)
               return PyInt_FromLong(Py_SAFE_DOWNCAST(x, PY_LONG_LONG, long));
       return PyLong_FromLongLong(x);
#else
       return _PyLong_FromByteArray((const unsigned char *)p,
                                     8,
                                     0, /* little-endian */
                                     1  /* signed */);
#endif
}

static PyObject *
bu_ulonglong(const char *p, const formatdef *f)
{
#ifdef HAVE_LONG_LONG
       unsigned PY_LONG_LONG x = 0;
       Py_ssize_t i = f->stdsize;
       const unsigned char *bytes = (const unsigned char *)p;
       do {
               x = (x<<8) | *bytes++;
       } while (--i > 0);
       if (x <= LONG_MAX)
               return PyInt_FromLong(Py_SAFE_DOWNCAST(x, unsigned PY_LONG_LONG, long));
       return PyLong_FromUnsignedLongLong(x);
#else
       return _PyLong_FromByteArray((const unsigned char *)p,
                                     8,
                                     0, /* little-endian */
                                     0  /* signed */);
#endif
}

static PyObject *
bu_float(const char *p, const formatdef *f)
{
       return unpack_float_internal(p, 0);
}

static PyObject *
bu_double(const char *p, const formatdef *f)
{
       return unpack_double_internal(p, 0);
}


/***************

HELPER ROUTINES for little-endian un-packing

******************/


static PyObject *
lu_int(const char *p, const formatdef *f)
{
       long x = 0;
       Py_ssize_t i = f->stdsize;
       const unsigned char *bytes = (const unsigned char *)p;
       do {
               x = (x<<8) | bytes[--i];
       } while (i > 0);
       /* Extend the sign bit. */
       if (SIZEOF_LONG > f->size)
               x |= -(x & (1L << ((8 * f->size) - 1)));
       return PyInt_FromLong(x);
}

static PyObject *
lu_uint(const char *p, const formatdef *f)
{
       unsigned long x = 0;
       Py_ssize_t i = f->stdsize;
       const unsigned char *bytes = (const unsigned char *)p;
       do {
               x = (x<<8) | bytes[--i];
       } while (i > 0);
       if (x <= LONG_MAX)
               return PyInt_FromLong((long)x);
       return PyLong_FromUnsignedLong((long)x);
}

static PyObject *
lu_longlong(const char *p, const formatdef *f)
{
#ifdef HAVE_LONG_LONG
       PY_LONG_LONG x = 0;
       Py_ssize_t i = f->stdsize;
       const unsigned char *bytes = (const unsigned char *)p;
       do {
               x = (x<<8) | bytes[--i];
       } while (i > 0);
       /* Extend the sign bit. */
       if (SIZEOF_LONG_LONG > f->size)
               x |= -(x & (1L << ((8 * f->size) - 1)));
       if (x >= LONG_MIN && x <= LONG_MAX)
               return PyInt_FromLong(Py_SAFE_DOWNCAST(x, PY_LONG_LONG, long));
       return PyLong_FromLongLong(x);
#else
       return _PyLong_FromByteArray((const unsigned char *)p,
                                     8,
                                     1, /* little-endian */
                                     1  /* signed */);
#endif
}

static PyObject *
lu_ulonglong(const char *p, const formatdef *f)
{
#ifdef HAVE_LONG_LONG
       unsigned PY_LONG_LONG x = 0;
       Py_ssize_t i = f->stdsize;
       const unsigned char *bytes = (const unsigned char *)p;
       do {
               x = (x<<8) | bytes[--i];
       } while (i > 0);
       if (x <= LONG_MAX)
               return PyInt_FromLong(Py_SAFE_DOWNCAST(x, unsigned PY_LONG_LONG, long));
       return PyLong_FromUnsignedLongLong(x);
#else
       return _PyLong_FromByteArray((const unsigned char *)p,
                                     8,
                                     1, /* little-endian */
                                     0  /* signed */);
#endif
}

static PyObject *
lu_float(const char *p, const formatdef *f)
{
       return unpack_float_internal(p, 1);
}

static PyObject *
lu_double(const char *p, const formatdef *f)
{
       return unpack_double_internal(p, 1);
}


/***************

HELPER ROUTINES for big-endian packing

******************/


static int
bp_int(char *p, PyObject *v, const formatdef *f)
{
       long x;
       Py_ssize_t i;
       if (get_wrapped_long(v, &x) < 0)
               return -1;
       i = f->stdsize;
       if (i != SIZEOF_LONG) {
               if ((i == 2) && (x < -32768 || x > 32767))
                       RANGE_ERROR(x, f, 0, 0xffffL);
#if (SIZEOF_LONG != 4)
               else if ((i == 4) && (x < -2147483648L || x > 2147483647L))
                       RANGE_ERROR(x, f, 0, 0xffffffffL);
#endif
#ifdef PY_STRUCT_OVERFLOW_MASKING
               else if ((i == 1) && (x < -128 || x > 127))
                       RANGE_ERROR(x, f, 0, 0xffL);
#endif
       }
       do {
               p[--i] = (char)x;
               x >>= 8;
       } while (i > 0);
       return 0;
}

static int
bp_uint(char *p, PyObject *v, const formatdef *f)
{
       unsigned long x;
       Py_ssize_t i;
       if (get_wrapped_ulong(v, &x) < 0)
               return -1;
       i = f->stdsize;
       if (i != SIZEOF_LONG) {
               unsigned long maxint = 1;
               maxint <<= (unsigned long)(i * 8);
               if (x >= maxint)
                       RANGE_ERROR(x, f, 1, maxint - 1);
       }
       do {
               p[--i] = (char)x;
               x >>= 8;
       } while (i > 0);
       return 0;
}

static int
bp_longlong(char *p, PyObject *v, const formatdef *f)
{
       int res;
       v = get_pylong(v);
       if (v == NULL)
               return -1;
       res = _PyLong_AsByteArray((PyLongObject *)v,
                                 (unsigned char *)p,
                                 8,
                                 0, /* little_endian */
                                 1  /* signed */);
       Py_DECREF(v);
       return res;
}

static int
bp_ulonglong(char *p, PyObject *v, const formatdef *f)
{
       int res;
       v = get_pylong(v);
       if (v == NULL)
               return -1;
       res = _PyLong_AsByteArray((PyLongObject *)v,
                                 (unsigned char *)p,
                                 8,
                                 0, /* little_endian */
                                 0  /* signed */);
       Py_DECREF(v);
       return res;
}

static int
bp_float(char *p, PyObject *v, const formatdef *f)
{
       double x = PyFloat_AsDouble(v);
       if (x == -1 && PyErr_Occurred()) {
               PyErr_SetString(StructError,
                               "required argument is not a float");
               return -1;
       }
       return _PyFloat_Pack4(x, (unsigned char *)p, 0);
}

static int
bp_double(char *p, PyObject *v, const formatdef *f)
{
       double x = PyFloat_AsDouble(v);
       if (x == -1 && PyErr_Occurred()) {
               PyErr_SetString(StructError,
                               "required argument is not a float");
               return -1;
       }
       return _PyFloat_Pack8(x, (unsigned char *)p, 0);
}


/***************

HELPER ROUTINES for little-endian un-packing

******************/


static int
lp_int(char *p, PyObject *v, const formatdef *f)
{
       long x;
       Py_ssize_t i;
       if (get_wrapped_long(v, &x) < 0)
               return -1;
       i = f->stdsize;
       if (i != SIZEOF_LONG) {
               if ((i == 2) && (x < -32768 || x > 32767))
                       RANGE_ERROR(x, f, 0, 0xffffL);
#if (SIZEOF_LONG != 4)
               else if ((i == 4) && (x < -2147483648L || x > 2147483647L))
                       RANGE_ERROR(x, f, 0, 0xffffffffL);
#endif
#ifdef PY_STRUCT_OVERFLOW_MASKING
               else if ((i == 1) && (x < -128 || x > 127))
                       RANGE_ERROR(x, f, 0, 0xffL);
#endif
       }
       do {
               *p++ = (char)x;
               x >>= 8;
       } while (--i > 0);
       return 0;
}

static int
lp_uint(char *p, PyObject *v, const formatdef *f)
{
       unsigned long x;
       Py_ssize_t i;
       if (get_wrapped_ulong(v, &x) < 0)
               return -1;
       i = f->stdsize;
       if (i != SIZEOF_LONG) {
               unsigned long maxint = 1;
               maxint <<= (unsigned long)(i * 8);
               if (x >= maxint)
                       RANGE_ERROR(x, f, 1, maxint - 1);
       }
       do {
               *p++ = (char)x;
               x >>= 8;
       } while (--i > 0);
       return 0;
}

static int
lp_longlong(char *p, PyObject *v, const formatdef *f)
{
       int res;
       v = get_pylong(v);
       if (v == NULL)
               return -1;
       res = _PyLong_AsByteArray((PyLongObject*)v,
                                 (unsigned char *)p,
                                 8,
                                 1, /* little_endian */
                                 1  /* signed */);
       Py_DECREF(v);
       return res;
}

static int
lp_ulonglong(char *p, PyObject *v, const formatdef *f)
{
       int res;
       v = get_pylong(v);
       if (v == NULL)
               return -1;
       res = _PyLong_AsByteArray((PyLongObject*)v,
                                 (unsigned char *)p,
                                 8,
                                 1, /* little_endian */
                                 0  /* signed */);
       Py_DECREF(v);
       return res;
}

static int
lp_float(char *p, PyObject *v, const formatdef *f)
{
       double x = PyFloat_AsDouble(v);
       if (x == -1 && PyErr_Occurred()) {
               PyErr_SetString(StructError,
                               "required argument is not a float");
               return -1;
       }
       return _PyFloat_Pack4(x, (unsigned char *)p, 1);
}

static int
lp_double(char *p, PyObject *v, const formatdef *f)
{
       double x = PyFloat_AsDouble(v);
       if (x == -1 && PyErr_Occurred()) {
               PyErr_SetString(StructError,
                               "required argument is not a float");
               return -1;
       }
       return _PyFloat_Pack8(x, (unsigned char *)p, 1);
}



static PyObject *
unpack_short(const char *p, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
               short x;
               memcpy((char *)&x, p, sizeof x);
               return PyInt_FromLong((long)x);
       }
       else if F_IS_LE(c) {
               return lu_int(p, f);
       }
       else {
               return bu_int(p, f);
       }
}


static int
pack_short(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{

       if F_IS_NATIVE(c) {
               long x;
               short y;
               if (get_long(v, &x) < 0)
                       return -1;
               if (x < SHRT_MIN || x > SHRT_MAX){
                       PyErr_SetString(StructError,
                                       "short format requires " STRINGIFY(SHRT_MIN)
                                       " <= number <= " STRINGIFY(SHRT_MAX));
                       return -1;
               }
               y = (short)x;
               memcpy(p, (char *)&y, sizeof y);
               return 0;
       }
       else if F_IS_LE(c) {
               return lp_int(p, v, f);
       }
       else {
               return bp_int(p, v, f);
       }
}


static PyObject *
unpack_ushort(const char *p, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
               unsigned short x;
               memcpy((char *)&x, p, sizeof x);
               return PyInt_FromLong((long)x);
       }
       else if F_IS_LE(c) {
               return lu_uint(p, f);
       }
       else {
               return bu_uint(p, f);
       }
}


static int
pack_ushort(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{

       if F_IS_NATIVE(c) {
               long x;
               unsigned short y;
               if (get_long(v, &x) < 0)
                       return -1;
               if (x < 0 || x > USHRT_MAX){
                       PyErr_SetString(StructError,
                                       "short format requires "
                                       "0 <= number <= " STRINGIFY(USHRT_MAX));
                       return -1;
               }
               y = (unsigned short)x;
               memcpy(p, (char *)&y, sizeof y);
               return 0;
       }
       else if F_IS_LE(c) {
               return lp_uint(p, v, f);
       }
       else {
               return bp_uint(p, v, f);
       }
}

static PyObject *
unpack_int(const char *p, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
               int x;
               memcpy((char *)&x, p, sizeof x);
               return PyInt_FromLong((long)x);
       }
       else if F_IS_LE(c) {
               return lu_int(p, f);
       }
       else {
               return bu_int(p, f);
       }
}


static int
pack_int(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
               long x;
               int y;
               if (get_long(v, &x) < 0)
                       return -1;
               if (x < SHRT_MIN || x > SHRT_MAX){
                       PyErr_SetString(StructError,
                                       "int format requires " STRINGIFY(SHRT_MIN)
                                       " <= number <= " STRINGIFY(SHRT_MAX));
                       return -1;
               }
               y = (int)x;
               memcpy(p, (char *)&y, sizeof y);
               return 0;
       }
       else if F_IS_LE(c) {
               return lp_int(p, v, f);
       }
       else {
               return bp_int(p, v, f);
       }
}


static PyObject *
unpack_uint(const char *p, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
               unsigned int x;
               memcpy((char *)&x, p, sizeof x);
#if (SIZEOF_LONG > SIZEOF_INT)
               return PyInt_FromLong((long)x);
#else
               if (x <= ((unsigned int)LONG_MAX))
                       return PyInt_FromLong((long)x);
               return PyLong_FromUnsignedLong((unsigned long)x);
#endif
       }
       else if F_IS_LE(c) {
               return lu_uint(p, f);
       }
       else {
               return bu_uint(p, f);
       }
}


static int
pack_uint(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{

       if F_IS_NATIVE(c) {
               unsigned long x;
               unsigned int y;
               if (get_ulong(v, &x) < 0)
                       return _range_error(f, 1);
               y = (unsigned int)x;
#if (SIZEOF_LONG > SIZEOF_INT)
               if (x > ((unsigned long)UINT_MAX))
                       return _range_error(f, 1);
#endif
               memcpy(p, (char *)&y, sizeof y);
               return 0;
       }
       else if F_IS_LE(c) {
               return lp_uint(p, v, f);
       }
       else {
               return bp_uint(p, v, f);
       }
}


static PyObject *
unpack_long(const char *p, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
               long x;
               memcpy((char *)&x, p, sizeof x);
               return PyInt_FromLong(x);
       }
       else if F_IS_LE(c) {
               return lu_int(p, f);
       }
       else {
               return bu_int(p, f);
       }
}


static int
pack_long(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
               long x;
               if (get_long(v, &x) < 0)
                       return -1;
               memcpy(p, (char *)&x, sizeof x);
               return 0;
       }
       else if F_IS_LE(c) {
               return lp_int(p, v, f);
       }
       else {
               return bp_int(p, v, f);
       }
}


static PyObject *
unpack_ulong(const char *p, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
               unsigned long x;
               memcpy((char *)&x, p, sizeof x);
               if (x <= LONG_MAX)
                       return PyInt_FromLong((long) x);
               return PyLong_FromUnsignedLong(x);
       }
       else if F_IS_LE(c) {
               return lu_uint(p, f);
       }
       else {
               return bu_uint(p, f);
       }
}


static int
pack_ulong(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
               unsigned long x;
               if (get_ulong(v, &x) < 0)
                       return _range_error(f, 1);
               memcpy(p, (char *)&x, sizeof x);
               return 0;
       }
       else if F_IS_LE(c) {
               return lp_uint(p, v, f);
       }
       else {
               return bp_uint(p, v, f);
       }
}

#ifndef HAVE_LONG_LONG
static void
_no_longlong(void)
{
       PyErr_Format(StructError, "native long-long not supported.");
}
#endif



static PyObject *
unpack_longlong(const char *p, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
#ifdef HAVE_LONG_LONG
               PY_LONG_LONG x;
               memcpy((char *)&x, p, sizeof x);
               if (x >= LONG_MIN && x <= LONG_MAX)
                       return PyInt_FromLong(Py_SAFE_DOWNCAST(x, PY_LONG_LONG, long));
               return PyLong_FromLongLong(x);
#else
               _no_longlong();
               return NULL;
#endif
       }
       else if F_IS_LE(c) {
               return lu_int(p, f);
       }
       else {
               return bu_int(p, f);
       }
}


static int
pack_longlong(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
#ifdef HAVE_LONG_LONG
               PY_LONG_LONG x;
               if (get_longlong(v, &x) < 0)
                       return -1;
               memcpy(p, (char *)&x, sizeof x);
               return 0;
#else
               _no_longlong();
               return -1;
#endif
       }
       else if F_IS_LE(c) {
               return lp_longlong(p, v, f);
       }
       else {
               return bp_longlong(p, v, f);
       }
}


static PyObject *
unpack_ulonglong(const char *p, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
#ifdef HAVE_LONG_LONG
               unsigned PY_LONG_LONG x;
               memcpy((char *)&x, p, sizeof x);
               if (x <= LONG_MAX)
                       return PyInt_FromLong(Py_SAFE_DOWNCAST(x, unsigned PY_LONG_LONG, long));
               return PyLong_FromUnsignedLongLong(x);
#else
               _no_longlong();
               return NULL;
#endif
       }
       else if F_IS_LE(c) {
               return lu_ulonglong(p, f);
       }
       else {
               return bu_ulonglong(p, f);
       }
}


static int
pack_ulonglong(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{

       if F_IS_NATIVE(c) {
#ifdef HAVE_LONG_LONG
               unsigned PY_LONG_LONG x;
               if (get_ulonglong(v, &x) < 0)
                       return -1;
               memcpy(p, (char *)&x, sizeof x);
               return 0;
#else
               _no_longlong();
               return -1;
#endif
       }
       else if F_IS_LE(c) {
               return lp_ulonglong(p, v, f);
       }
       else {
               return bp_ulonglong(p, v, f);
       }
}


static PyObject *
unpack_bool(const char *p, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
               BOOL_TYPE x;
               memcpy((char *)&x, p, sizeof x);
	       return PyBool_FromLong(x != 0);
       }
       else {
               char x;
               memcpy((char *)&x, p, sizeof x);
	       return PyBool_FromLong(x != 0);
       }
}


static int
pack_bool(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
               BOOL_TYPE y;
               y = PyObject_IsTrue(v);
               memcpy(p, (char *)&y, sizeof y);
       }
       else {
               char y;
               y = PyObject_IsTrue(v);
               memcpy(p, (char *)&y, sizeof y);
       }
       return 0;

}


static PyObject *
unpack_float(const char *p, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
               float x;
               memcpy((char *)&x, p, sizeof x);
               return PyFloat_FromDouble((double)x);
       }
       else {
               return unpack_float_internal(p, F_IS_LE(c));
       }
}


static int
pack_float(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{
       double xi;

       xi = PyFloat_AsDouble(v);
       if (xi == -1 && PyErr_Occurred()) {
               PyErr_SetString(StructError,
                               "required argument is not a float");
               return -1;
       }
       if F_IS_NATIVE(c) {
               float x = (float) xi;
               memcpy(p, (char *)&x, sizeof x);
               return 0;
       }
       else {
               return _PyFloat_Pack4(xi, (unsigned char *)p, F_IS_LE(c));
       }
}


static PyObject *
unpack_double(const char *p, const formatdef *f, const formatcode *c)
{
       if F_IS_NATIVE(c) {
               double x;
               memcpy((char *)&x, p, sizeof x);
               return PyFloat_FromDouble((double)x);
       }
       else {
               return unpack_double_internal(p, F_IS_LE(c));
       }
}

static int
pack_double(char *p, PyObject *v, const formatdef *f, const formatcode *c)
{
       double x;

       x = PyFloat_AsDouble(v);
       if (x == -1 && PyErr_Occurred()) {
               PyErr_SetString(StructError,
                               "required argument is not a float");
               return -1;
       }
       if F_IS_NATIVE(c) {
               memcpy(p, (char *)&x, sizeof x);
               return 0;
       }
       else {
               return _PyFloat_Pack8(x, (unsigned char *)p, F_IS_LE(c));
       }
}

/* XXX Function Stub */
static PyObject *
unpack_longdouble(const char *p, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError, 
                    "unpack_longdouble is not implemented.");

    return NULL;
}

/* XXX Function Stub */
static int
pack_longdouble(char *p, PyObject *obj, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "pack_longdouble is not implemented.");
    return -1;
}

/* XXX Function Stub */
static PyObject *
unpack_cmplx(const char *p, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "unpack_cmplx is not implemented.");
    return NULL;
}

/* XXX Function Stub */
static int
pack_cmplx(char *p, PyObject *obj, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "pack_cmplx is not implemented.");
    return -1;
}

/* XXX Function Stub */
static PyObject *
unpack_gptr(const char *p, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "unpack_gptr is not implemented.");
    return NULL;
}

/* XXX Function Stub */
static int
pack_gptr(char *p, PyObject *obj, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "pack_gptr is not implemented.");
    return -1;
}

/* XXX Function Stub */
static PyObject *
unpack_array(const char *p, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "unpack_array is not implemented.");
    return NULL;
}

/* XXX Function Stub */
static int
pack_array(char *p, PyObject *obj, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "pack_array is not implemented.");
    return -1;
}

/* XXX Function Stub */
static PyObject *
unpack_void_ptr(const char *p, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "unpack_void_ptr is not implemented.");
    return NULL;
}

/* XXX Function Stub */
static int
pack_void_ptr(char *p, PyObject *obj, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "pack_void_ptr is not implemented.");
    return -1;
}

/* XXX Function Stub */
static PyObject *
unpack_object_ptr(const char *p, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "unpack_object_ptr is not implemented.");
    return NULL;
}

/* XXX Function Stub */
static int
pack_object_ptr(char *p, PyObject *obj, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "pack_object_ptr is not implemented.");
    return -1;
}

/* XXX Function Stub */
static PyObject *
unpack_struct(const char *p, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "unpack_struct is not implemented.");
    return NULL;
}

/* XXX Function Stub */
static int
pack_struct(char *p, PyObject *obj, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "pack_struct is not implemented.");
    return -1;
}

/* XXX Function Stub */
static PyObject *
unpack_funcptr(const char *p, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "unpack_funcPtr is not implemented.");
    return NULL;
}

/* XXX Function Stub */
static int
pack_funcptr(char *p, PyObject *obj, const formatdef *f, const formatcode *c)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "pack_funcptr is not implemented.");
    return -1;
}


/* There is only one un-packing and packing table,
  to allow switching of the endian-ness within a struct
  native, little-endian, and big-endian packing
  are handled by these functions based on the
  format-code flag.  Whether or not to use alignment is
  also handled by the flag.  Alignment is
  used only in native mode (no flag set).
*/
static formatdef general_table[] = {
    {'x',  sizeof(char),      1,  0,              NULL,              NULL},
    {'t',  0,                 0,  0,              unpack_bit,        pack_bit},
    {'b',  sizeof(char),      1,  0,              unpack_byte,       pack_byte},
    {'B',  sizeof(char),      1,  0,              unpack_ubyte,      pack_ubyte},
    {'c',  sizeof(char),      1,  0,              unpack_char,       pack_char},
    {'u',  2,                 2,  0,              unpack_ucs2,       pack_ucs2},
    {'w',  4,                 4,  0,              unpack_ucs4,       pack_ucs4},
    {'s',  sizeof(char),      1,  0,              NULL,              NULL},
    {'p',  sizeof(char),      1,  0,              NULL,              NULL},
    {'h',  sizeof(short),     2,  SHORT_ALIGN,    unpack_short,      pack_short},
    {'H',  sizeof(short),     2,  SHORT_ALIGN,    unpack_ushort,     pack_ushort},
    {'i',  sizeof(int),       4,  INT_ALIGN,      unpack_int,        pack_int},
    {'I',  sizeof(int),       4,  INT_ALIGN,      unpack_uint,       pack_uint},
    {'l',  sizeof(long),      4,  LONG_ALIGN,     unpack_long,       pack_long},
    {'L',  sizeof(long),      4,  LONG_ALIGN,     unpack_ulong,      pack_ulong},
    {'q',  0,                 8,  0,              unpack_longlong,   pack_longlong},
    {'Q',  0,                 8,  0,              unpack_ulonglong,  pack_ulonglong},
    {'?',  sizeof(BOOL_TYPE), 1,  BOOL_ALIGN,     unpack_bool,       pack_bool},
    {'f',  sizeof(float),     4,  FLOAT_ALIGN,    unpack_float,      pack_float},
    {'d',  sizeof(double),    8,  DOUBLE_ALIGN,   unpack_double,     pack_double},
    {'g',  0,                 16, 0,              unpack_longdouble, pack_longdouble},
    {'Z',  0,                 0,  0,              unpack_cmplx,      pack_cmplx},
    {'&',  sizeof(void*),     -1, VOID_PTR_ALIGN, unpack_gptr,       pack_gptr},
    {'(',  0,                 -1, 0,              unpack_array,      pack_array},
    {'P',  sizeof(void*),     -1, VOID_PTR_ALIGN, unpack_void_ptr,   pack_void_ptr},
    {'O',  sizeof(PyObject*), -1, VOID_PTR_ALIGN, unpack_object_ptr, pack_object_ptr},
    {'T',  0,                 -1, 0,              unpack_struct,     pack_struct},
    {'X',  sizeof(void*),     -1, VOID_PTR_ALIGN, unpack_funcptr,    pack_funcptr},
    {0}
};


/* Get the table entry for a format code */

static const formatdef *
getentry(int c, const formatdef *f)
{
    for (; f->format != '\0'; f++) {
        if (f->format == c) {
            return f;
        }
    }
    PyErr_SetString(StructError, "bad char in struct format");
    return NULL;
}


/* Align a size according to a format code */

static int
align(Py_ssize_t size, char c, const formatdef *e)
{
    if (e->format == c) {
        if (e->alignment) {
            size = ((size + e->alignment - 1)
                    / e->alignment)
                    * e->alignment;
        }
    }
    return size;
}


/* calculate the size of a format string */

static int
prepare_s(PyStructObject *self)
{
       const formatdef *table;
       const formatdef *entry;
       formatcode *codes;

       const char *ptr_fmt_str;
       const char *fmt;
       char c;
       Py_ssize_t size, len, num, itemsize, x;

       fmt = PyString_AS_STRING(self->s_format);

       table = general_table;

       ptr_fmt_str = fmt;
       size = 0;
       len = 0;
       while ((c = *ptr_fmt_str++) != '\0') {
               if (isspace(Py_CHARMASK(c)))
                       continue;
               if ('0' <= c && c <= '9') {
                       num = c - '0';
                       while ('0' <= (c = *ptr_fmt_str++) && c <= '9') {
                               x = num*10 + (c - '0');
                               if (x/10 != num) {
                                       PyErr_SetString(
                                               StructError,
                                               "overflow in item count");
                                       return -1;
                               }
                               num = x;
                       }
                       if (c == '\0')
                               break;
               }
               else
                       num = 1;

               entry = getentry(c, table);
               if (entry == NULL)
                       return -1;

               switch (c) {
                       case 's': /* fall through */
                       case 'p': len++; break;
                       case 'x': break;
                       default: len += num; break;
               }

               itemsize = entry->size;
               size = align(size, c, entry);
               x = num * itemsize;
               size += x;
               if (x/itemsize != num || size < 0) {
                       PyErr_SetString(StructError,
                                       "total struct size too long");
                       return -1;
               }
       }

       self->s_size = size;
       self->s_len = len;
       codes = PyMem_MALLOC((len + 1) * sizeof(formatcode));
       if (codes == NULL) {
               PyErr_NoMemory();
               return -1;
       }
       self->s_codes = codes;

       ptr_fmt_str = fmt;
       size = 0;
       while ((c = *ptr_fmt_str++) != '\0') {
               if (isspace(Py_CHARMASK(c)))
                       continue;
               if ('0' <= c && c <= '9') {
                       num = c - '0';
                       while ('0' <= (c = *ptr_fmt_str++) && c <= '9')
                               num = num*10 + (c - '0');
                       if (c == '\0')
                               break;
               }
               else
                       num = 1;

               entry = getentry(c, table);

               size = align(size, c, entry);
               if (c == 's' || c == 'p') {
                       codes->offset = size;
                       codes->size = num;
                       codes->fmtdef = entry;
                       codes++;
                       size += num;
               } else if (c == 'x') {
                       size += num;
               } else {
                       while (--num >= 0) {
                               codes->offset = size;
                               codes->size = entry->size;
                               codes->fmtdef = entry;
                               codes++;
                               size += entry->size;
                       }
               }
       }
       codes->fmtdef = NULL;
       codes->offset = size;
       codes->size = 0;

       return 0;
}

static PyObject *
s_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
       PyObject *self;

       assert(type != NULL && type->tp_alloc != NULL);

       self = type->tp_alloc(type, 0);
       if (self != NULL) {
               PyStructObject *s = (PyStructObject*)self;
               Py_INCREF(Py_None);
               s->s_format = Py_None;
               s->s_codes = NULL;
               s->s_size = -1;
               s->s_len = -1;
       }
       return self;
}

static int
s_init(PyObject *self, PyObject *args, PyObject *kwds)
{
       PyStructObject *soself = (PyStructObject *)self;
       PyObject *o_format = NULL;
       int ret = 0;
       static char *kwlist[] = {"format", 0};

       assert(PyStruct_Check(self));

       if (!PyArg_ParseTupleAndKeywords(args, kwds, "S:Struct", kwlist,
                                        &o_format))
               return -1;

       Py_INCREF(o_format);
       Py_XDECREF(soself->s_format);
       soself->s_format = o_format;

       ret = prepare_s(soself);
       return ret;
}

static void
s_dealloc(PyStructObject *s)
{
       if (s->weakreflist != NULL)
               PyObject_ClearWeakRefs((PyObject *)s);
       if (s->s_codes != NULL) {
               PyMem_FREE(s->s_codes);
       }
       Py_XDECREF(s->s_format);
       Py_Type(s)->tp_free((PyObject *)s);
}

static PyObject *
s_unpack_internal(PyStructObject *soself, char *startfrom) {
       formatcode *code;
       Py_ssize_t i = 0;
       PyObject *result = PyTuple_New(soself->s_len);
       if (result == NULL)
               return NULL;

       for (code = soself->s_codes; code->fmtdef != NULL; code++) {
               PyObject *v;
               const formatdef *e = code->fmtdef;
               const char *res = startfrom + code->offset;
               if (e->format == 's') {
                       v = PyString_FromStringAndSize(res, code->size);
               } else if (e->format == 'p') {
                       Py_ssize_t n = *(unsigned char*)res;
                       if (n >= code->size)
                               n = code->size - 1;
                       v = PyString_FromStringAndSize(res + 1, n);
               } else {
                       v = e->unpack(res, e);
               }
               if (v == NULL)
                       goto fail;
               PyTuple_SET_ITEM(result, i++, v);
       }

       return result;
fail:
       Py_DECREF(result);
       return NULL;
}


PyDoc_STRVAR(s_unpack__doc__,
"S.unpack(str) -> (v1, v2, ...)\n\
\n\
Return tuple containing values unpacked according to this Struct's format.\n\
Requires len(str) == self.size. See struct.__doc__ for more on format\n\
strings.");

static PyObject *
s_unpack(PyObject *self, PyObject *inputstr)
{
       PyStructObject *soself = (PyStructObject *)self;
       assert(PyStruct_Check(self));
       assert(soself->s_codes != NULL);
       if (inputstr == NULL || !PyString_Check(inputstr) ||
               PyString_GET_SIZE(inputstr) != soself->s_size) {
               PyErr_Format(StructError,
                       "unpack requires a string argument of length %zd",
                       soself->s_size);
               return NULL;
       }
       return s_unpack_internal(soself, PyString_AS_STRING(inputstr));
}

PyDoc_STRVAR(s_unpack_from__doc__,
"S.unpack_from(buffer[, offset]) -> (v1, v2, ...)\n\
\n\
Return tuple containing values unpacked according to this Struct's format.\n\
Unlike unpack, unpack_from can unpack values from any object supporting\n\
the buffer API, not just str. Requires len(buffer[offset:]) >= self.size.\n\
See struct.__doc__ for more on format strings.");

static PyObject *
s_unpack_from(PyObject *self, PyObject *args, PyObject *kwds)
{
       static char *kwlist[] = {"buffer", "offset", 0};
#if (PY_VERSION_HEX < 0x02050000)
       static char *fmt = "z#|i:unpack_from";
#else
       static char *fmt = "z#|n:unpack_from";
#endif
       Py_ssize_t buffer_len = 0, offset = 0;
       char *buffer = NULL;
       PyStructObject *soself = (PyStructObject *)self;
       assert(PyStruct_Check(self));
       assert(soself->s_codes != NULL);

       if (!PyArg_ParseTupleAndKeywords(args, kwds, fmt, kwlist,
                                        &buffer, &buffer_len, &offset))
               return NULL;

       if (buffer == NULL) {
               PyErr_Format(StructError,
                       "unpack_from requires a buffer argument");
               return NULL;
       }

       if (offset < 0)
               offset += buffer_len;

       if (offset < 0 || (buffer_len - offset) < soself->s_size) {
               PyErr_Format(StructError,
                       "unpack_from requires a buffer of at least %zd bytes",
                       soself->s_size);
               return NULL;
       }
       return s_unpack_internal(soself, buffer + offset);
}


/*
 * Guts of the pack function.
 *
 * Takes a struct object, a tuple of arguments, and offset in that tuple of
 * argument for where to start processing the arguments for packing, and a
 * character buffer for writing the packed string.  The caller must insure
 * that the buffer may contain the required length for packing the arguments.
 * 0 is returned on success, 1 is returned if there is an error.
 *
 */
static int
s_pack_internal(PyStructObject *soself, PyObject *args, int offset, char* buf)
{
       formatcode *code;
       /* XXX(nnorwitz): why does i need to be a local?  can we use
          the offset parameter or do we need the wider width? */
       Py_ssize_t i;

       memset(buf, '\0', soself->s_size);
       i = offset;
       for (code = soself->s_codes; code->fmtdef != NULL; code++) {
               Py_ssize_t n;
               PyObject *v = PyTuple_GET_ITEM(args, i++);
               const formatdef *e = code->fmtdef;
               char *res = buf + code->offset;
               if (e->format == 's') {
                       if (!PyString_Check(v)) {
                               PyErr_SetString(StructError,
                                               "argument for 's' must be a string");
                               return -1;
                       }
                       n = PyString_GET_SIZE(v);
                       if (n > code->size)
                               n = code->size;
                       if (n > 0)
                               memcpy(res, PyString_AS_STRING(v), n);
               } else if (e->format == 'p') {
                       if (!PyString_Check(v)) {
                               PyErr_SetString(StructError,
                                               "argument for 'p' must be a string");
                               return -1;
                       }
                       n = PyString_GET_SIZE(v);
                       if (n > (code->size - 1))
                               n = code->size - 1;
                       if (n > 0)
                               memcpy(res + 1, PyString_AS_STRING(v), n);
                       if (n > 255)
                               n = 255;
                       *res = Py_SAFE_DOWNCAST(n, Py_ssize_t, unsigned char);
               } else {
                       if (e->pack(res, v, e) < 0) {
                               if (PyLong_Check(v) && PyErr_ExceptionMatches(PyExc_OverflowError))
                                       PyErr_SetString(StructError,
                                                       "long too large to convert to int");
                               return -1;
                       }
               }
       }

       /* Success */
       return 0;
}


PyDoc_STRVAR(s_pack__doc__,
"S.pack(v1, v2, ...) -> string\n\
\n\
Return a string containing values v1, v2, ... packed according to this\n\
Struct's format. See struct.__doc__ for more on format strings.");

static PyObject *
s_pack(PyObject *self, PyObject *args)
{
       PyStructObject *soself;
       PyObject *result;

       /* Validate arguments. */
       soself = (PyStructObject *)self;
       assert(PyStruct_Check(self));
       assert(soself->s_codes != NULL);
       if (PyTuple_GET_SIZE(args) != soself->s_len)
       {
               PyErr_Format(StructError,
                       "pack requires exactly %zd arguments", soself->s_len);
               return NULL;
       }

       /* Allocate a new string */
       result = PyString_FromStringAndSize((char *)NULL, soself->s_size);
       if (result == NULL)
               return NULL;

       /* Call the guts */
       if ( s_pack_internal(soself, args, 0, PyString_AS_STRING(result)) != 0 ) {
               Py_DECREF(result);
               return NULL;
       }

       return result;
}

PyDoc_STRVAR(s_pack_into__doc__,
"S.pack_into(buffer, offset, v1, v2, ...)\n\
\n\
Pack the values v1, v2, ... according to this Struct's format, write \n\
the packed bytes into the writable buffer buf starting at offset.  Note\n\
that the offset is not an optional argument.  See struct.__doc__ for \n\
more on format strings.");

static PyObject *
s_pack_into(PyObject *self, PyObject *args)
{
       PyStructObject *soself;
       char *buffer;
       Py_ssize_t buffer_len, offset;

       /* Validate arguments.  +1 is for the first arg as buffer. */
       soself = (PyStructObject *)self;
       assert(PyStruct_Check(self));
       assert(soself->s_codes != NULL);
       if (PyTuple_GET_SIZE(args) != (soself->s_len + 2))
       {
               PyErr_Format(StructError,
                            "pack_into requires exactly %zd arguments",
                            (soself->s_len + 2));
               return NULL;
       }

       /* Extract a writable memory buffer from the first argument */
       if ( PyObject_AsWriteBuffer(PyTuple_GET_ITEM(args, 0),
                                                               (void**)&buffer, &buffer_len) == -1 ) {
               return NULL;
       }
       assert( buffer_len >= 0 );

       /* Extract the offset from the first argument */
       offset = PyInt_AsSsize_t(PyTuple_GET_ITEM(args, 1));

       /* Support negative offsets. */
       if (offset < 0)
               offset += buffer_len;

       /* Check boundaries */
       if (offset < 0 || (buffer_len - offset) < soself->s_size) {
               PyErr_Format(StructError,
                            "pack_into requires a buffer of at least %zd bytes",
                            soself->s_size);
               return NULL;
       }

       /* Call the guts */
       if ( s_pack_internal(soself, args, 2, buffer + offset) != 0 ) {
               return NULL;
       }

       Py_RETURN_NONE;
}

static PyObject *
s_get_format(PyStructObject *self, void *unused)
{
       Py_INCREF(self->s_format);
       return self->s_format;
}

static PyObject *
s_get_size(PyStructObject *self, void *unused)
{
       return PyInt_FromSsize_t(self->s_size);
}

static PyObject *
s_get_numcodes(PyStructObject *self, void *unused)
{
       return PyInt_FromSsize_t(self->s_len);
}

/* List of functions */

static struct PyMethodDef s_methods[] = {
       {"pack",        s_pack,         METH_VARARGS, s_pack__doc__},
       {"pack_into",   s_pack_into,    METH_VARARGS, s_pack_into__doc__},
       {"unpack",      s_unpack,       METH_O, s_unpack__doc__},
       {"unpack_from", (PyCFunction)s_unpack_from, METH_KEYWORDS,
                       s_unpack_from__doc__},
       {NULL,   NULL}          /* sentinel */
};

PyDoc_STRVAR(s__doc__, "Compiled struct object");

#define OFF(x) offsetof(PyStructObject, x)

static PyGetSetDef s_getsetlist[] = {
       {"format", (getter)s_get_format, (setter)NULL, "struct format string", NULL},
       {"size", (getter)s_get_size, (setter)NULL, "struct size in bytes", NULL},
       {"numcodes",  (getter)s_get_len, (setter)NULL, "number of elements in struct", NULL},
       {NULL} /* sentinel */
};

static
PyTypeObject PyStructType = {
       PyObject_HEAD_INIT(NULL)
       0,
       "Struct",
       sizeof(PyStructObject),
       0,
       (destructor)s_dealloc,  /* tp_dealloc */
       0,                                      /* tp_print */
       0,                                      /* tp_getattr */
       0,                                      /* tp_setattr */
       0,                                      /* tp_compare */
       0,                                      /* tp_repr */
       0,                                      /* tp_as_number */
       0,                                      /* tp_as_sequence */
       0,                                      /* tp_as_mapping */
       0,                                      /* tp_hash */
       0,                                      /* tp_call */
       0,                                      /* tp_str */
       PyObject_GenericGetAttr,        /* tp_getattro */
       PyObject_GenericSetAttr,        /* tp_setattro */
       0,                                      /* tp_as_buffer */
       Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
       s__doc__,                       /* tp_doc */
       0,                                      /* tp_traverse */
       0,                                      /* tp_clear */
       0,                                      /* tp_richcompare */
       offsetof(PyStructObject, weakreflist),  /* tp_weaklistoffset */
       0,                                      /* tp_iter */
       0,                                      /* tp_iternext */
       s_methods,                      /* tp_methods */
       NULL,                           /* tp_members */
       s_getsetlist,           /* tp_getset */
       0,                              /* tp_base */
       0,                                      /* tp_dict */
       0,                                      /* tp_descr_get */
       0,                                      /* tp_descr_set */
       0,                                      /* tp_dictoffset */
       s_init,                         /* tp_init */
       PType_GenericAlloc,            /* tp_alloc */
       s_new,                          /* tp_new */
       PyObject_Del,           /* tp_free */
};


static
PyTypeObject PyFieldTupleType = {
       PyObject_HEAD_INIT(NULL)
       0,
       "fieldtuple",
       sizeof(PyFieldTupleObject),
       0,
       (destructor)f_dealloc,                 /* tp_dealloc */
       0,                                      /* tp_print */
       0,                                      /* tp_getattr */
       0,                                      /* tp_setattr */
       0,                                      /* tp_compare */
       0,                                      /* tp_repr */
       0,                                      /* tp_as_number */
       0,                                      /* tp_as_sequence */
       &f_as_mapping,                          /* tp_as_mapping */
       0,                                      /* tp_hash */
       0,                                      /* tp_call */
       0,                                      /* tp_str */
       PyObject_GenericGetAttr,                /* tp_getattro */
       PyObject_GenericSetAttr,                /* tp_setattro */
       0,                                      /* tp_as_buffer */
       Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
       f__doc__,                               /* tp_doc */
       0,                                      /* tp_traverse */
       0,                                      /* tp_clear */
       0,                                      /* tp_richcompare */
       0,                                      /* tp_weaklistoffset */
       0,                                      /* tp_iter */
       0,                                      /* tp_iternext */
       0,                                      /* tp_methods */
       NULL,                                   /* tp_members */
       0,                                      /* tp_getset */
       &PyTuple_Type,                          /* tp_base */
       0,                                      /* tp_dict */
       0,                                      /* tp_descr_get */
       0,                                      /* tp_descr_set */
       0,                                      /* tp_dictoffset */
       f_init,                         /* tp_init */
       PyType_GenericAlloc,            /* tp_alloc */
       f_new,                          /* tp_new */
       PyObject_Del,                   /* tp_free */
};

/* Module initialization */

PyMODINIT_FUNC
init_struct(void)
{
       PyObject *m = Py_InitModule("_struct", NULL);
       if (m == NULL)
               return;

       /* fill in LONG_LONG and LONG_DOUBLE table-entries if needed */
       {
               const formatdef *entry;
               const formatdef *table;

               table = (const formatdef *)general_table;
#ifdef HAVE_LONG_LONG

               entry = getentry('q',table);
               entry->size = sizeof(PY_LONG_LONG);
               entry->alignment = LONG_LONG_ALIGN;

               entry = getentry('Q',table);
               entry->size = sizeof(PY_LONG_LONG);
               entry->alignment = LONG_LONG_ALIGN;
#endif

#ifdef HAVE_LONG_DOUBLE
               entry = getentry('g',table);
               entry->size = sizeof(long double);
               entry->alignment = LONG_DOUBLE_ALIGN;
#endif
       }

       Py_Type(&PyStructType) = &PyType_Type;
       if (PyType_Ready(&PyStructType) < 0)
               return;

       Py_Type(&PyFieldTupleType) = &PyType_Type;
       if (PyType_Read(&PyFieldTupleType) < 0)
               return;

#ifdef PY_STRUCT_OVERFLOW_MASKING
       if (pyint_zero == NULL) {
               pyint_zero = PyInt_FromLong(0);
               if (pyint_zero == NULL)
                       return;
       }
       if (pylong_ulong_mask == NULL) {
 #if (SIZEOF_LONG == 4)
               pylong_ulong_mask = PyLong_FromString("FFFFFFFF", NULL, 16);
 #else
               pylong_ulong_mask = PyLong_FromString("FFFFFFFFFFFFFFFF", NULL, 16);
 #endif
               if (pylong_ulong_mask == NULL)
                       return;
       }
#endif


       /* Add some symbolic constants to the module */
       if (StructError == NULL) {
               StructError = PyErr_NewException("struct.error", NULL, NULL);
               if (StructError == NULL)
                       return;
       }

       Py_INCREF(StructError);
       PyModule_AddObject(m, "error", StructError);

       Py_INCREF((PyObject*)&PyStructType);
       PyModule_AddObject(m, "Struct", (PyObject*)&PyStructType);

       PyModule_AddIntConstant(m, "_PY_STRUCT_RANGE_CHECKING", 1);
#ifdef PY_STRUCT_OVERFLOW_MASKING
       PyModule_AddIntConstant(m, "_PY_STRUCT_OVERFLOW_MASKING", 1);
#endif
#ifdef PY_STRUCT_FLOAT_COERCE
       PyModule_AddIntConstant(m, "_PY_STRUCT_FLOAT_COERCE", 1);
#endif

}



