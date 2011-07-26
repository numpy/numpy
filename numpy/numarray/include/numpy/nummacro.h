/* Primarily for compatibility with numarray C-API */

#if !defined(_ndarraymacro)
#define _ndarraymacro

/* The structs defined here are private implementation details of numarray
which are subject to change w/o notice.
*/

#define PY_BOOL_CHAR "b"
#define PY_INT8_CHAR "b"
#define PY_INT16_CHAR "h"
#define PY_INT32_CHAR "i"
#define PY_FLOAT32_CHAR "f"
#define PY_FLOAT64_CHAR "d"
#define PY_UINT8_CHAR "h"
#define PY_UINT16_CHAR "i"
#define PY_UINT32_CHAR "i" /* Unless longer int available */
#define PY_COMPLEX64_CHAR "D"
#define PY_COMPLEX128_CHAR "D"

#define PY_LONG_CHAR "l"
#define PY_LONG_LONG_CHAR "L"

#define pyFPE_DIVIDE_BY_ZERO  1
#define pyFPE_OVERFLOW        2
#define pyFPE_UNDERFLOW       4
#define pyFPE_INVALID         8

#define isNonZERO(x) (x != 0) /* to convert values to boolean 1's or 0's */

typedef enum
{
	NUM_CONTIGUOUS=1,
	NUM_NOTSWAPPED=0x0200,
	NUM_ALIGNED=0x0100,
	NUM_WRITABLE=0x0400,
	NUM_COPY=0x0020,

	NUM_C_ARRAY  = (NUM_CONTIGUOUS | NUM_ALIGNED | NUM_NOTSWAPPED),
	NUM_UNCONVERTED = 0
} NumRequirements;

#define UNCONVERTED 0
#define C_ARRAY     (NUM_CONTIGUOUS | NUM_NOTSWAPPED | NUM_ALIGNED)

#define MUST_BE_COMPUTED 2

#define NUM_FLOORDIVIDE(a,b,out) (out) = floor((a)/(b))

#define NA_Begin() Py_Initialize(); import_libnumarray();
#define NA_End()   NA_Done(); Py_Finalize();

#define NA_OFFSETDATA(num) ((void *) PyArray_DATA(num))

/* unaligned NA_COPY functions */
#define NA_COPY1(i, o)  (*(o) = *(i))
#define NA_COPY2(i, o)  NA_COPY1(i, o), NA_COPY1(i+1, o+1)
#define NA_COPY4(i, o)  NA_COPY2(i, o), NA_COPY2(i+2, o+2)
#define NA_COPY8(i, o)  NA_COPY4(i, o), NA_COPY4(i+4, o+4)
#define NA_COPY16(i, o) NA_COPY8(i, o), NA_COPY8(i+8, o+8)

/* byteswapping macros: these fail if i==o */
#define NA_SWAP1(i, o)  NA_COPY1(i, o)
#define NA_SWAP2(i, o)  NA_SWAP1(i, o+1), NA_SWAP1(i+1, o)
#define NA_SWAP4(i, o)  NA_SWAP2(i, o+2), NA_SWAP2(i+2, o)
#define NA_SWAP8(i, o)  NA_SWAP4(i, o+4), NA_SWAP4(i+4, o)
#define NA_SWAP16(i, o) NA_SWAP8(i, o+8), NA_SWAP8(i+8, o)

/* complex byteswaps must swap each part (real, imag) independently */
#define NA_COMPLEX_SWAP8(i, o)  NA_SWAP4(i, o), NA_SWAP4(i+4, o+4)
#define NA_COMPLEX_SWAP16(i, o) NA_SWAP8(i, o), NA_SWAP8(i+8, o+8)

/* byteswapping macros:  these work even if i == o */
#define NA_TSWAP1(i, o, t) NA_COPY1(i, t), NA_SWAP1(t, o)
#define NA_TSWAP2(i, o, t) NA_COPY2(i, t), NA_SWAP2(t, o)
#define NA_TSWAP4(i, o, t) NA_COPY4(i, t), NA_SWAP4(t, o)
#define NA_TSWAP8(i, o, t) NA_COPY8(i, t), NA_SWAP8(t, o)

/* fast copy functions for %N aligned i and o */
#define NA_ACOPY1(i, o) (((Int8    *)o)[0]   = ((Int8    *)i)[0])
#define NA_ACOPY2(i, o) (((Int16   *)o)[0]   = ((Int16   *)i)[0])
#define NA_ACOPY4(i, o) (((Int32   *)o)[0]   = ((Int32   *)i)[0])
#define NA_ACOPY8(i, o) (((Float64 *)o)[0]   = ((Float64 *)i)[0])
#define NA_ACOPY16(i, o) (((Complex64 *)o)[0]   = ((Complex64 *)i)[0])

/* from here down, type("ai") is NDInfo*  */

#define NA_PTR(ai)   ((char *) NA_OFFSETDATA((ai)))
#define NA_PTR1(ai, i)       (NA_PTR(ai) + \
                              (i)*PyArray_STRIDES(ai)[0])
#define NA_PTR2(ai, i, j)    (NA_PTR(ai) + \
                              (i)*PyArray_STRIDES(ai)[0] + \
                              (j)*PyArray_STRIDES(ai)[1])
#define NA_PTR3(ai, i, j, k) (NA_PTR(ai) + \
                              (i)*PyArray_STRIDES(ai)[0] + \
                              (j)*PyArray_STRIDES(ai)[1] + \
                              (k)*PyArray_STRIDES(ai)[2])

#define NA_SET_TEMP(ai, type, v) (((type *) &__temp__)[0] = v)

#define NA_SWAPComplex64 NA_COMPLEX_SWAP16
#define NA_SWAPComplex32 NA_COMPLEX_SWAP8
#define NA_SWAPFloat64   NA_SWAP8
#define NA_SWAPFloat32   NA_SWAP4
#define NA_SWAPInt64     NA_SWAP8
#define NA_SWAPUInt64    NA_SWAP8
#define NA_SWAPInt32     NA_SWAP4
#define NA_SWAPUInt32    NA_SWAP4
#define NA_SWAPInt16     NA_SWAP2
#define NA_SWAPUInt16    NA_SWAP2
#define NA_SWAPInt8      NA_SWAP1
#define NA_SWAPUInt8     NA_SWAP1
#define NA_SWAPBool      NA_SWAP1

#define NA_COPYComplex64 NA_COPY16
#define NA_COPYComplex32 NA_COPY8
#define NA_COPYFloat64   NA_COPY8
#define NA_COPYFloat32   NA_COPY4
#define NA_COPYInt64     NA_COPY8
#define NA_COPYUInt64    NA_COPY8
#define NA_COPYInt32     NA_COPY4
#define NA_COPYUInt32    NA_COPY4
#define NA_COPYInt16     NA_COPY2
#define NA_COPYUInt16    NA_COPY2
#define NA_COPYInt8      NA_COPY1
#define NA_COPYUInt8     NA_COPY1
#define NA_COPYBool      NA_COPY1

#ifdef __cplusplus
extern "C" {
#endif

#define _makeGetPb(type)		\
static type _NA_GETPb_##type(char *ptr)	\
{						\
	type temp;				\
	NA_SWAP##type(ptr, (char *)&temp);	\
	return temp;				\
}

#define _makeGetPa(type)	             	\
static type _NA_GETPa_##type(char *ptr)         \
{						\
	type temp;				\
	NA_COPY##type(ptr, (char *)&temp);	\
	return temp;				\
}

_makeGetPb(Complex64)
_makeGetPb(Complex32)
_makeGetPb(Float64)
_makeGetPb(Float32)
_makeGetPb(Int64)
_makeGetPb(UInt64)
_makeGetPb(Int32)
_makeGetPb(UInt32)
_makeGetPb(Int16)
_makeGetPb(UInt16)
_makeGetPb(Int8)
_makeGetPb(UInt8)
_makeGetPb(Bool)

_makeGetPa(Complex64)
_makeGetPa(Complex32)
_makeGetPa(Float64)
_makeGetPa(Float32)
_makeGetPa(Int64)
_makeGetPa(UInt64)
_makeGetPa(Int32)
_makeGetPa(UInt32)
_makeGetPa(Int16)
_makeGetPa(UInt16)
_makeGetPa(Int8)
_makeGetPa(UInt8)
_makeGetPa(Bool)

#undef _makeGetPb
#undef _makeGetPa

#define _makeSetPb(type)		\
static void _NA_SETPb_##type(char *ptr, type v)	\
{						\
	NA_SWAP##type(((char *)&v), ptr);	\
	return;					\
}

#define _makeSetPa(type) \
static void _NA_SETPa_##type(char *ptr, type v)	\
{						\
	NA_COPY##type(((char *)&v), ptr);	\
	return;					\
}

_makeSetPb(Complex64)
_makeSetPb(Complex32)
_makeSetPb(Float64)
_makeSetPb(Float32)
_makeSetPb(Int64)
_makeSetPb(UInt64)
_makeSetPb(Int32)
_makeSetPb(UInt32)
_makeSetPb(Int16)
_makeSetPb(UInt16)
_makeSetPb(Int8)
_makeSetPb(UInt8)
_makeSetPb(Bool)

_makeSetPa(Complex64)
_makeSetPa(Complex32)
_makeSetPa(Float64)
_makeSetPa(Float32)
_makeSetPa(Int64)
_makeSetPa(UInt64)
_makeSetPa(Int32)
_makeSetPa(UInt32)
_makeSetPa(Int16)
_makeSetPa(UInt16)
_makeSetPa(Int8)
_makeSetPa(UInt8)
_makeSetPa(Bool)

#undef _makeSetPb
#undef _makeSetPa

#ifdef __cplusplus
	}
#endif

/* ========================== ptr get/set ================================ */

/* byteswapping */
#define NA_GETPb(ai, type, ptr) _NA_GETPb_##type(ptr)

/* aligning */
#define NA_GETPa(ai, type, ptr) _NA_GETPa_##type(ptr)

/* fast (aligned, !byteswapped) */
#define NA_GETPf(ai, type, ptr) (*((type *) (ptr)))

#define NA_GETP(ai, type, ptr) \
   (PyArray_ISCARRAY(ai) ? NA_GETPf(ai, type, ptr) \
                   : (PyArray_ISBYTESWAPPED(ai) ? \
                                      NA_GETPb(ai, type, ptr) \
                                    : NA_GETPa(ai, type, ptr)))

/* NOTE:  NA_SET* macros cannot be used as values. */

/* byteswapping */
#define NA_SETPb(ai, type, ptr, v) _NA_SETPb_##type(ptr, v)

/* aligning */
#define NA_SETPa(ai, type, ptr, v) _NA_SETPa_##type(ptr, v)

/* fast (aligned, !byteswapped) */
#define NA_SETPf(ai, type, ptr, v) ((*((type *) ptr)) = (v))

#define NA_SETP(ai, type, ptr, v) \
    if (PyArray_ISCARRAY(ai)) { \
         NA_SETPf((ai), type, (ptr), (v)); \
    } else if (PyArray_ISBYTESWAPPED(ai)) { \
	 NA_SETPb((ai), type, (ptr), (v)); \
    } else \
         NA_SETPa((ai), type, (ptr), (v))

/* ========================== 1 index get/set ============================ */

/* byteswapping */
#define NA_GET1b(ai, type, i)    NA_GETPb(ai, type, NA_PTR1(ai, i))
/* aligning */
#define NA_GET1a(ai, type, i)    NA_GETPa(ai, type, NA_PTR1(ai, i))
/* fast (aligned, !byteswapped) */
#define NA_GET1f(ai, type, i)    NA_GETPf(ai, type, NA_PTR1(ai, i))
/* testing */
#define NA_GET1(ai, type, i)     NA_GETP(ai, type, NA_PTR1(ai, i))

/* byteswapping */
#define NA_SET1b(ai, type, i, v) NA_SETPb(ai, type, NA_PTR1(ai, i), v)
/* aligning */
#define NA_SET1a(ai, type, i, v) NA_SETPa(ai, type, NA_PTR1(ai, i), v)
/* fast (aligned, !byteswapped) */
#define NA_SET1f(ai, type, i, v) NA_SETPf(ai, type, NA_PTR1(ai, i), v)
/* testing */
#define NA_SET1(ai, type, i, v)  NA_SETP(ai, type,  NA_PTR1(ai, i), v)

/* ========================== 2 index get/set ============================= */

/* byteswapping */
#define NA_GET2b(ai, type, i, j)    NA_GETPb(ai, type, NA_PTR2(ai, i, j))
/* aligning */
#define NA_GET2a(ai, type, i, j)    NA_GETPa(ai, type, NA_PTR2(ai, i, j))
/* fast (aligned, !byteswapped) */
#define NA_GET2f(ai, type, i, j)    NA_GETPf(ai, type, NA_PTR2(ai, i, j))
/* testing */
#define NA_GET2(ai, type, i, j)     NA_GETP(ai, type, NA_PTR2(ai, i, j))

/* byteswapping */
#define NA_SET2b(ai, type, i, j, v) NA_SETPb(ai, type, NA_PTR2(ai, i, j), v)
/* aligning */
#define NA_SET2a(ai, type, i, j, v) NA_SETPa(ai, type, NA_PTR2(ai, i, j), v)
/* fast (aligned, !byteswapped) */
#define NA_SET2f(ai, type, i, j, v) NA_SETPf(ai, type, NA_PTR2(ai, i, j), v)

#define NA_SET2(ai, type, i, j,  v)  NA_SETP(ai, type,  NA_PTR2(ai, i, j), v)

/* ========================== 3 index get/set ============================= */

/* byteswapping */
#define NA_GET3b(ai, type, i, j, k)    NA_GETPb(ai, type, NA_PTR3(ai, i, j, k))
/* aligning */
#define NA_GET3a(ai, type, i, j, k)    NA_GETPa(ai, type, NA_PTR3(ai, i, j, k))
/* fast (aligned, !byteswapped) */
#define NA_GET3f(ai, type, i, j, k)    NA_GETPf(ai, type, NA_PTR3(ai, i, j, k))
/* testing */
#define NA_GET3(ai, type, i, j, k)     NA_GETP(ai, type, NA_PTR3(ai, i, j, k))

/* byteswapping */
#define NA_SET3b(ai, type, i, j, k, v) \
        NA_SETPb(ai, type, NA_PTR3(ai, i, j, k), v)
/* aligning */
#define NA_SET3a(ai, type, i, j, k, v) \
        NA_SETPa(ai, type, NA_PTR3(ai, i, j, k), v)
/* fast (aligned, !byteswapped) */
#define NA_SET3f(ai, type, i, j, k, v) \
        NA_SETPf(ai, type, NA_PTR3(ai, i, j, k), v)
#define NA_SET3(ai, type, i, j, k, v) \
        NA_SETP(ai, type,  NA_PTR3(ai, i, j, k), v)

/* ========================== 1D get/set ================================== */

#define NA_GET1Db(ai, type, base, cnt, out) \
        { int i, stride = PyArray_STRIDES(ai)[PyArray_NDIM(ai)-1]; \
           for(i=0; i<cnt; i++) { \
               out[i] = NA_GETPb(ai, type, base); \
               base += stride; \
           } \
        }

#define NA_GET1Da(ai, type, base, cnt, out)                                   \
        { int i, stride = PyArray_STRIDES(ai)[PyArray_NDIM(ai)-1]; \
           for(i=0; i<cnt; i++) {                                             \
               out[i] = NA_GETPa(ai, type, base);                             \
               base += stride;                                                \
           }                                                                  \
        }

#define NA_GET1Df(ai, type, base, cnt, out)                                   \
        { int i, stride = PyArray_STRIDES(ai)[PyArray_NDIM(ai)-1]; \
           for(i=0; i<cnt; i++) {                                             \
               out[i] = NA_GETPf(ai, type, base);                             \
               base += stride;                                                \
           }                                                                  \
        }

#define NA_GET1D(ai, type, base, cnt, out)                                    \
        if (PyArray_ISCARRAY(ai)) {                                           \
	      NA_GET1Df(ai, type, base, cnt, out);                            \
        } else if (PyArray_ISBYTESWAPPED(ai)) {                               \
              NA_GET1Db(ai, type, base, cnt, out);                            \
        } else {                                                              \
              NA_GET1Da(ai, type, base, cnt, out);                            \
	}

#define NA_SET1Db(ai, type, base, cnt, in)                                    \
        { int i, stride = PyArray_STRIDES(ai)[PyArray_NDIM(ai)-1]; \
           for(i=0; i<cnt; i++) {                                             \
               NA_SETPb(ai, type, base, in[i]);                               \
               base += stride;                                                \
           }                                                                  \
        }

#define NA_SET1Da(ai, type, base, cnt, in)                                    \
        { int i, stride = PyArray_STRIDES(ai)[PyArray_NDIM(ai)-1]; \
           for(i=0; i<cnt; i++) {                                             \
               NA_SETPa(ai, type, base, in[i]);                               \
               base += stride;                                                \
           }                                                                  \
        }

#define NA_SET1Df(ai, type, base, cnt, in)                                    \
        { int i, stride = PyArray_STRIDES(ai)[PyArray_NDIM(ai)-1]; \
           for(i=0; i<cnt; i++) {                                             \
               NA_SETPf(ai, type, base, in[i]);                               \
               base += stride;                                                \
           }                                                                  \
        }

#define NA_SET1D(ai, type, base, cnt, out)                                    \
        if (PyArray_ISCARRAY(ai)) {                                           \
              NA_SET1Df(ai, type, base, cnt, out);                            \
        } else if (PyArray_ISBYTESWAPPED(ai)) {                               \
              NA_SET1Db(ai, type, base, cnt, out);                            \
        } else {                                                              \
	      NA_SET1Da(ai, type, base, cnt, out);                            \
	}

/* ========================== utilities ================================== */

#if !defined(MIN)
#define MIN(x,y) (((x)<=(y)) ? (x) : (y))
#endif

#if !defined(MAX)
#define MAX(x,y) (((x)>=(y)) ? (x) : (y))
#endif

#if !defined(ABS)
#define ABS(x) (((x) >= 0) ? (x) : -(x))
#endif

#define ELEM(x)  (sizeof(x)/sizeof(x[0]))

#define BOOLEAN_BITWISE_NOT(x) ((x) ^ 1)

#define NA_NBYTES(a) (PyArray_DESCR(a)->elsize * NA_elements(a))

#if defined(NA_SMP)
#define BEGIN_THREADS Py_BEGIN_ALLOW_THREADS
#define END_THREADS Py_END_ALLOW_THREADS
#else
#define BEGIN_THREADS
#define END_THREADS
#endif

#if !defined(NA_isnan)

#define U32(u) (* (Int32 *) &(u) )
#define U64(u) (* (Int64 *) &(u) )

#define NA_isnan32(u) \
  ( (( U32(u) & 0x7f800000) == 0x7f800000)  && ((U32(u) & 0x007fffff) != 0)) ? 1:0

#if !defined(_MSC_VER)
#define NA_isnan64(u) \
  ( (( U64(u) & 0x7ff0000000000000LL) == 0x7ff0000000000000LL)  && ((U64(u) & 0x000fffffffffffffLL) != 0)) ? 1:0
#else
#define NA_isnan64(u) \
  ( (( U64(u) & 0x7ff0000000000000i64) == 0x7ff0000000000000i64)  && ((U64(u) & 0x000fffffffffffffi64) != 0)) ? 1:0
#endif

#define NA_isnanC32(u) (NA_isnan32(((Complex32 *)&(u))->r) || NA_isnan32(((Complex32 *)&(u))->i))
#define NA_isnanC64(u) (NA_isnan64(((Complex64 *)&(u))->r) || NA_isnan64(((Complex64 *)&(u))->i))

#endif /* NA_isnan */


#endif /* _ndarraymacro */
