#if !defined(__arraybase_h)
#define _arraybase_h 1

#define SZ_BUF  79
#define MAXDIM MAX_DIMS

#define maybelong intp

typedef enum
{
  tAny,
  tBool=PyArray_BOOL,
  tInt8=PyArray_INT8,
  tUInt8=PyArray_UINT8,
  tInt16=PyArray_INT16,
  tUInt16=PyArray_UINT16,
  tInt32=PyArray_INT32,
  tUInt32=PyArray_UINT32,
  tInt64=PyArray_INT64,
  tUInt64=PyArray_UINT64,
  tFloat32=PyArray_FLOAT32,
  tFloat64=PyArray_FLOAT64,
  tComplex32=PyArray_COMPLEX64,
  tComplex64=PyArray_COMPLEX128,
  tObject=PyArray_OBJECT,        /* placeholder... does nothing */
  tDefault = tFloat64,
#if BITSOF_LONG == 64
  tLong = tInt64,
#else
  tLong = tInt32,
#endif
  tMaxType
} NumarrayType;

#define nNumarrayType 16

#define HAS_UINT64 1

typedef enum
{
        NUM_LITTLE_ENDIAN=0,
        NUM_BIG_ENDIAN = 1
} NumarrayByteOrder;



#define Complex64 Complex64_
typedef struct { Float32 r, i; } Complex32;
typedef struct { Float64 r, i; } Complex64;

#define WRITABLE WRITEABLE
#define CHECKOVERFLOW 0x800
#define UPDATEDICT 0x1000
#define FORTRAN_CONTIGUOUS FORTRAN
#define IS_CARRAY (CONTIGUOUS | ALIGNED)

#define PyArray(m)                      ((PyArrayObject *)(m))
#define PyArray_ISFORTRAN_CONTIGUOUS(m) (((PyArray(m))->flags & FORTRAN_CONTIGUOUS) != 0)
#define PyArray_ISBYTESWAPPED(m) (!PyArray_ISNOTSWAPPED(m))
#define PyArray_ISWRITABLE  PyArray_ISWRITEABLE 
#define PyArray_ISSPACESAVER(m)  0
#define PyArray_ISCARRAY(m)      (((PyArray(m))->flags & IS_CARRAY) == IS_CARRAY)


#endif /* 
