#if !defined(__arraybase_h)
#define _arraybase_h 1

#define SZ_BUF  79
#define MAXDIM NPY_MAXDIMS
#define MAXARGS 18

typedef npy_intp maybelong;
typedef npy_bool Bool;
typedef npy_int8 Int8;
typedef npy_uint8 UInt8;
typedef npy_int16 Int16;
typedef npy_uint16 UInt16;
typedef npy_int32 Int32;
typedef npy_uint32 UInt32; 
typedef npy_int64 Int64;
typedef npy_uint64 UInt64;
typedef npy_float32 Float32;
typedef npy_float64 Float64;

typedef enum
{
  tAny=-1,
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
  tMaxType=PyArray_NTYPES,
  tDefault = tFloat64,
#if NPY_BITSOF_LONG == 64
  tLong = tInt64,
#else
  tLong = tInt32,
#endif
} NumarrayType;

#define nNumarrayType PyArray_NTYPES

#define HAS_UINT64 1

typedef enum
{
        NUM_LITTLE_ENDIAN=0,
        NUM_BIG_ENDIAN = 1
} NumarrayByteOrder;

typedef struct { Float32 r, i; } Complex32;
typedef struct { Float64 r, i; } Complex64;

#define WRITABLE NPY_WRITEABLE
#define CHECKOVERFLOW 0x800
#define UPDATEDICT 0x1000
#define FORTRAN_CONTIGUOUS NPY_FORTRAN
#define IS_CARRAY (NPY_CONTIGUOUS | NPY_ALIGNED)

#define PyArray(m)                      ((PyArrayObject *)(m))
#define PyArray_ISFORTRAN_CONTIGUOUS(m) (((PyArray(m))->flags & FORTRAN_CONTIGUOUS) != 0)
#define PyArray_ISWRITABLE  PyArray_ISWRITEABLE 


#endif 
