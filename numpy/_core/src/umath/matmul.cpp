#include "matmul.h"

/***********************************************************************************
** Defining matmul functions
***********************************************************************************/
#define DEFINE_MATMUL_FUNCTION(NAME, TYPE)                          \
void NAME##_matmul(char **args,                                 \
                   npy_intp const *dimensions,                  \
                   npy_intp const *steps,                       \
                   void *NPY_UNUSED(func))                      \
{                                                               \
    matmul<TYPE>(args, dimensions, steps, nullptr);             \
}

DEFINE_MATMUL_FUNCTION(FLOAT, npy_float)
DEFINE_MATMUL_FUNCTION(DOUBLE, npy_double)
DEFINE_MATMUL_FUNCTION(LONGDOUBLE, npy_longdouble)
DEFINE_MATMUL_FUNCTION(HALF, npy_half)
DEFINE_MATMUL_FUNCTION(CFLOAT, npy_cfloat)
DEFINE_MATMUL_FUNCTION(CDOUBLE, npy_cdouble)
DEFINE_MATMUL_FUNCTION(CLONGDOUBLE, npy_clongdouble)
DEFINE_MATMUL_FUNCTION(UBYTE, npy_ubyte)
DEFINE_MATMUL_FUNCTION(USHORT, npy_ushort)
DEFINE_MATMUL_FUNCTION(UINT, npy_uint)
DEFINE_MATMUL_FUNCTION(ULONG, npy_ulong)
DEFINE_MATMUL_FUNCTION(ULONGLONG, npy_ulonglong)
DEFINE_MATMUL_FUNCTION(BYTE, npy_byte)
DEFINE_MATMUL_FUNCTION(SHORT, npy_short)
DEFINE_MATMUL_FUNCTION(INT, npy_int)
DEFINE_MATMUL_FUNCTION(LONG, npy_long)
DEFINE_MATMUL_FUNCTION(LONGLONG, npy_longlong)
DEFINE_MATMUL_FUNCTION(BOOL, npy_bool)
DEFINE_MATMUL_FUNCTION(OBJECT, PyObject*)

#undef DEFINE_MATMUL_FUNCTION

/***********************************************************************************
** Defining vecdot functions
***********************************************************************************/
#define DEFINE_VECDOT_FUNCTION(NAME, TYPE)                          \
void NAME##_vecdot(char **args,                                 \
                   npy_intp const *dimensions,                  \
                   npy_intp const *steps,                       \
                   void *NPY_UNUSED(func))                      \
{                                                               \
    vecdot<TYPE>(args, dimensions, steps, nullptr);             \
}

DEFINE_VECDOT_FUNCTION(FLOAT, npy_float)
DEFINE_VECDOT_FUNCTION(DOUBLE, npy_double)
DEFINE_VECDOT_FUNCTION(LONGDOUBLE, npy_longdouble)
DEFINE_VECDOT_FUNCTION(HALF, npy_half)
DEFINE_VECDOT_FUNCTION(CFLOAT, npy_cfloat)
DEFINE_VECDOT_FUNCTION(CDOUBLE, npy_cdouble)
DEFINE_VECDOT_FUNCTION(CLONGDOUBLE, npy_clongdouble)
DEFINE_VECDOT_FUNCTION(UBYTE, npy_ubyte)
DEFINE_VECDOT_FUNCTION(USHORT, npy_ushort)
DEFINE_VECDOT_FUNCTION(UINT, npy_uint)
DEFINE_VECDOT_FUNCTION(ULONG, npy_ulong)
DEFINE_VECDOT_FUNCTION(ULONGLONG, npy_ulonglong)
DEFINE_VECDOT_FUNCTION(BYTE, npy_byte)
DEFINE_VECDOT_FUNCTION(SHORT, npy_short)
DEFINE_VECDOT_FUNCTION(INT, npy_int)
DEFINE_VECDOT_FUNCTION(LONG, npy_long)
DEFINE_VECDOT_FUNCTION(LONGLONG, npy_longlong)
DEFINE_VECDOT_FUNCTION(BOOL, npy_bool)
DEFINE_VECDOT_FUNCTION(OBJECT, PyObject*)

#undef DEFINE_VECDOT_FUNCTION

/***********************************************************************************
** Defining matvec functions
***********************************************************************************/
#define DEFINE_MATVEC_FUNCTION(NAME, TYPE)                          \
void NAME##_matvec(char **args,                                 \
                   npy_intp const *dimensions,                  \
                   npy_intp const *steps,                       \
                   void *NPY_UNUSED(func))                      \
{                                                               \
    matvec<TYPE>(args, dimensions, steps, nullptr);             \
}

DEFINE_MATVEC_FUNCTION(FLOAT, npy_float)
DEFINE_MATVEC_FUNCTION(DOUBLE, npy_double)
DEFINE_MATVEC_FUNCTION(LONGDOUBLE, npy_longdouble)
DEFINE_MATVEC_FUNCTION(HALF, npy_half)
DEFINE_MATVEC_FUNCTION(CFLOAT, npy_cfloat)
DEFINE_MATVEC_FUNCTION(CDOUBLE, npy_cdouble)
DEFINE_MATVEC_FUNCTION(CLONGDOUBLE, npy_clongdouble)
DEFINE_MATVEC_FUNCTION(UBYTE, npy_ubyte)
DEFINE_MATVEC_FUNCTION(USHORT, npy_ushort)
DEFINE_MATVEC_FUNCTION(UINT, npy_uint)
DEFINE_MATVEC_FUNCTION(ULONG, npy_ulong)
DEFINE_MATVEC_FUNCTION(ULONGLONG, npy_ulonglong)
DEFINE_MATVEC_FUNCTION(BYTE, npy_byte)
DEFINE_MATVEC_FUNCTION(SHORT, npy_short)
DEFINE_MATVEC_FUNCTION(INT, npy_int)
DEFINE_MATVEC_FUNCTION(LONG, npy_long)
DEFINE_MATVEC_FUNCTION(LONGLONG, npy_longlong)
DEFINE_MATVEC_FUNCTION(BOOL, npy_bool)
DEFINE_MATVEC_FUNCTION(OBJECT, PyObject*)

#undef DEFINE_MATVEC_FUNCTION

/***********************************************************************************
** Defining vecmat functions
***********************************************************************************/
#define DEFINE_VECMAT_FUNCTION(NAME, TYPE)                          \
void NAME##_vecmat(char **args,                                 \
                   npy_intp const *dimensions,                  \
                   npy_intp const *steps,                       \
                   void *NPY_UNUSED(func))                      \
{                                                               \
    vecmat<TYPE>(args, dimensions, steps, nullptr);             \
}

DEFINE_VECMAT_FUNCTION(FLOAT, npy_float)
DEFINE_VECMAT_FUNCTION(DOUBLE, npy_double)
DEFINE_VECMAT_FUNCTION(LONGDOUBLE, npy_longdouble)
DEFINE_VECMAT_FUNCTION(HALF, npy_half)
DEFINE_VECMAT_FUNCTION(CFLOAT, npy_cfloat)
DEFINE_VECMAT_FUNCTION(CDOUBLE, npy_cdouble)
DEFINE_VECMAT_FUNCTION(CLONGDOUBLE, npy_clongdouble)
DEFINE_VECMAT_FUNCTION(UBYTE, npy_ubyte)
DEFINE_VECMAT_FUNCTION(USHORT, npy_ushort)
DEFINE_VECMAT_FUNCTION(UINT, npy_uint)
DEFINE_VECMAT_FUNCTION(ULONG, npy_ulong)
DEFINE_VECMAT_FUNCTION(ULONGLONG, npy_ulonglong)
DEFINE_VECMAT_FUNCTION(BYTE, npy_byte)
DEFINE_VECMAT_FUNCTION(SHORT, npy_short)
DEFINE_VECMAT_FUNCTION(INT, npy_int)
DEFINE_VECMAT_FUNCTION(LONG, npy_long)
DEFINE_VECMAT_FUNCTION(LONGLONG, npy_longlong)
DEFINE_VECMAT_FUNCTION(BOOL, npy_bool)
DEFINE_VECMAT_FUNCTION(OBJECT, PyObject*)

#undef DEFINE_VECMAT_FUNCTION
