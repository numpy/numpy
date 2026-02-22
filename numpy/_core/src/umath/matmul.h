#ifndef _NPY_CORE_SRC_UMATH_MATMUL_H_
#define _NPY_CORE_SRC_UMATH_MATMUL_H_

#ifdef __cplusplus
extern "C" {
#endif

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_config.h"
#include "numpy/npy_common.h"
#include "numpy/arrayobject.h"


/***********************************************************************************
** Declaring matmul functions
***********************************************************************************/
#define DECLARE_MATMUL_FUNCTION(TYPE)                        \
NPY_NO_EXPORT void                                           \
TYPE##_matmul(char **args,                                   \
              npy_intp const *dimensions,                    \
              npy_intp const *steps,                         \
              void *NPY_UNUSED(func));

DECLARE_MATMUL_FUNCTION(FLOAT)
DECLARE_MATMUL_FUNCTION(DOUBLE)
DECLARE_MATMUL_FUNCTION(LONGDOUBLE)
DECLARE_MATMUL_FUNCTION(HALF)
DECLARE_MATMUL_FUNCTION(CFLOAT)
DECLARE_MATMUL_FUNCTION(CDOUBLE)
DECLARE_MATMUL_FUNCTION(CLONGDOUBLE)
DECLARE_MATMUL_FUNCTION(UBYTE)
DECLARE_MATMUL_FUNCTION(USHORT)
DECLARE_MATMUL_FUNCTION(UINT)
DECLARE_MATMUL_FUNCTION(ULONG)
DECLARE_MATMUL_FUNCTION(ULONGLONG)
DECLARE_MATMUL_FUNCTION(BYTE)
DECLARE_MATMUL_FUNCTION(SHORT)
DECLARE_MATMUL_FUNCTION(INT)
DECLARE_MATMUL_FUNCTION(LONG)
DECLARE_MATMUL_FUNCTION(LONGLONG)
DECLARE_MATMUL_FUNCTION(BOOL)
DECLARE_MATMUL_FUNCTION(OBJECT)

#undef DECLARE_MATMUL_FUNCTION

/***********************************************************************************
** Declaring vecdot functions
***********************************************************************************/
#define DECLARE_VECDOT_FUNCTION(TYPE)                        \
NPY_NO_EXPORT void                                           \
TYPE##_vecdot(char **args,                                   \
              npy_intp const *dimensions,                    \
              npy_intp const *steps,                         \
              void *NPY_UNUSED(func));

DECLARE_VECDOT_FUNCTION(FLOAT)
DECLARE_VECDOT_FUNCTION(DOUBLE)
DECLARE_VECDOT_FUNCTION(LONGDOUBLE)
DECLARE_VECDOT_FUNCTION(HALF)
DECLARE_VECDOT_FUNCTION(CFLOAT)
DECLARE_VECDOT_FUNCTION(CDOUBLE)
DECLARE_VECDOT_FUNCTION(CLONGDOUBLE)
DECLARE_VECDOT_FUNCTION(UBYTE)
DECLARE_VECDOT_FUNCTION(USHORT)
DECLARE_VECDOT_FUNCTION(UINT)
DECLARE_VECDOT_FUNCTION(ULONG)
DECLARE_VECDOT_FUNCTION(ULONGLONG)
DECLARE_VECDOT_FUNCTION(BYTE)
DECLARE_VECDOT_FUNCTION(SHORT)
DECLARE_VECDOT_FUNCTION(INT)
DECLARE_VECDOT_FUNCTION(LONG)
DECLARE_VECDOT_FUNCTION(LONGLONG)
DECLARE_VECDOT_FUNCTION(BOOL)
DECLARE_VECDOT_FUNCTION(OBJECT)

#undef DECLARE_VECDOT_FUNCTION

/***********************************************************************************
** Declaring matvec functions
***********************************************************************************/
#define DECLARE_MATVEC_FUNCTION(TYPE)                        \
NPY_NO_EXPORT void                                           \
TYPE##_matvec(char **args,                                   \
              npy_intp const *dimensions,                    \
              npy_intp const *steps,                         \
              void *NPY_UNUSED(func));

DECLARE_MATVEC_FUNCTION(FLOAT)
DECLARE_MATVEC_FUNCTION(DOUBLE)
DECLARE_MATVEC_FUNCTION(LONGDOUBLE)
DECLARE_MATVEC_FUNCTION(HALF)
DECLARE_MATVEC_FUNCTION(CFLOAT)
DECLARE_MATVEC_FUNCTION(CDOUBLE)
DECLARE_MATVEC_FUNCTION(CLONGDOUBLE)
DECLARE_MATVEC_FUNCTION(UBYTE)
DECLARE_MATVEC_FUNCTION(USHORT)
DECLARE_MATVEC_FUNCTION(UINT)
DECLARE_MATVEC_FUNCTION(ULONG)
DECLARE_MATVEC_FUNCTION(ULONGLONG)
DECLARE_MATVEC_FUNCTION(BYTE)
DECLARE_MATVEC_FUNCTION(SHORT)
DECLARE_MATVEC_FUNCTION(INT)
DECLARE_MATVEC_FUNCTION(LONG)
DECLARE_MATVEC_FUNCTION(LONGLONG)
DECLARE_MATVEC_FUNCTION(BOOL)
DECLARE_MATVEC_FUNCTION(OBJECT)

#undef DECLARE_MATVEC_FUNCTION

/***********************************************************************************
** Declaring vecmat functions
***********************************************************************************/
#define DECLARE_VECMAT_FUNCTION(TYPE)                        \
NPY_NO_EXPORT void                                           \
TYPE##_vecmat(char **args,                                   \
              npy_intp const *dimensions,                    \
              npy_intp const *steps,                         \
              void *NPY_UNUSED(func));

DECLARE_VECMAT_FUNCTION(FLOAT)
DECLARE_VECMAT_FUNCTION(DOUBLE)
DECLARE_VECMAT_FUNCTION(LONGDOUBLE)
DECLARE_VECMAT_FUNCTION(HALF)
DECLARE_VECMAT_FUNCTION(CFLOAT)
DECLARE_VECMAT_FUNCTION(CDOUBLE)
DECLARE_VECMAT_FUNCTION(CLONGDOUBLE)
DECLARE_VECMAT_FUNCTION(UBYTE)
DECLARE_VECMAT_FUNCTION(USHORT)
DECLARE_VECMAT_FUNCTION(UINT)
DECLARE_VECMAT_FUNCTION(ULONG)
DECLARE_VECMAT_FUNCTION(ULONGLONG)
DECLARE_VECMAT_FUNCTION(BYTE)
DECLARE_VECMAT_FUNCTION(SHORT)
DECLARE_VECMAT_FUNCTION(INT)
DECLARE_VECMAT_FUNCTION(LONG)
DECLARE_VECMAT_FUNCTION(LONGLONG)
DECLARE_VECMAT_FUNCTION(BOOL)
DECLARE_VECMAT_FUNCTION(OBJECT)

#undef DECLARE_VECMAT_FUNCTION

#ifdef __cplusplus
} // extern "C"
#endif

#endif
