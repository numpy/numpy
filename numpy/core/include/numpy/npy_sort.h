#ifndef __NPY_SORT_H__
#define __NPY_SORT_H__

/* Python include is for future object sorts */
#include <Python.h>
#include <numpy/npy_common.h>


int BOOL_aquicksort(npy_bool *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int BOOL_aheapsort(npy_bool *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void BOOL_amergesort0(npy_intp *pl, npy_intp *pr, npy_bool *v, npy_intp *pw);
int BOOL_amergesort(npy_bool *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int BYTE_aquicksort(npy_byte *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int BYTE_aheapsort(npy_byte *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void BYTE_amergesort0(npy_intp *pl, npy_intp *pr, npy_byte *v, npy_intp *pw);
int BYTE_amergesort(npy_byte *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int UBYTE_aquicksort(npy_ubyte *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int UBYTE_aheapsort(npy_ubyte *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void UBYTE_amergesort0(npy_intp *pl, npy_intp *pr, npy_ubyte *v, npy_intp *pw);
int UBYTE_amergesort(npy_ubyte *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int SHORT_aquicksort(npy_short *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int SHORT_aheapsort(npy_short *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void SHORT_amergesort0(npy_intp *pl, npy_intp *pr, npy_short *v, npy_intp *pw);
int SHORT_amergesort(npy_short *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int USHORT_aquicksort(npy_ushort *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int USHORT_aheapsort(npy_ushort *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void USHORT_amergesort0(npy_intp *pl, npy_intp *pr, npy_ushort *v, npy_intp *pw);
int USHORT_amergesort(npy_ushort *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int INT_aquicksort(npy_int *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int INT_aheapsort(npy_int *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void INT_amergesort0(npy_intp *pl, npy_intp *pr, npy_int *v, npy_intp *pw);
int INT_amergesort(npy_int *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int UINT_aquicksort(npy_uint *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int UINT_aheapsort(npy_uint *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void UINT_amergesort0(npy_intp *pl, npy_intp *pr, npy_uint *v, npy_intp *pw);
int UINT_amergesort(npy_uint *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int LONG_aquicksort(npy_long *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int LONG_aheapsort(npy_long *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void LONG_amergesort0(npy_intp *pl, npy_intp *pr, npy_long *v, npy_intp *pw);
int LONG_amergesort(npy_long *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int ULONG_aquicksort(npy_ulong *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int ULONG_aheapsort(npy_ulong *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void ULONG_amergesort0(npy_intp *pl, npy_intp *pr, npy_ulong *v, npy_intp *pw);
int ULONG_amergesort(npy_ulong *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int LONGLONG_aquicksort(npy_longlong *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int LONGLONG_aheapsort(npy_longlong *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void LONGLONG_amergesort0(npy_intp *pl, npy_intp *pr, npy_longlong *v, npy_intp *pw);
int LONGLONG_amergesort(npy_longlong *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int ULONGLONG_aquicksort(npy_ulonglong *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int ULONGLONG_aheapsort(npy_ulonglong *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void ULONGLONG_amergesort0(npy_intp *pl, npy_intp *pr, npy_ulonglong *v, npy_intp *pw);
int ULONGLONG_amergesort(npy_ulonglong *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int HALF_aquicksort(npy_ushort *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int HALF_aheapsort(npy_ushort *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void HALF_amergesort0(npy_intp *pl, npy_intp *pr, npy_ushort *v, npy_intp *pw);
int HALF_amergesort(npy_ushort *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int FLOAT_aquicksort(npy_float *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int FLOAT_aheapsort(npy_float *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void FLOAT_amergesort0(npy_intp *pl, npy_intp *pr, npy_float *v, npy_intp *pw);
int FLOAT_amergesort(npy_float *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int DOUBLE_aquicksort(npy_double *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int DOUBLE_aheapsort(npy_double *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void DOUBLE_amergesort0(npy_intp *pl, npy_intp *pr, npy_double *v, npy_intp *pw);
int DOUBLE_amergesort(npy_double *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int LONGDOUBLE_aquicksort(npy_longdouble *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int LONGDOUBLE_aheapsort(npy_longdouble *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void LONGDOUBLE_amergesort0(npy_intp *pl, npy_intp *pr, npy_longdouble *v, npy_intp *pw);
int LONGDOUBLE_amergesort(npy_longdouble *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int CFLOAT_aquicksort(npy_cfloat *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int CFLOAT_aheapsort(npy_cfloat *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void CFLOAT_amergesort0(npy_intp *pl, npy_intp *pr, npy_cfloat *v, npy_intp *pw);
int CFLOAT_amergesort(npy_cfloat *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int CDOUBLE_aquicksort(npy_cdouble *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int CDOUBLE_aheapsort(npy_cdouble *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void CDOUBLE_amergesort0(npy_intp *pl, npy_intp *pr, npy_cdouble *v, npy_intp *pw);
int CDOUBLE_amergesort(npy_cdouble *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int CLONGDOUBLE_aquicksort(npy_clongdouble *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int CLONGDOUBLE_aheapsort(npy_clongdouble *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
void CLONGDOUBLE_amergesort0(npy_intp *pl, npy_intp *pr, npy_clongdouble *v, npy_intp *pw);
int CLONGDOUBLE_amergesort(npy_clongdouble *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int STRING_aheapsort(char *v, npy_intp *tosort, npy_intp n, PyArrayObject *arr);
int STRING_aquicksort(char *v, npy_intp* tosort, npy_intp num, PyArrayObject *arr);
void STRING_amergesort0(npy_intp *pl, npy_intp *pr, char *v, npy_intp *pw, int len);
int STRING_amergesort(char *v, npy_intp *tosort, npy_intp num, PyArrayObject *arr);


int UNICODE_aheapsort(npy_ucs4 *v, npy_intp *tosort, npy_intp n, PyArrayObject *arr);
int UNICODE_aquicksort(npy_ucs4 *v, npy_intp* tosort, npy_intp num, PyArrayObject *arr);
void UNICODE_amergesort0(npy_intp *pl, npy_intp *pr, npy_ucs4 *v, npy_intp *pw, int len);
int UNICODE_amergesort(npy_ucs4 *v, npy_intp *tosort, npy_intp num, PyArrayObject *arr);

#endif
