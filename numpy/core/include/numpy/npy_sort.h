#ifndef __NPY_SORT_H__
#define __NPY_SORT_H__

/* Python include is for future object sorts */
#include <Python.h>
#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>


int BOOL_quicksort(npy_bool *start, npy_intp num, void *NOT_USED);
int BOOL_heapsort(npy_bool *start, npy_intp n, void *NOT_USED);
int BOOL_mergesort(npy_bool *start, npy_intp num, void *NOT_USED);
int BOOL_aquicksort(npy_bool *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int BOOL_aheapsort(npy_bool *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int BOOL_amergesort(npy_bool *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int BYTE_quicksort(npy_byte *start, npy_intp num, void *NOT_USED);
int BYTE_heapsort(npy_byte *start, npy_intp n, void *NOT_USED);
int BYTE_mergesort(npy_byte *start, npy_intp num, void *NOT_USED);
int BYTE_aquicksort(npy_byte *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int BYTE_aheapsort(npy_byte *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int BYTE_amergesort(npy_byte *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int UBYTE_quicksort(npy_ubyte *start, npy_intp num, void *NOT_USED);
int UBYTE_heapsort(npy_ubyte *start, npy_intp n, void *NOT_USED);
int UBYTE_mergesort(npy_ubyte *start, npy_intp num, void *NOT_USED);
int UBYTE_aquicksort(npy_ubyte *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int UBYTE_aheapsort(npy_ubyte *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int UBYTE_amergesort(npy_ubyte *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int SHORT_quicksort(npy_short *start, npy_intp num, void *NOT_USED);
int SHORT_heapsort(npy_short *start, npy_intp n, void *NOT_USED);
int SHORT_mergesort(npy_short *start, npy_intp num, void *NOT_USED);
int SHORT_aquicksort(npy_short *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int SHORT_aheapsort(npy_short *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int SHORT_amergesort(npy_short *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int USHORT_quicksort(npy_ushort *start, npy_intp num, void *NOT_USED);
int USHORT_heapsort(npy_ushort *start, npy_intp n, void *NOT_USED);
int USHORT_mergesort(npy_ushort *start, npy_intp num, void *NOT_USED);
int USHORT_aquicksort(npy_ushort *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int USHORT_aheapsort(npy_ushort *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int USHORT_amergesort(npy_ushort *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int INT_quicksort(npy_int *start, npy_intp num, void *NOT_USED);
int INT_heapsort(npy_int *start, npy_intp n, void *NOT_USED);
int INT_mergesort(npy_int *start, npy_intp num, void *NOT_USED);
int INT_aquicksort(npy_int *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int INT_aheapsort(npy_int *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int INT_amergesort(npy_int *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int UINT_quicksort(npy_uint *start, npy_intp num, void *NOT_USED);
int UINT_heapsort(npy_uint *start, npy_intp n, void *NOT_USED);
int UINT_mergesort(npy_uint *start, npy_intp num, void *NOT_USED);
int UINT_aquicksort(npy_uint *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int UINT_aheapsort(npy_uint *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int UINT_amergesort(npy_uint *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int LONG_quicksort(npy_long *start, npy_intp num, void *NOT_USED);
int LONG_heapsort(npy_long *start, npy_intp n, void *NOT_USED);
int LONG_mergesort(npy_long *start, npy_intp num, void *NOT_USED);
int LONG_aquicksort(npy_long *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int LONG_aheapsort(npy_long *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int LONG_amergesort(npy_long *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int ULONG_quicksort(npy_ulong *start, npy_intp num, void *NOT_USED);
int ULONG_heapsort(npy_ulong *start, npy_intp n, void *NOT_USED);
int ULONG_mergesort(npy_ulong *start, npy_intp num, void *NOT_USED);
int ULONG_aquicksort(npy_ulong *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int ULONG_aheapsort(npy_ulong *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int ULONG_amergesort(npy_ulong *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int LONGLONG_aquicksort(npy_longlong *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int LONGLONG_aheapsort(npy_longlong *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int LONGLONG_amergesort(npy_longlong *v, npy_intp *tosort, npy_intp num, void *NOT_USED);
int LONGLONG_quicksort(npy_longlong *start, npy_intp num, void *NOT_USED);
int LONGLONG_heapsort(npy_longlong *start, npy_intp n, void *NOT_USED);
int LONGLONG_mergesort(npy_longlong *start, npy_intp num, void *NOT_USED);


int ULONGLONG_quicksort(npy_ulonglong *start, npy_intp num, void *NOT_USED);
int ULONGLONG_heapsort(npy_ulonglong *start, npy_intp n, void *NOT_USED);
int ULONGLONG_mergesort(npy_ulonglong *start, npy_intp num, void *NOT_USED);
int ULONGLONG_aquicksort(npy_ulonglong *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int ULONGLONG_aheapsort(npy_ulonglong *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int ULONGLONG_amergesort(npy_ulonglong *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int HALF_quicksort(npy_ushort *start, npy_intp num, void *NOT_USED);
int HALF_heapsort(npy_ushort *start, npy_intp n, void *NOT_USED);
int HALF_mergesort(npy_ushort *start, npy_intp num, void *NOT_USED);
int HALF_aquicksort(npy_ushort *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int HALF_aheapsort(npy_ushort *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int HALF_amergesort(npy_ushort *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int FLOAT_quicksort(npy_float *start, npy_intp num, void *NOT_USED);
int FLOAT_heapsort(npy_float *start, npy_intp n, void *NOT_USED);
int FLOAT_mergesort(npy_float *start, npy_intp num, void *NOT_USED);
int FLOAT_aquicksort(npy_float *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int FLOAT_aheapsort(npy_float *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int FLOAT_amergesort(npy_float *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int DOUBLE_quicksort(npy_double *start, npy_intp num, void *NOT_USED);
int DOUBLE_heapsort(npy_double *start, npy_intp n, void *NOT_USED);
int DOUBLE_mergesort(npy_double *start, npy_intp num, void *NOT_USED);
int DOUBLE_aquicksort(npy_double *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int DOUBLE_aheapsort(npy_double *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int DOUBLE_amergesort(npy_double *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int LONGDOUBLE_quicksort(npy_longdouble *start, npy_intp num, void *NOT_USED);
int LONGDOUBLE_heapsort(npy_longdouble *start, npy_intp n, void *NOT_USED);
int LONGDOUBLE_mergesort(npy_longdouble *start, npy_intp num, void *NOT_USED);
int LONGDOUBLE_aquicksort(npy_longdouble *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int LONGDOUBLE_aheapsort(npy_longdouble *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int LONGDOUBLE_amergesort(npy_longdouble *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int CFLOAT_quicksort(npy_cfloat *start, npy_intp num, void *NOT_USED);
int CFLOAT_heapsort(npy_cfloat *start, npy_intp n, void *NOT_USED);
int CFLOAT_mergesort(npy_cfloat *start, npy_intp num, void *NOT_USED);
int CFLOAT_aquicksort(npy_cfloat *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int CFLOAT_aheapsort(npy_cfloat *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int CFLOAT_amergesort(npy_cfloat *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int CDOUBLE_quicksort(npy_cdouble *start, npy_intp num, void *NOT_USED);
int CDOUBLE_heapsort(npy_cdouble *start, npy_intp n, void *NOT_USED);
int CDOUBLE_mergesort(npy_cdouble *start, npy_intp num, void *NOT_USED);
int CDOUBLE_aquicksort(npy_cdouble *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int CDOUBLE_aheapsort(npy_cdouble *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int CDOUBLE_amergesort(npy_cdouble *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int CLONGDOUBLE_quicksort(npy_clongdouble *start, npy_intp num, void *NOT_USED);
int CLONGDOUBLE_heapsort(npy_clongdouble *start, npy_intp n, void *NOT_USED);
int CLONGDOUBLE_mergesort(npy_clongdouble *start, npy_intp num, void *NOT_USED);
int CLONGDOUBLE_aquicksort(npy_clongdouble *v, npy_intp* tosort, npy_intp num, void *NOT_USED);
int CLONGDOUBLE_aheapsort(npy_clongdouble *v, npy_intp *tosort, npy_intp n, void *NOT_USED);
int CLONGDOUBLE_amergesort(npy_clongdouble *v, npy_intp *tosort, npy_intp num, void *NOT_USED);


int STRING_mergesort(char *start, npy_intp num, PyArrayObject *arr);
int STRING_quicksort(char *start, npy_intp num, PyArrayObject *arr);
int STRING_heapsort(char *start, npy_intp n, PyArrayObject *arr);
int STRING_aheapsort(char *v, npy_intp *tosort, npy_intp n, PyArrayObject *arr);
int STRING_aquicksort(char *v, npy_intp* tosort, npy_intp num, PyArrayObject *arr);
int STRING_amergesort(char *v, npy_intp *tosort, npy_intp num, PyArrayObject *arr);


int UNICODE_mergesort(npy_ucs4 *start, npy_intp num, PyArrayObject *arr);
int UNICODE_quicksort(npy_ucs4 *start, npy_intp num, PyArrayObject *arr);
int UNICODE_heapsort(npy_ucs4 *start, npy_intp n, PyArrayObject *arr);
int UNICODE_aheapsort(npy_ucs4 *v, npy_intp *tosort, npy_intp n, PyArrayObject *arr);
int UNICODE_aquicksort(npy_ucs4 *v, npy_intp* tosort, npy_intp num, PyArrayObject *arr);
int UNICODE_amergesort(npy_ucs4 *v, npy_intp *tosort, npy_intp num, PyArrayObject *arr);

#endif
