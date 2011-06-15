#ifndef __NPY_SORT_H__
#define __NPY_SORT_H__

/* Python include is for future object sorts */
#include <Python.h>
#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>


int quicksort_npy_bool(npy_bool *vec, npy_intp cnt, void *null);
int heapsort_npy_bool(npy_bool *vec, npy_intp cnt, void *null);
int mergesort_npy_bool(npy_bool *vec, npy_intp cnt, void *null);
int aquicksort_npy_bool(npy_bool *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_bool(npy_bool *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_bool(npy_bool *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_byte(npy_byte *vec, npy_intp cnt, void *null);
int heapsort_npy_byte(npy_byte *vec, npy_intp cnt, void *null);
int mergesort_npy_byte(npy_byte *vec, npy_intp cnt, void *null);
int aquicksort_npy_byte(npy_byte *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_byte(npy_byte *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_byte(npy_byte *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_ubyte(npy_ubyte *vec, npy_intp cnt, void *null);
int heapsort_npy_ubyte(npy_ubyte *vec, npy_intp cnt, void *null);
int mergesort_npy_ubyte(npy_ubyte *vec, npy_intp cnt, void *null);
int aquicksort_npy_ubyte(npy_ubyte *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_ubyte(npy_ubyte *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_ubyte(npy_ubyte *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_short(npy_short *vec, npy_intp cnt, void *null);
int heapsort_npy_short(npy_short *vec, npy_intp cnt, void *null);
int mergesort_npy_short(npy_short *vec, npy_intp cnt, void *null);
int aquicksort_npy_short(npy_short *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_short(npy_short *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_short(npy_short *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_ushort(npy_ushort *vec, npy_intp cnt, void *null);
int heapsort_npy_ushort(npy_ushort *vec, npy_intp cnt, void *null);
int mergesort_npy_ushort(npy_ushort *vec, npy_intp cnt, void *null);
int aquicksort_npy_ushort(npy_ushort *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_ushort(npy_ushort *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_ushort(npy_ushort *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_int(npy_int *vec, npy_intp cnt, void *null);
int heapsort_npy_int(npy_int *vec, npy_intp cnt, void *null);
int mergesort_npy_int(npy_int *vec, npy_intp cnt, void *null);
int aquicksort_npy_int(npy_int *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_int(npy_int *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_int(npy_int *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_uint(npy_uint *vec, npy_intp cnt, void *null);
int heapsort_npy_uint(npy_uint *vec, npy_intp cnt, void *null);
int mergesort_npy_uint(npy_uint *vec, npy_intp cnt, void *null);
int aquicksort_npy_uint(npy_uint *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_uint(npy_uint *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_uint(npy_uint *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_long(npy_long *vec, npy_intp cnt, void *null);
int heapsort_npy_long(npy_long *vec, npy_intp cnt, void *null);
int mergesort_npy_long(npy_long *vec, npy_intp cnt, void *null);
int aquicksort_npy_long(npy_long *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_long(npy_long *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_long(npy_long *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_ulong(npy_ulong *vec, npy_intp cnt, void *null);
int heapsort_npy_ulong(npy_ulong *vec, npy_intp cnt, void *null);
int mergesort_npy_ulong(npy_ulong *vec, npy_intp cnt, void *null);
int aquicksort_npy_ulong(npy_ulong *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_ulong(npy_ulong *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_ulong(npy_ulong *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_longlong(npy_longlong *vec, npy_intp cnt, void *null);
int heapsort_npy_longlong(npy_longlong *vec, npy_intp cnt, void *null);
int mergesort_npy_longlong(npy_longlong *vec, npy_intp cnt, void *null);
int aquicksort_npy_longlong(npy_longlong *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_longlong(npy_longlong *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_longlong(npy_longlong *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_ulonglong(npy_ulonglong *vec, npy_intp cnt, void *null);
int heapsort_npy_ulonglong(npy_ulonglong *vec, npy_intp cnt, void *null);
int mergesort_npy_ulonglong(npy_ulonglong *vec, npy_intp cnt, void *null);
int aquicksort_npy_ulonglong(npy_ulonglong *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_ulonglong(npy_ulonglong *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_ulonglong(npy_ulonglong *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_half(npy_ushort *vec, npy_intp cnt, void *null);
int heapsort_npy_half(npy_ushort *vec, npy_intp cnt, void *null);
int mergesort_npy_half(npy_ushort *vec, npy_intp cnt, void *null);
int aquicksort_npy_half(npy_ushort *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_half(npy_ushort *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_half(npy_ushort *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_float(npy_float *vec, npy_intp cnt, void *null);
int heapsort_npy_float(npy_float *vec, npy_intp cnt, void *null);
int mergesort_npy_float(npy_float *vec, npy_intp cnt, void *null);
int aquicksort_npy_float(npy_float *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_float(npy_float *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_float(npy_float *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_double(npy_double *vec, npy_intp cnt, void *null);
int heapsort_npy_double(npy_double *vec, npy_intp cnt, void *null);
int mergesort_npy_double(npy_double *vec, npy_intp cnt, void *null);
int aquicksort_npy_double(npy_double *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_double(npy_double *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_double(npy_double *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_longdouble(npy_longdouble *vec, npy_intp cnt, void *null);
int heapsort_npy_longdouble(npy_longdouble *vec, npy_intp cnt, void *null);
int mergesort_npy_longdouble(npy_longdouble *vec, npy_intp cnt, void *null);
int aquicksort_npy_longdouble(npy_longdouble *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_longdouble(npy_longdouble *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_longdouble(npy_longdouble *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_cfloat(npy_cfloat *vec, npy_intp cnt, void *null);
int heapsort_npy_cfloat(npy_cfloat *vec, npy_intp cnt, void *null);
int mergesort_npy_cfloat(npy_cfloat *vec, npy_intp cnt, void *null);
int aquicksort_npy_cfloat(npy_cfloat *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_cfloat(npy_cfloat *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_cfloat(npy_cfloat *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_cdouble(npy_cdouble *vec, npy_intp cnt, void *null);
int heapsort_npy_cdouble(npy_cdouble *vec, npy_intp cnt, void *null);
int mergesort_npy_cdouble(npy_cdouble *vec, npy_intp cnt, void *null);
int aquicksort_npy_cdouble(npy_cdouble *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_cdouble(npy_cdouble *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_cdouble(npy_cdouble *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_clongdouble(npy_clongdouble *vec, npy_intp cnt, void *null);
int heapsort_npy_clongdouble(npy_clongdouble *vec, npy_intp cnt, void *null);
int mergesort_npy_clongdouble(npy_clongdouble *vec, npy_intp cnt, void *null);
int aquicksort_npy_clongdouble(npy_clongdouble *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_npy_clongdouble(npy_clongdouble *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_npy_clongdouble(npy_clongdouble *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_npy_string(npy_char *vec, npy_intp cnt, PyArrayObject *arr);
int heapsort_npy_string(npy_char *vec, npy_intp cnt, PyArrayObject *arr);
int mergesort_npy_string(npy_char *vec, npy_intp cnt, PyArrayObject *arr);
int aquicksort_npy_string(npy_char *vec, npy_intp *ind, npy_intp cnt, PyArrayObject *arr);
int aheapsort_npy_string(npy_char *vec, npy_intp *ind, npy_intp cnt, PyArrayObject *arr);
int amergesort_npy_string(npy_char *vec, npy_intp *ind, npy_intp cnt, PyArrayObject *arr);


int quicksort_npy_unicode(npy_ucs4 *vec, npy_intp cnt, PyArrayObject *arr);
int heapsort_npy_unicode(npy_ucs4 *vec, npy_intp cnt, PyArrayObject *arr);
int mergesort_npy_unicode(npy_ucs4 *vec, npy_intp cnt, PyArrayObject *arr);
int aquicksort_npy_unicode(npy_ucs4 *vec, npy_intp *ind, npy_intp cnt, PyArrayObject *arr);
int aheapsort_npy_unicode(npy_ucs4 *vec, npy_intp *ind, npy_intp cnt, PyArrayObject *arr);
int amergesort_npy_unicode(npy_ucs4 *vec, npy_intp *ind, npy_intp cnt, PyArrayObject *arr);

#endif
