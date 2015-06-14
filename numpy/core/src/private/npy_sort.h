#ifndef __NPY_SORT_H__
#define __NPY_SORT_H__

/* Python include is for future object sorts */
#include <Python.h>
#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

#define NPY_ENOMEM 1
#define NPY_ECOMP 2


int quicksort_bool(void *vec, npy_intp cnt, void *null);
int heapsort_bool(void *vec, npy_intp cnt, void *null);
int mergesort_bool(void *vec, npy_intp cnt, void *null);
int aquicksort_bool(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_bool(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_bool(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_byte(void *vec, npy_intp cnt, void *null);
int heapsort_byte(void *vec, npy_intp cnt, void *null);
int mergesort_byte(void *vec, npy_intp cnt, void *null);
int aquicksort_byte(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_byte(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_byte(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_ubyte(void *vec, npy_intp cnt, void *null);
int heapsort_ubyte(void *vec, npy_intp cnt, void *null);
int mergesort_ubyte(void *vec, npy_intp cnt, void *null);
int aquicksort_ubyte(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_ubyte(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_ubyte(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_short(void *vec, npy_intp cnt, void *null);
int heapsort_short(void *vec, npy_intp cnt, void *null);
int mergesort_short(void *vec, npy_intp cnt, void *null);
int aquicksort_short(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_short(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_short(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_ushort(void *vec, npy_intp cnt, void *null);
int heapsort_ushort(void *vec, npy_intp cnt, void *null);
int mergesort_ushort(void *vec, npy_intp cnt, void *null);
int aquicksort_ushort(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_ushort(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_ushort(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_int(void *vec, npy_intp cnt, void *null);
int heapsort_int(void *vec, npy_intp cnt, void *null);
int mergesort_int(void *vec, npy_intp cnt, void *null);
int aquicksort_int(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_int(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_int(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_uint(void *vec, npy_intp cnt, void *null);
int heapsort_uint(void *vec, npy_intp cnt, void *null);
int mergesort_uint(void *vec, npy_intp cnt, void *null);
int aquicksort_uint(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_uint(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_uint(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_long(void *vec, npy_intp cnt, void *null);
int heapsort_long(void *vec, npy_intp cnt, void *null);
int mergesort_long(void *vec, npy_intp cnt, void *null);
int aquicksort_long(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_long(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_long(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_ulong(void *vec, npy_intp cnt, void *null);
int heapsort_ulong(void *vec, npy_intp cnt, void *null);
int mergesort_ulong(void *vec, npy_intp cnt, void *null);
int aquicksort_ulong(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_ulong(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_ulong(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_longlong(void *vec, npy_intp cnt, void *null);
int heapsort_longlong(void *vec, npy_intp cnt, void *null);
int mergesort_longlong(void *vec, npy_intp cnt, void *null);
int aquicksort_longlong(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_longlong(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_longlong(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_ulonglong(void *vec, npy_intp cnt, void *null);
int heapsort_ulonglong(void *vec, npy_intp cnt, void *null);
int mergesort_ulonglong(void *vec, npy_intp cnt, void *null);
int aquicksort_ulonglong(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_ulonglong(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_ulonglong(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_half(void *vec, npy_intp cnt, void *null);
int heapsort_half(void *vec, npy_intp cnt, void *null);
int mergesort_half(void *vec, npy_intp cnt, void *null);
int aquicksort_half(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_half(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_half(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_float(void *vec, npy_intp cnt, void *null);
int heapsort_float(void *vec, npy_intp cnt, void *null);
int mergesort_float(void *vec, npy_intp cnt, void *null);
int aquicksort_float(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_float(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_float(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_double(void *vec, npy_intp cnt, void *null);
int heapsort_double(void *vec, npy_intp cnt, void *null);
int mergesort_double(void *vec, npy_intp cnt, void *null);
int aquicksort_double(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_double(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_double(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_longdouble(void *vec, npy_intp cnt, void *null);
int heapsort_longdouble(void *vec, npy_intp cnt, void *null);
int mergesort_longdouble(void *vec, npy_intp cnt, void *null);
int aquicksort_longdouble(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_longdouble(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_longdouble(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_cfloat(void *vec, npy_intp cnt, void *null);
int heapsort_cfloat(void *vec, npy_intp cnt, void *null);
int mergesort_cfloat(void *vec, npy_intp cnt, void *null);
int aquicksort_cfloat(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_cfloat(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_cfloat(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_cdouble(void *vec, npy_intp cnt, void *null);
int heapsort_cdouble(void *vec, npy_intp cnt, void *null);
int mergesort_cdouble(void *vec, npy_intp cnt, void *null);
int aquicksort_cdouble(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_cdouble(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_cdouble(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_clongdouble(void *vec, npy_intp cnt, void *null);
int heapsort_clongdouble(void *vec, npy_intp cnt, void *null);
int mergesort_clongdouble(void *vec, npy_intp cnt, void *null);
int aquicksort_clongdouble(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_clongdouble(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_clongdouble(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_string(void *vec, npy_intp cnt, void *arr);
int heapsort_string(void *vec, npy_intp cnt, void *arr);
int mergesort_string(void *vec, npy_intp cnt, void *arr);
int aquicksort_string(void *vec, npy_intp *ind, npy_intp cnt, void *arr);
int aheapsort_string(void *vec, npy_intp *ind, npy_intp cnt, void *arr);
int amergesort_string(void *vec, npy_intp *ind, npy_intp cnt, void *arr);


int quicksort_unicode(void *vec, npy_intp cnt, void *arr);
int heapsort_unicode(void *vec, npy_intp cnt, void *arr);
int mergesort_unicode(void *vec, npy_intp cnt, void *arr);
int aquicksort_unicode(void *vec, npy_intp *ind, npy_intp cnt, void *arr);
int aheapsort_unicode(void *vec, npy_intp *ind, npy_intp cnt, void *arr);
int amergesort_unicode(void *vec, npy_intp *ind, npy_intp cnt, void *arr);


int quicksort_datetime(void *vec, npy_intp cnt, void *null);
int heapsort_datetime(void *vec, npy_intp cnt, void *null);
int mergesort_datetime(void *vec, npy_intp cnt, void *null);
int aquicksort_datetime(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_datetime(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_datetime(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_timedelta(void *vec, npy_intp cnt, void *null);
int heapsort_timedelta(void *vec, npy_intp cnt, void *null);
int mergesort_timedelta(void *vec, npy_intp cnt, void *null);
int aquicksort_timedelta(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_timedelta(void *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_timedelta(void *vec, npy_intp *ind, npy_intp cnt, void *null);


int npy_quicksort(void *vec, npy_intp cnt, void *arr);
int npy_heapsort(void *vec, npy_intp cnt, void *arr);
int npy_mergesort(void *vec, npy_intp cnt, void *arr);
int npy_aquicksort(void *vec, npy_intp *ind, npy_intp cnt, void *arr);
int npy_aheapsort(void *vec, npy_intp *ind, npy_intp cnt, void *arr);
int npy_amergesort(void *vec, npy_intp *ind, npy_intp cnt, void *arr);

#endif
