#ifndef __NPY_SORT_H__
#define __NPY_SORT_H__

/* Python include is for future object sorts */
#include <Python.h>
#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>

#define NPY_ENOMEM 1
#define NPY_ECOMP 2

typedef int (*npy_comparator)(const void *, const void *);

int quicksort_bool(npy_bool *vec, npy_intp cnt, void *null);
int heapsort_bool(npy_bool *vec, npy_intp cnt, void *null);
int mergesort_bool(npy_bool *vec, npy_intp cnt, void *null);
int aquicksort_bool(npy_bool *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_bool(npy_bool *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_bool(npy_bool *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_byte(npy_byte *vec, npy_intp cnt, void *null);
int heapsort_byte(npy_byte *vec, npy_intp cnt, void *null);
int mergesort_byte(npy_byte *vec, npy_intp cnt, void *null);
int aquicksort_byte(npy_byte *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_byte(npy_byte *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_byte(npy_byte *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_ubyte(npy_ubyte *vec, npy_intp cnt, void *null);
int heapsort_ubyte(npy_ubyte *vec, npy_intp cnt, void *null);
int mergesort_ubyte(npy_ubyte *vec, npy_intp cnt, void *null);
int aquicksort_ubyte(npy_ubyte *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_ubyte(npy_ubyte *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_ubyte(npy_ubyte *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_short(npy_short *vec, npy_intp cnt, void *null);
int heapsort_short(npy_short *vec, npy_intp cnt, void *null);
int mergesort_short(npy_short *vec, npy_intp cnt, void *null);
int aquicksort_short(npy_short *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_short(npy_short *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_short(npy_short *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_ushort(npy_ushort *vec, npy_intp cnt, void *null);
int heapsort_ushort(npy_ushort *vec, npy_intp cnt, void *null);
int mergesort_ushort(npy_ushort *vec, npy_intp cnt, void *null);
int aquicksort_ushort(npy_ushort *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_ushort(npy_ushort *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_ushort(npy_ushort *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_int(npy_int *vec, npy_intp cnt, void *null);
int heapsort_int(npy_int *vec, npy_intp cnt, void *null);
int mergesort_int(npy_int *vec, npy_intp cnt, void *null);
int aquicksort_int(npy_int *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_int(npy_int *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_int(npy_int *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_uint(npy_uint *vec, npy_intp cnt, void *null);
int heapsort_uint(npy_uint *vec, npy_intp cnt, void *null);
int mergesort_uint(npy_uint *vec, npy_intp cnt, void *null);
int aquicksort_uint(npy_uint *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_uint(npy_uint *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_uint(npy_uint *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_long(npy_long *vec, npy_intp cnt, void *null);
int heapsort_long(npy_long *vec, npy_intp cnt, void *null);
int mergesort_long(npy_long *vec, npy_intp cnt, void *null);
int aquicksort_long(npy_long *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_long(npy_long *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_long(npy_long *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_ulong(npy_ulong *vec, npy_intp cnt, void *null);
int heapsort_ulong(npy_ulong *vec, npy_intp cnt, void *null);
int mergesort_ulong(npy_ulong *vec, npy_intp cnt, void *null);
int aquicksort_ulong(npy_ulong *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_ulong(npy_ulong *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_ulong(npy_ulong *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_longlong(npy_longlong *vec, npy_intp cnt, void *null);
int heapsort_longlong(npy_longlong *vec, npy_intp cnt, void *null);
int mergesort_longlong(npy_longlong *vec, npy_intp cnt, void *null);
int aquicksort_longlong(npy_longlong *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_longlong(npy_longlong *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_longlong(npy_longlong *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_ulonglong(npy_ulonglong *vec, npy_intp cnt, void *null);
int heapsort_ulonglong(npy_ulonglong *vec, npy_intp cnt, void *null);
int mergesort_ulonglong(npy_ulonglong *vec, npy_intp cnt, void *null);
int aquicksort_ulonglong(npy_ulonglong *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_ulonglong(npy_ulonglong *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_ulonglong(npy_ulonglong *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_half(npy_ushort *vec, npy_intp cnt, void *null);
int heapsort_half(npy_ushort *vec, npy_intp cnt, void *null);
int mergesort_half(npy_ushort *vec, npy_intp cnt, void *null);
int aquicksort_half(npy_ushort *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_half(npy_ushort *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_half(npy_ushort *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_float(npy_float *vec, npy_intp cnt, void *null);
int heapsort_float(npy_float *vec, npy_intp cnt, void *null);
int mergesort_float(npy_float *vec, npy_intp cnt, void *null);
int aquicksort_float(npy_float *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_float(npy_float *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_float(npy_float *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_double(npy_double *vec, npy_intp cnt, void *null);
int heapsort_double(npy_double *vec, npy_intp cnt, void *null);
int mergesort_double(npy_double *vec, npy_intp cnt, void *null);
int aquicksort_double(npy_double *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_double(npy_double *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_double(npy_double *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_longdouble(npy_longdouble *vec, npy_intp cnt, void *null);
int heapsort_longdouble(npy_longdouble *vec, npy_intp cnt, void *null);
int mergesort_longdouble(npy_longdouble *vec, npy_intp cnt, void *null);
int aquicksort_longdouble(npy_longdouble *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_longdouble(npy_longdouble *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_longdouble(npy_longdouble *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_cfloat(npy_cfloat *vec, npy_intp cnt, void *null);
int heapsort_cfloat(npy_cfloat *vec, npy_intp cnt, void *null);
int mergesort_cfloat(npy_cfloat *vec, npy_intp cnt, void *null);
int aquicksort_cfloat(npy_cfloat *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_cfloat(npy_cfloat *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_cfloat(npy_cfloat *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_cdouble(npy_cdouble *vec, npy_intp cnt, void *null);
int heapsort_cdouble(npy_cdouble *vec, npy_intp cnt, void *null);
int mergesort_cdouble(npy_cdouble *vec, npy_intp cnt, void *null);
int aquicksort_cdouble(npy_cdouble *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_cdouble(npy_cdouble *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_cdouble(npy_cdouble *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_clongdouble(npy_clongdouble *vec, npy_intp cnt, void *null);
int heapsort_clongdouble(npy_clongdouble *vec, npy_intp cnt, void *null);
int mergesort_clongdouble(npy_clongdouble *vec, npy_intp cnt, void *null);
int aquicksort_clongdouble(npy_clongdouble *vec, npy_intp *ind, npy_intp cnt, void *null);
int aheapsort_clongdouble(npy_clongdouble *vec, npy_intp *ind, npy_intp cnt, void *null);
int amergesort_clongdouble(npy_clongdouble *vec, npy_intp *ind, npy_intp cnt, void *null);


int quicksort_string(npy_char *vec, npy_intp cnt, PyArrayObject *arr);
int heapsort_string(npy_char *vec, npy_intp cnt, PyArrayObject *arr);
int mergesort_string(npy_char *vec, npy_intp cnt, PyArrayObject *arr);
int aquicksort_string(npy_char *vec, npy_intp *ind, npy_intp cnt, PyArrayObject *arr);
int aheapsort_string(npy_char *vec, npy_intp *ind, npy_intp cnt, PyArrayObject *arr);
int amergesort_string(npy_char *vec, npy_intp *ind, npy_intp cnt, PyArrayObject *arr);


int quicksort_unicode(npy_ucs4 *vec, npy_intp cnt, PyArrayObject *arr);
int heapsort_unicode(npy_ucs4 *vec, npy_intp cnt, PyArrayObject *arr);
int mergesort_unicode(npy_ucs4 *vec, npy_intp cnt, PyArrayObject *arr);
int aquicksort_unicode(npy_ucs4 *vec, npy_intp *ind, npy_intp cnt, PyArrayObject *arr);
int aheapsort_unicode(npy_ucs4 *vec, npy_intp *ind, npy_intp cnt, PyArrayObject *arr);
int amergesort_unicode(npy_ucs4 *vec, npy_intp *ind, npy_intp cnt, PyArrayObject *arr);


int npy_quicksort(void *base, size_t num, size_t size, npy_comparator cmp);
int npy_heapsort(void *base, size_t num, size_t size, npy_comparator cmp);
int npy_mergesort(void *base, size_t num, size_t size, npy_comparator cmp);

#endif
