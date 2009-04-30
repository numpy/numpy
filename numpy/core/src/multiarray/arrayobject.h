#ifndef _NPY_INTERNAL_ARRAYOBJECT_H_
#define _NPY_INTERNAL_ARRAYOBJECT_H_

#ifndef _MULTIARRAYMODULE
#error You should not include this
#endif

extern NPY_NO_EXPORT PyArray_Descr **userdescrs;


#define SOBJ_NOTFANCY 0
#define SOBJ_ISFANCY 1
#define SOBJ_BADARRAY 2
#define SOBJ_TOOMANY 3
#define SOBJ_LISTTUP 4

NPY_NO_EXPORT void
_unaligned_strided_byte_copy(char *dst, intp outstrides, char *src,
                             intp instrides, intp N, int elsize);

NPY_NO_EXPORT void
_strided_byte_swap(void *p, intp stride, intp n, int size);

NPY_NO_EXPORT int
_zerofill(PyArrayObject *ret);

NPY_NO_EXPORT char *
index2ptr(PyArrayObject *mp, intp i);

NPY_NO_EXPORT PyObject *
_strings_richcompare(PyArrayObject *self, PyArrayObject *other, int cmp_op,
                     int rstrip);

NPY_NO_EXPORT void
copy_and_swap(void *dst, void *src, int itemsize, intp numitems,
              intp srcstrides, int swap);

NPY_NO_EXPORT void
byte_swap_vector(void *p, intp n, int size);

NPY_NO_EXPORT PyObject *
add_new_axes_0d(PyArrayObject *,  int);

NPY_NO_EXPORT int
count_new_axes_0d(PyObject *tuple);

NPY_NO_EXPORT PyObject *
array_richcompare(PyArrayObject *self, PyObject *other, int cmp_op);

NPY_NO_EXPORT PyObject *
array_subscript_simple(PyArrayObject *self, PyObject *op);

NPY_NO_EXPORT size_t
_array_fill_strides(intp *strides, intp *dims, int nd, size_t itemsize,
                    int inflag, int *objflags);

NPY_NO_EXPORT int
_IsAligned(PyArrayObject *ap);

NPY_NO_EXPORT Bool
_IsWriteable(PyArrayObject *ap);

NPY_NO_EXPORT PyArray_Descr *
_array_find_python_scalar_type(PyObject *op);

NPY_NO_EXPORT PyArray_Descr *
_array_typedescr_fromstr(char *str);

/* Number protocol */
#include "number.h"

/* Converting data types API */
#include "convert_datatype.h"

/* Object Conversion API */
#include "convert.h"

#endif
