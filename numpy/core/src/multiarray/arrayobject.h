#ifndef _NPY_INTERNAL_ARRAYOBJECT_H_
#define _NPY_INTERNAL_ARRAYOBJECT_H_

#ifndef _MULTIARRAYMODULE
#error You should not include this
#endif

extern NPY_NO_EXPORT PyArray_Descr **userdescrs;


#define error_converting(x)  (((x) == -1) && PyErr_Occurred())

#define SOBJ_NOTFANCY 0
#define SOBJ_ISFANCY 1
#define SOBJ_BADARRAY 2
#define SOBJ_TOOMANY 3
#define SOBJ_LISTTUP 4

NPY_NO_EXPORT int
_flat_copyinto(PyObject *dst, PyObject *src, NPY_ORDER order);

NPY_NO_EXPORT PyArray_Descr *
_array_small_type(PyArray_Descr *chktype, PyArray_Descr* mintype);

NPY_NO_EXPORT PyObject *
array_big_item(PyArrayObject *, intp);

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
/* FIXME: just remove _check_axis ? */
#define _check_axis PyArray_CheckAxis

NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out);

NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out);

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyArrayObject *m1, PyObject *m2, PyObject *op);

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
_array_find_type(PyObject *op, PyArray_Descr *minitype, int max);

NPY_NO_EXPORT PyArray_Descr *
_array_find_python_scalar_type(PyObject *op);

NPY_NO_EXPORT PyArray_Descr *
_array_typedescr_fromstr(char *str);

/* FIXME: this is defined in multiarraymodule.c ... */

NPY_NO_EXPORT PyArray_Descr *
_arraydescr_fromobj(PyObject *obj);

/* FIXME: this is in scalartypes.inc.src */
NPY_NO_EXPORT void
initialize_numeric_types(void);

NPY_NO_EXPORT void
format_longdouble(char *buf, size_t buflen, longdouble val, unsigned int prec);

NPY_NO_EXPORT void
gentype_struct_free(void *ptr, void *arg);

NPY_NO_EXPORT int
_typenum_fromtypeobj(PyObject *type, int user);

NPY_NO_EXPORT void *
scalar_value(PyObject *scalar, PyArray_Descr *descr);

/* FIXME: this is defined in arratypes.inc.src */
NPY_NO_EXPORT int
set_typeinfo(PyObject *dict);

extern NPY_NO_EXPORT PyArray_Descr LONG_Descr;
extern NPY_NO_EXPORT PyArray_Descr INT_Descr;

/* Number protocol */
#include "number.h"

/* Converting data types API */
#include "convert_datatype.h"

/* Object Conversion API */
#include "convert.h"

#endif
