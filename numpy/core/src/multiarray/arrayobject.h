#ifndef _NPY_INTERNAL_ARRAYOBJECT_H_
#define _NPY_INTERNAL_ARRAYOBJECT_H_

#ifndef _MULTIARRAYMODULE
#error You should not include this
#endif

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

NPY_NO_EXPORT int
_IsAligned(PyArrayObject *ap);

NPY_NO_EXPORT Bool
_IsWriteable(PyArrayObject *ap);

#endif
