#ifndef _NPY_ARRAYITERATORS_H_
#define _NPY_ARRAYITERATORS_H_

NPY_NO_EXPORT PyObject
*iter_subscript(PyArrayIterObject *, PyObject *);

NPY_NO_EXPORT int
iter_ass_subscript(PyArrayIterObject *, PyObject *, PyObject *);

#endif
