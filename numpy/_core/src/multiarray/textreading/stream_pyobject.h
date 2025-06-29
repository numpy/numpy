
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_PYOBJECT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_PYOBJECT_H_

/* Any file that includes Python.h must include it before any other files */
/* https://docs.python.org/3/extending/extending.html#a-simple-example */
/* npy_common.h includes Python.h so it also counts in this list */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "textreading/stream.h"

NPY_NO_EXPORT stream *
stream_python_file(PyObject *obj, const char *encoding);

NPY_NO_EXPORT stream *
stream_python_iterable(PyObject *obj, const char *encoding);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_PYOBJECT_H_ */
