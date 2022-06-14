
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_PYOBJECT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_PYOBJECT_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "textreading/stream.h"

NPY_NO_EXPORT stream *
stream_python_file(PyObject *obj, const char *encoding);

NPY_NO_EXPORT stream *
stream_python_iterable(PyObject *obj, const char *encoding);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_PYOBJECT_H_ */
