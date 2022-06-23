
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_ROWS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_ROWS_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

#include "textreading/stream.h"
#include "textreading/field_types.h"
#include "textreading/parser_config.h"


NPY_NO_EXPORT PyArrayObject *
read_rows(stream *s, npy_intp nrows, parser_config *pconfig,
          PyObject *usecols_obj, Py_ssize_t skiplines, PyObject *converters,
          PyArrayObject *data_array, PyObject *dtype);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_ROWS_H_ */
