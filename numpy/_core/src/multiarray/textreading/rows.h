
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_ROWS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_ROWS_H_

/* Any file that includes Python.h must include it before any other files */
/* https://docs.python.org/3/extending/extending.html#a-simple-example */
/* npy_common.h includes Python.h so it also counts in this list */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

#include "textreading/stream.h"
#include "textreading/field_types.h"
#include "textreading/parser_config.h"


NPY_NO_EXPORT PyArrayObject *
read_rows(stream *s,
        npy_intp nrows, Py_ssize_t num_field_types, field_type *field_types,
        parser_config *pconfig, Py_ssize_t num_usecols, Py_ssize_t *usecols,
        Py_ssize_t skiplines, PyObject *converters,
        PyArrayObject *data_array, PyArray_Descr *out_descr,
        bool homogeneous);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_ROWS_H_ */
