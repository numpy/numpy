
#ifndef _ROWS_H_
#define _ROWS_H_

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

#include "textreading/stream.h"
#include "textreading/field_types.h"
#include "textreading/parser_config.h"


PyArrayObject *
read_rows(stream *s,
        npy_intp nrows, int num_field_types, field_type *field_types,
        parser_config *pconfig, int num_usecols, int *usecols,
        Py_ssize_t skiplines, PyObject *converters,
        PyArrayObject *data_array, PyArray_Descr *out_descr,
        bool homogeneous);

#endif
