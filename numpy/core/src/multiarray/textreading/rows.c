
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"
#include "alloc.h"

#include <string.h>
#include <stdbool.h>

#include "textreading/stream.h"
#include "textreading/tokenize.h"
#include "textreading/conversions.h"
#include "textreading/field_types.h"
#include "textreading/rows.h"
#include "textreading/growth.h"

/*
 * Minimum size to grow the allocation by (or 25%). The 8KiB means the actual
 * growths is within `8 KiB <= size < 16 KiB` (depending on the row size).
 */
#define MIN_BLOCK_SIZE (1 << 13)



/*
 *  Create the array of converter functions from the Python converters.
 */
static PyObject **
create_conv_funcs(
        PyObject *converters, Py_ssize_t num_fields, const Py_ssize_t *usecols)
{
    assert(converters != Py_None);

    PyObject **conv_funcs = PyMem_Calloc(num_fields, sizeof(PyObject *));
    if (conv_funcs == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    if (PyCallable_Check(converters)) {
        /* a single converter used for all columns individually */
        for (Py_ssize_t i = 0; i < num_fields; i++) {
            Py_INCREF(converters);
            conv_funcs[i] = converters;
        }
        return conv_funcs;
    }
    else if (!PyDict_Check(converters)) {
        PyErr_SetString(PyExc_TypeError,
                "converters must be a dictionary mapping columns to converter "
                "functions or a single callable.");
        goto error;
    }

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(converters, &pos, &key, &value)) {
        Py_ssize_t column = PyNumber_AsSsize_t(key, PyExc_IndexError);
        if (column == -1 && PyErr_Occurred()) {
            PyErr_Format(PyExc_TypeError,
                    "keys of the converters dictionary must be integers; "
                    "got %.100R", key);
            goto error;
        }
        if (usecols != NULL) {
            /*
             * This code searches for the corresponding usecol.  It is
             * identical to the legacy usecols code, which has two weaknesses:
             * 1. It fails for duplicated usecols only setting converter for
             *    the first one.
             * 2. It fails e.g. if usecols uses negative indexing and
             *    converters does not.  (This is a feature, since it allows
             *    us to correctly normalize converters to result column here.)
             */
            Py_ssize_t i = 0;
            for (; i < num_fields; i++) {
                if (column == usecols[i]) {
                    column = i;
                    break;
                }
            }
            if (i == num_fields) {
                continue;  /* ignore unused converter */
            }
        }
        else {
            if (column < -num_fields || column >= num_fields) {
                PyErr_Format(PyExc_ValueError,
                        "converter specified for column %zd, which is invalid "
                        "for the number of fields %zd.", column, num_fields);
                goto error;
            }
            if (column < 0) {
                column += num_fields;
            }
        }
        if (!PyCallable_Check(value)) {
            PyErr_Format(PyExc_TypeError,
                    "values of the converters dictionary must be callable, "
                    "but the value associated with key %R is not", key);
            goto error;
        }
        Py_INCREF(value);
        conv_funcs[column] = value;
    }
    return conv_funcs;

  error:
    for (Py_ssize_t i = 0; i < num_fields; i++) {
        Py_XDECREF(conv_funcs[i]);
    }
    PyMem_FREE(conv_funcs);
    return NULL;
}

/**
 * Read a file into the provided array, or create (and possibly grow) an
 * array to read into.
 *
 * @param s The stream object/struct providing reading capabilities used by
 *        the tokenizer.
 * @param max_rows The number of rows to read, or -1.  If negative
 *        all rows are read.
 * @param num_field_types The number of field types stored in `field_types`.
 * @param field_types Information about the dtype for each column (or one if
 *        `homogeneous`).
 * @param pconfig Pointer to the parser config object used by both the
 *        tokenizer and the conversion functions.
 * @param num_usecols The number of columns in `usecols`.
 * @param usecols An array of length `num_usecols` or NULL.  If given indicates
 *        which column is read for each individual row (negative columns are
 *        accepted).
 * @param skiplines The number of lines to skip, these lines are ignored.
 * @param converters Python dictionary of converters.  Finalizing converters
 *        is difficult without information about the number of columns.
 * @param data_array An array to be filled or NULL.  In either case a new
 *        reference is returned (the reference to `data_array` is not stolen).
 * @param out_descr The dtype used for allocating a new array.  This is not
 *        used if `data_array` is provided.  Note that the actual dtype of the
 *        returned array can differ for strings.
 * @param num_cols Pointer in which the actual (discovered) number of columns
 *        is returned.  This is only relevant if `homogeneous` is true.
 * @param homogeneous Whether the datatype of the array is not homogeneous,
 *        i.e. not structured.  In this case the number of columns has to be
 *        discovered an the returned array will be 2-dimensional rather than
 *        1-dimensional.
 *
 * @returns Returns the result as an array object or NULL on error.  The result
 *          is always a new reference (even when `data_array` was passed in).
 */
NPY_NO_EXPORT PyArrayObject *
read_rows(stream *s,
        npy_intp max_rows, Py_ssize_t num_field_types, field_type *field_types,
        parser_config *pconfig, Py_ssize_t num_usecols, Py_ssize_t *usecols,
        Py_ssize_t skiplines, PyObject *converters,
        PyArrayObject *data_array, PyArray_Descr *out_descr,
        bool homogeneous)
{
    char *data_ptr = NULL;
    Py_ssize_t current_num_fields;
    npy_intp row_size = out_descr->elsize;
    PyObject **conv_funcs = NULL;

    bool needs_init = PyDataType_FLAGCHK(out_descr, NPY_NEEDS_INIT);

    int ndim = homogeneous ? 2 : 1;
    npy_intp result_shape[2] = {0, 1};

    bool data_array_allocated = data_array == NULL;
    /* Make sure we own `data_array` for the purpose of error handling */
    Py_XINCREF(data_array);
    size_t rows_per_block = 1;  /* will be increased depending on row size */
    npy_intp data_allocated_rows = 0;

    /* We give a warning if max_rows is used and an empty line is encountered */
    bool give_empty_row_warning = max_rows >= 0;

    int ts_result = 0;
    tokenizer_state ts;
    if (npy_tokenizer_init(&ts, pconfig) < 0) {
        goto error;
    }

    /* Set the actual number of fields if it is already known, otherwise -1 */
    Py_ssize_t actual_num_fields = -1;
    if (usecols != NULL) {
        assert(homogeneous || num_field_types == num_usecols);
        actual_num_fields = num_usecols;
    }
    else if (!homogeneous) {
        assert(usecols == NULL || num_field_types == num_usecols);
        actual_num_fields = num_field_types;
    }

    for (Py_ssize_t i = 0; i < skiplines; i++) {
        ts.state = TOKENIZE_GOTO_LINE_END;
        ts_result = npy_tokenize(s, &ts, pconfig);
        if (ts_result < 0) {
            goto error;
        }
        else if (ts_result != 0) {
            /* Fewer lines than skiplines is acceptable */
            break;
        }
    }

    Py_ssize_t row_count = 0;  /* number of rows actually processed */
    while ((max_rows < 0 || row_count < max_rows) && ts_result == 0) {
        ts_result = npy_tokenize(s, &ts, pconfig);
        if (ts_result < 0) {
            goto error;
        }
        current_num_fields = ts.num_fields;
        field_info *fields = ts.fields;
        if (NPY_UNLIKELY(ts.num_fields == 0)) {
            /*
             * Deprecated NumPy 1.23, 2021-01-13 (not really a deprecation,
             * but similar policy should apply to removing the warning again)
             */
             /* Tokenizer may give a final "empty line" even if there is none */
            if (give_empty_row_warning && ts_result == 0) {
                give_empty_row_warning = false;
                if (PyErr_WarnFormat(PyExc_UserWarning, 3,
                        "Input line %zd contained no data and will not be "
                        "counted towards `max_rows=%zd`.  This differs from "
                        "the behaviour in NumPy <=1.22 which counted lines "
                        "rather than rows.  If desired, the previous behaviour "
                        "can be achieved by using `itertools.islice`.\n"
                        "Please see the 1.23 release notes for an example on "
                        "how to do this.  If you wish to ignore this warning, "
                        "use `warnings.filterwarnings`.  This warning is "
                        "expected to be removed in the future and is given "
                        "only once per `loadtxt` call.",
                        row_count + skiplines + 1, max_rows) < 0) {
                    goto error;
                }
            }
            continue;  /* Ignore empty line */
        }

        if (NPY_UNLIKELY(data_ptr == NULL)) {
            // We've deferred some of the initialization tasks to here,
            // because we've now read the first line, and we definitively
            // know how many fields (i.e. columns) we will be processing.
            if (actual_num_fields == -1) {
                actual_num_fields = current_num_fields;
            }

            if (converters != Py_None) {
                conv_funcs = create_conv_funcs(
                        converters, actual_num_fields, usecols);
                if (conv_funcs == NULL) {
                    goto error;
                }
            }

            /* Note that result_shape[1] is only used if homogeneous is true */
            result_shape[1] = actual_num_fields;
            if (homogeneous) {
                row_size *= actual_num_fields;
            }

            if (data_array == NULL) {
                if (max_rows < 0) {
                    /*
                     * Negative max_rows denotes to read the whole file, we
                     * approach this by allocating ever larger blocks.
                     * Adds a number of rows based on `MIN_BLOCK_SIZE`.
                     * Note: later code grows assuming this is a power of two.
                     */
                    if (row_size == 0) {
                        /* actual rows_per_block should not matter here */
                        rows_per_block = 512;
                    }
                    else {
                        /* safe on overflow since min_rows will be 0 or 1 */
                        size_t min_rows = (
                                (MIN_BLOCK_SIZE + row_size - 1) / row_size);
                        while (rows_per_block < min_rows) {
                            rows_per_block *= 2;
                        }
                    }
                    data_allocated_rows = rows_per_block;
                }
                else {
                    data_allocated_rows = max_rows;
                }
                result_shape[0] = data_allocated_rows;
                Py_INCREF(out_descr);
                /*
                 * We do not use Empty, as it would fill with None
                 * and requiring decref'ing if we shrink again.
                 */
                data_array = (PyArrayObject *)PyArray_SimpleNewFromDescr(
                        ndim, result_shape, out_descr);
#ifdef NPY_RELAXED_STRIDES_DEBUG
                /* Incompatible with NPY_RELAXED_STRIDES_DEBUG due to growing */
                if (result_shape[0] == 1) {
                    PyArray_STRIDES(data_array)[0] = row_size;
                }
#endif /* NPY_RELAXED_STRIDES_DEBUG */
                if (data_array == NULL) {
                    goto error;
                }
                if (needs_init) {
                    memset(PyArray_BYTES(data_array), 0, PyArray_NBYTES(data_array));
                }
            }
            else {
                assert(max_rows >=0);
                data_allocated_rows = max_rows;
            }
            data_ptr = PyArray_BYTES(data_array);
        }

        if (!usecols && (actual_num_fields != current_num_fields)) {
            if (homogeneous) {
                PyErr_Format(PyExc_ValueError,
                    "the number of columns changed from %zd to %zd at row %zd; "
                    "use `usecols` to select a subset and avoid this error",
                    actual_num_fields, current_num_fields, row_count+1);
            }
            else {
                PyErr_Format(PyExc_ValueError,
                    "the dtype passed requires %zd columns but %zd were found "
                    "at row %zd; "
                    "use `usecols` to select a subset and avoid this error",
                    actual_num_fields, current_num_fields, row_count+1);
            }
            goto error;
        }

        if (NPY_UNLIKELY(data_allocated_rows == row_count)) {
            /*
             * Grow by ~25% and rounded up to the next rows_per_block
             * NOTE: This is based on very crude timings and could be refined!
             */
            npy_intp new_rows = data_allocated_rows;
            npy_intp alloc_size = grow_size_and_multiply(
                    &new_rows, rows_per_block, row_size);
            if (alloc_size < 0) {
                /* should normally error much earlier, but make sure */
                PyErr_SetString(PyExc_ValueError,
                        "array is too big. Cannot read file as a single array; "
                        "providing a maximum number of rows to read may help.");
                goto error;
            }

            char *new_data = PyDataMem_UserRENEW(
                    PyArray_BYTES(data_array), alloc_size ? alloc_size : 1,
                    PyArray_HANDLER(data_array));
            if (new_data == NULL) {
                PyErr_NoMemory();
                goto error;
            }
            /* Replace the arrays data since it may have changed */
            ((PyArrayObject_fields *)data_array)->data = new_data;
            ((PyArrayObject_fields *)data_array)->dimensions[0] = new_rows;
            data_ptr = new_data + row_count * row_size;
            data_allocated_rows = new_rows;
            if (needs_init) {
                memset(data_ptr, '\0', (new_rows - row_count) * row_size);
            }
        }

        for (Py_ssize_t i = 0; i < actual_num_fields; ++i) {
            Py_ssize_t f;  /* The field, either 0 (if homogeneous) or i. */
            Py_ssize_t col;  /* The column as read, remapped by usecols */
            char *item_ptr;
            if (homogeneous) {
                f = 0;
                item_ptr = data_ptr + i * field_types[0].descr->elsize;
            }
            else {
                f = i;
                item_ptr = data_ptr + field_types[f].structured_offset;
            }

            if (usecols == NULL) {
                col = i;
            }
            else {
                col = usecols[i];
                if (col < 0) {
                    // Python-like column indexing: k = -1 means the last column.
                    col += current_num_fields;
                }
                if (NPY_UNLIKELY((col < 0) || (col >= current_num_fields))) {
                    PyErr_Format(PyExc_ValueError,
                            "invalid column index %zd at row %zd with %zd "
                            "columns",
                            usecols[i], row_count+1, current_num_fields);
                    goto error;
                }
            }

            /*
             * The following function calls represent the main "conversion"
             * step, i.e. parsing the unicode string for each field and storing
             * the result in the array.
             */
            int parser_res;
            Py_UCS4 *str = ts.field_buffer + fields[col].offset;
            Py_UCS4 *end = ts.field_buffer + fields[col + 1].offset - 1;
            if (conv_funcs == NULL || conv_funcs[i] == NULL) {
                parser_res = field_types[f].set_from_ucs4(field_types[f].descr,
                        str, end, item_ptr, pconfig);
            }
            else {
                parser_res = npy_to_generic_with_converter(field_types[f].descr,
                        str, end, item_ptr, pconfig, conv_funcs[i]);
            }

            if (NPY_UNLIKELY(parser_res < 0)) {
                PyObject *exc, *val, *tb;
                PyErr_Fetch(&exc, &val, &tb);

                size_t length = end - str;
                PyObject *string = PyUnicode_FromKindAndData(
                        PyUnicode_4BYTE_KIND, str, length);
                if (string == NULL) {
                    npy_PyErr_ChainExceptions(exc, val, tb);
                    goto error;
                }
                PyErr_Format(PyExc_ValueError,
                        "could not convert string %.100R to %S at "
                        "row %zd, column %zd.",
                        string, field_types[f].descr, row_count, col+1);
                Py_DECREF(string);
                npy_PyErr_ChainExceptionsCause(exc, val, tb);
                goto error;
            }
        }

        ++row_count;
        data_ptr += row_size;
    }

    npy_tokenizer_clear(&ts);
    if (conv_funcs != NULL) {
        for (Py_ssize_t i = 0; i < actual_num_fields; i++) {
            Py_XDECREF(conv_funcs[i]);
        }
        PyMem_FREE(conv_funcs);
    }

    if (data_array == NULL) {
        assert(row_count == 0 && result_shape[0] == 0);
        if (actual_num_fields == -1) {
            /*
             * We found no rows and have to discover the number of elements
             * we have no choice but to guess 1.
             * NOTE: It may make sense to move this outside of here to refine
             *       the behaviour where necessary.
             */
            result_shape[1] = 1;
        }
        else {
            result_shape[1] = actual_num_fields;
        }
        Py_INCREF(out_descr);
        data_array = (PyArrayObject *)PyArray_Empty(
                ndim, result_shape, out_descr, 0);
    }

    /*
     * Note that if there is no data, `data_array` may still be NULL and
     * row_count is 0.  In that case, always realloc just in case.
     */
    if (data_array_allocated && data_allocated_rows != row_count) {
        size_t size = row_count * row_size;
        char *new_data = PyDataMem_UserRENEW(
                PyArray_BYTES(data_array), size ? size : 1,
                PyArray_HANDLER(data_array));
        if (new_data == NULL) {
            Py_DECREF(data_array);
            PyErr_NoMemory();
            return NULL;
        }
        ((PyArrayObject_fields *)data_array)->data = new_data;
        ((PyArrayObject_fields *)data_array)->dimensions[0] = row_count;
    }

    return data_array;

  error:
    if (conv_funcs != NULL) {
        for (Py_ssize_t i = 0; i < actual_num_fields; i++) {
            Py_XDECREF(conv_funcs[i]);
        }
        PyMem_FREE(conv_funcs);
    }
    npy_tokenizer_clear(&ts);
    Py_XDECREF(data_array);
    return NULL;
}
