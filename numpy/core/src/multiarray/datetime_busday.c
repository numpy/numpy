/*
 * This file implements business day functionality for NumPy datetime.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define _MULTIARRAYMODULE
#include <numpy/arrayobject.h>

#include "npy_config.h"
#include "numpy/npy_3kcompat.h"

#include "numpy/arrayscalars.h"
#include "_datetime.h"
#include "datetime_busday.h"

/*
 * Applies the given offsets in business days to the dates provided.
 * This is the low-level function which requires already cleaned input
 * data.
 *
 * dates:    An array of dates with 'datetime64[D]' data type.
 * offsets:  An array safely convertible into type 'timedelta64[D]'.
 * out:      Either NULL, or an array with 'datetime64[D]' data type
 *              in which to place the resulting dates.
 * roll:     A rule for how to treat non-business day dates.
 * weekmask: A 7-element boolean mask, 1 for possible business days and 0
 *              for non-business days.
 * busdays_in_weekmask: A count of how many 1's there are in weekmask.
 * holiday_count: The number of elements in 'holidays'.
 * holidays: A sorted list of dates matching '[D]' unit metadata, with
 *           any dates falling on a day of the week without weekmask[i] == 1
 *           already filtered out.
 *
 * For each (date, offset) in the broadcasted pair of (dates, offsets),
 * does the following:
 *  + Applies the 'roll' rule to the date to either produce NaT, raise
 *    an exception, or land on a valid business day.
 *  + Adds 'offset' business days to the valid business day found.
 *  + Sets the value in 'out' if provided, or the allocated output array
 *    otherwise.
 */
NPY_NO_EXPORT PyArrayObject *
business_day_offset(PyArrayObject *dates, PyArrayObject *offsets,
                    PyArrayObject *out,
                    NPY_BUSDAY_ROLL roll,
                    npy_bool *weekmask, int busdays_in_weekmask,
                    npy_intp holidays_count, npy_datetime *holidays)
{
    PyArray_DatetimeMetaData temp_meta;
    PyArray_Descr *dtypes[3] = {NULL, NULL, NULL};

    NpyIter *iter = NULL;
    PyArrayObject *op[3] = {NULL, NULL, NULL};
    npy_uint32 op_flags[3], flags;

    PyArrayObject *ret = NULL;

    /* First create the data types for dates and offsets */
    temp_meta.base = NPY_FR_D;
    temp_meta.num = 1;
    temp_meta.events = 1;
    dtypes[0] = create_datetime_dtype(NPY_DATETIME, &temp_meta);
    if (dtypes[0] == NULL) {
        ret = NULL;
        goto finish;
    }
    dtypes[1] = create_datetime_dtype(NPY_DATETIME, &temp_meta);
    if (dtypes[1] == NULL) {
        ret = NULL;
        goto finish;
    }
    dtypes[2] = dtypes[0];
    Py_INCREF(dtypes[2]);

    /* Set up the iterator parameters */
    flags = NPY_ITER_EXTERNAL_LOOP|
            NPY_ITER_BUFFERED|
            NPY_ITER_ZEROSIZE_OK;
    op[0] = dates;
    op_flags[0] = NPY_ITER_READONLY;
    op[1] = offsets;
    op_flags[1] = NPY_ITER_READONLY;
    op[2] = out;
    op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;

    /* Allocate the iterator */
    iter = NpyIter_MultiNew(3, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, dtypes);
    if (iter == NULL) {
        ret = NULL;
        goto finish;
    }

    /* Get the return object from the iterator */
    ret = NpyIter_GetOperandArray(iter)[2];
    Py_INCREF(ret);

finish:
    Py_XDECREF(dtypes[0]);
    Py_XDECREF(dtypes[1]);
    Py_XDECREF(dtypes[2]);
    if (iter != NULL) {
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_XDECREF(ret);
            ret = NULL;
        }
    }
    return ret;
}
