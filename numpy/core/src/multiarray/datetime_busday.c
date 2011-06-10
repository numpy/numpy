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
 * Applies the 'roll' strategy to 'date', placing the result in 'out'
 * and setting 'out_day_of_week' to the day of the week that results.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
apply_business_day_roll(npy_datetime date, npy_datetime *out,
                    int *out_day_of_week,
                    NPY_BUSDAY_ROLL roll,
                    npy_bool *weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    int day_of_week;

    /* Deal with NaT input */
    if (date == NPY_DATETIME_NAT) {
        *out = NPY_DATETIME_NAT;
        if (roll == NPY_BUSDAY_RAISE) {
            PyErr_SetString(PyExc_ValueError,
                    "NaT input in busday_offset");
            return -1;
        }
        else {
            return 0;
        }
    }

    /* Get the day of the week for 'date' (1970-01-05 is Monday) */
    day_of_week = (int)((date - 4) % 7);
    if (day_of_week < 0) {
        day_of_week += 7;
    }

    /* Apply the 'roll' if it's not a business day */
    if (weekmask[day_of_week] == 0) {
        npy_datetime start_date = date;
        int start_day_of_week = day_of_week;

        switch (roll) {
            case NPY_BUSDAY_FOLLOWING:
            case NPY_BUSDAY_MODIFIEDFOLLOWING: {
                do {
                    ++date;
                    if (++day_of_week == 7) {
                        day_of_week = 0;
                    }
                } while (weekmask[day_of_week] == 0);

                if (roll == NPY_BUSDAY_MODIFIEDFOLLOWING) {
                    /* If we crossed a month boundary, do preceding instead */
                    if (days_to_month_number(start_date) !=
                                days_to_month_number(date)) {
                        date = start_date;
                        day_of_week = start_day_of_week;

                        do {
                            --date;
                            if (--day_of_week == -1) {
                                day_of_week = 6;
                            }
                        } while (weekmask[day_of_week] == 0);
                    }
                }
                break;
            }
            case NPY_BUSDAY_PRECEDING:
            case NPY_BUSDAY_MODIFIEDPRECEDING: {
                do {
                    --date;
                    if (--day_of_week == -1) {
                        day_of_week = 6;
                    }
                } while (weekmask[day_of_week] == 0);

                if (roll == NPY_BUSDAY_MODIFIEDPRECEDING) {
                    /* If we crossed a month boundary, do following instead */
                    if (days_to_month_number(start_date) !=
                                days_to_month_number(date)) {
                        date = start_date;
                        day_of_week = start_day_of_week;

                        do {
                            ++date;
                            if (++day_of_week == 7) {
                                day_of_week = 0;
                            }
                        } while (weekmask[day_of_week] == 0);
                    }
                }
                break;
            }
            case NPY_BUSDAY_NAT: {
                date = NPY_DATETIME_NAT;
                break;
            }
            case NPY_BUSDAY_RAISE: {
                *out = NPY_DATETIME_NAT;
                PyErr_SetString(PyExc_ValueError,
                        "Non-business day date in busday_offset");
                return -1;
            }
        }
    }

    *out = date;
    *out_day_of_week = day_of_week;

    return 0;
}

/*
 * Applies a single business day offset. See the function
 * business_day_offset for the meaning of all the parameters.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
apply_business_day_offset(npy_datetime date, npy_int64 offset,
                    npy_datetime *out,
                    NPY_BUSDAY_ROLL roll,
                    npy_bool *weekmask, int busdays_in_weekmask,
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
{
    int day_of_week = 0;

    /* Roll the date to a business day */
    if (apply_business_day_roll(date, &date, &day_of_week,
                                roll,
                                weekmask,
                                holidays_begin, holidays_end) < 0) {
        return -1;
    }

    /* If we get a NaT, just return it */
    if (date == NPY_DATETIME_NAT) {
        return 0;
    }
    
    /* Now we're on a valid business day */
    if (offset > 0) {
        /* Jump by as many weeks as we can */
        date += (offset / busdays_in_weekmask) * 7;
        offset = offset % busdays_in_weekmask;

        /* Step until we use up the rest of the offset */
        while (offset > 0) {
            ++date;
            if (++day_of_week == 7) {
                day_of_week = 0;
            }
            offset -= weekmask[day_of_week];
        }
    }
    else if (offset < 0) {
        /* Jump by as many weeks as we can */
        date += (offset / busdays_in_weekmask) * 7;
        offset = offset % busdays_in_weekmask;

        /* Step until we use up the rest of the offset */
        while (offset < 0) {
            --date;
            if (--day_of_week == -1) {
                day_of_week = 6;
            }
            offset += weekmask[day_of_week];
        }
    }

    *out = date;
    return 0;
}

/*
 * Applies the given offsets in business days to the dates provided.
 * This is the low-level function which requires already cleaned input
 * data.
 *
 * dates:    An array of dates with 'datetime64[D]' data type.
 * offsets:  An array safely convertible into type int64.
 * out:      Either NULL, or an array with 'datetime64[D]' data type
 *              in which to place the resulting dates.
 * roll:     A rule for how to treat non-business day dates.
 * weekmask: A 7-element boolean mask, 1 for possible business days and 0
 *              for non-business days.
 * busdays_in_weekmask: A count of how many 1's there are in weekmask.
 * holidays_begin/holidays_end: A sorted list of dates matching '[D]'
 *           unit metadata, with any dates falling on a day of the
 *           week without weekmask[i] == 1 already filtered out.
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
                    npy_datetime *holidays_begin, npy_datetime *holidays_end)
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
        goto fail;
    }
    dtypes[1] = PyArray_DescrFromType(NPY_INT64);
    if (dtypes[1] == NULL) {
        goto fail;
    }
    dtypes[2] = dtypes[0];
    Py_INCREF(dtypes[2]);

    /* Set up the iterator parameters */
    flags = NPY_ITER_EXTERNAL_LOOP|
            NPY_ITER_BUFFERED|
            NPY_ITER_ZEROSIZE_OK;
    op[0] = dates;
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op[1] = offsets;
    op_flags[1] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op[2] = out;
    op_flags[2] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_ALIGNED;

    /* Allocate the iterator */
    iter = NpyIter_MultiNew(3, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, dtypes);
    if (iter == NULL) {
        goto fail;
    }

    /* Loop over all elements */
    if (NpyIter_GetIterSize(iter) > 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strideptr, *innersizeptr;

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        strideptr = NpyIter_GetInnerStrideArray(iter);
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

        do {
            char *data_dates = dataptr[0];
            char *data_offsets = dataptr[1];
            char *data_out = dataptr[2];
            npy_intp stride_dates = strideptr[0];
            npy_intp stride_offsets = strideptr[1];
            npy_intp stride_out = strideptr[2];
            npy_intp count = *innersizeptr;

            while (count--) {
                if (apply_business_day_offset(*(npy_int64 *)data_dates,
                                       *(npy_int64 *)data_offsets,
                                       (npy_int64 *)data_out,
                                       roll,
                                       weekmask, busdays_in_weekmask,
                                       holidays_begin, holidays_end) < 0) {
                    goto fail;
                }

                data_dates += stride_dates;
                data_offsets += stride_offsets;
                data_out += stride_out;
            }
        } while (iternext(iter));
    }

    /* Get the return object from the iterator */
    ret = NpyIter_GetOperandArray(iter)[2];
    Py_INCREF(ret);

    goto finish;

fail:
    Py_XDECREF(ret);
    ret = NULL;

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
