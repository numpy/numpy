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
#include "lowlevel_strided_loops.h"
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

    if (busdays_in_weekmask == 0) {
        PyErr_SetString(PyExc_ValueError,
                "the business day weekmask must have at least one "
                "valid business day");
        return NULL;
    }

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

static int
PyArray_BusDayRollConverter(PyObject *roll_in, NPY_BUSDAY_ROLL *roll)
{
    PyObject *obj = roll_in;
    char *str;
    Py_ssize_t len;

    /* Make obj into an ASCII string */
    Py_INCREF(obj);
    if (PyUnicode_Check(obj)) {
        /* accept unicode input */
        PyObject *obj_str;
        obj_str = PyUnicode_AsASCIIString(obj);
        if (obj_str == NULL) {
            Py_DECREF(obj);
            return 0;
        }
        Py_DECREF(obj);
        obj = obj_str;
    }

    if (PyBytes_AsStringAndSize(obj, &str, &len) < 0) {
        Py_DECREF(obj);
        return 0;
    }

    /* Use switch statements to quickly isolate the right enum value */
    switch (str[0]) {
        case 'b':
            if (strcmp(str, "backward") == 0) {
                *roll = NPY_BUSDAY_BACKWARD;
                goto finish;
            }
            break;
        case 'f':
            if (len > 2) switch (str[2]) {
                case 'r':
                    if (strcmp(str, "forward") == 0) {
                        *roll = NPY_BUSDAY_FORWARD;
                        goto finish;
                    }
                    break;
                case 'l':
                    if (strcmp(str, "following") == 0) {
                        *roll = NPY_BUSDAY_FOLLOWING;
                        goto finish;
                    }
                    break;
            }
            break;
        case 'm':
            if (len > 8) switch (str[8]) {
                case 'f':
                    if (strcmp(str, "modifiedfollowing") == 0) {
                        *roll = NPY_BUSDAY_MODIFIEDFOLLOWING;
                        goto finish;
                    }
                    break;
                case 'p':
                    if (strcmp(str, "modifiedpreceding") == 0) {
                        *roll = NPY_BUSDAY_MODIFIEDFOLLOWING;
                        goto finish;
                    }
                    break;
            }
            break;
        case 'n':
            if (strcmp(str, "nat") == 0) {
                *roll = NPY_BUSDAY_NAT;
                goto finish;
            }
            break;
        case 'p':
            if (strcmp(str, "preceding") == 0) {
                *roll = NPY_BUSDAY_PRECEDING;
                goto finish;
            }
            break;
        case 'r':
            if (strcmp(str, "raise") == 0) {
                *roll = NPY_BUSDAY_RAISE;
                goto finish;
            }
            break;
    }

    PyErr_Format(PyExc_ValueError,
            "Invalid business day roll parameter \"%s\"",
            str);
    Py_DECREF(obj);
    return 0;

finish:
    Py_DECREF(obj);
    return 1;
}

static int
PyArray_WeekMaskConverter(PyObject *weekmask_in, npy_bool *weekmask)
{
    PyObject *obj = weekmask_in;

    /* Make obj into an ASCII string if it is UNICODE */
    Py_INCREF(obj);
    if (PyUnicode_Check(obj)) {
        /* accept unicode input */
        PyObject *obj_str;
        obj_str = PyUnicode_AsASCIIString(obj);
        if (obj_str == NULL) {
            Py_DECREF(obj);
            return 0;
        }
        Py_DECREF(obj);
        obj = obj_str;
    }

    if (PyBytes_Check(obj)) {
        char *str;
        Py_ssize_t len;

        if (PyBytes_AsStringAndSize(obj, &str, &len) < 0) {
            Py_DECREF(obj);
            return 0;
        }

        /* Length 7 is a string like "1111100" */
        if (len == 7) {
            int i;
            for (i = 0; i < 7; ++i) {
                switch(str[i]) {
                    case '0':
                        weekmask[i] = 0;
                        break;
                    case '1':
                        weekmask[i] = 1;
                        break;
                    default:
                        goto invalid_weekmask_string;
                }
            }

            goto finish;
        }
        /* Length divisible by 3 is a string like "Mon" or "MonWedFri" */
        else if (len % 3 == 0) {
            int i;
            memset(weekmask, 0, 7);
            for (i = 0; i < len; i += 3) {
                switch (str[i]) {
                    case 'M':
                        if (str[i+1] == 'o' && str[i+2] == 'n') {
                            weekmask[0] = 1;
                        }
                        else {
                            goto invalid_weekmask_string;
                        }
                        break;
                    case 'T':
                        if (str[i+1] == 'u' && str[i+2] == 'e') {
                            weekmask[1] = 1;
                        }
                        else if (str[i+1] == 'h' && str[i+2] == 'u') {
                            weekmask[3] = 1;
                        }
                        else {
                            goto invalid_weekmask_string;
                        }
                        break;
                    case 'W':
                        if (str[i+1] == 'e' && str[i+2] == 'd') {
                            weekmask[2] = 1;
                        }
                        else {
                            goto invalid_weekmask_string;
                        }
                        break;
                    case 'F':
                        if (str[i+1] == 'r' && str[i+2] == 'i') {
                            weekmask[4] = 1;
                        }
                        else {
                            goto invalid_weekmask_string;
                        }
                        break;
                    case 'S':
                        if (str[i+1] == 'a' && str[i+2] == 't') {
                            weekmask[5] = 1;
                        }
                        else if (str[i+1] == 'u' && str[i+2] == 'n') {
                            weekmask[6] = 1;
                        }
                        else {
                            goto invalid_weekmask_string;
                        }
                        break;
                }
            }

            goto finish;
        }

invalid_weekmask_string:
        PyErr_Format(PyExc_ValueError,
                "Invalid business day weekmask string \"%s\"",
                str);
        Py_DECREF(obj);
        return 0;
    }
    /* Something like [1,1,1,1,1,0,0] */
    else if (PySequence_Check(obj)) {
        if (PySequence_Size(obj) != 7 ||
                (PyArray_Check(obj) && PyArray_NDIM(obj) != 1)) {
            PyErr_SetString(PyExc_ValueError,
                "A business day weekmask array must have length 7");
            Py_DECREF(obj);
            return 0;
        }
        else {
            int i;
            PyObject *f;

            for (i = 0; i < 7; ++i) {
                long val;

                f = PySequence_GetItem(obj, i);
                if (f == NULL) {
                    Py_DECREF(obj);
                    return 0;
                }

                val = PyInt_AsLong(f);
                if (val == -1 && PyErr_Occurred()) {
                    Py_DECREF(obj);
                    return 0;
                }
                if (val == 0) {
                    weekmask[i] = 0;
                }
                else if (val == 1) {
                    weekmask[i] = 1;
                }
                else {
                    PyErr_SetString(PyExc_ValueError,
                        "A business day weekmask array must have all "
                        "1's and 0's");
                    Py_DECREF(obj);
                    return 0;
                }
            }

            goto finish;
        }
    }

    PyErr_SetString(PyExc_ValueError,
            "Couldn't convert object into a business day weekmask");
    Py_DECREF(obj);
    return 0;


finish:
    Py_DECREF(obj);
    return 1;
}

static int
qsort_datetime_compare(const void *elem1, const void *elem2)
{
    npy_datetime e1 = *(npy_datetime *)elem1;
    npy_datetime e2 = *(npy_datetime *)elem2;

    return (e1 < e2) ? -1 : (e1 == e2) ? 0 : 1;
}

/*
 * A list of holidays, which should sorted, not contain any
 * duplicates or NaTs, and not include any days already excluded
 * by the associated weekmask.
 *
 * The data is manually managed with PyArray_malloc/PyArray_free.
 */
typedef struct {
    npy_datetime *begin, *end;
} npy_holidayslist;

/*
 * Sorts the the array of dates provided in place and removes
 * NaT, duplicates and any date which is already excluded on account
 * of the weekmask.
 *
 * Returns the number of dates left after removing weekmask-excluded
 * dates.
 */
static void
normalize_holidays_list(npy_holidayslist *holidays, npy_bool *weekmask)
{
    npy_datetime *dates = holidays->begin;
    npy_intp count = holidays->end - dates;

    npy_datetime lastdate = NPY_DATETIME_NAT;
    npy_intp trimcount, i;
    int day_of_week;

    /* Sort the dates */
    qsort(dates, count, sizeof(npy_datetime), &qsort_datetime_compare);

    /* Sweep throught the array, eliminating unnecessary values */
    trimcount = 0;
    for (i = 0; i < count; ++i) {
        npy_datetime date = dates[i];

        /* Skip any NaT or duplicate */
        if (date != NPY_DATETIME_NAT && date != lastdate) {
            /* Get the day of the week (1970-01-05 is Monday) */
            day_of_week = (int)((date - 4) % 7);
            if (day_of_week < 0) {
                day_of_week += 7;
            }

            /*
             * If the holiday falls on a possible business day,
             * then keep it.
             */
            if (weekmask[day_of_week] == 1) {
                dates[trimcount++] = date;
                lastdate = date;
            }
        }
    }

    /* Adjust the end of the holidays array */
    holidays->end = dates + trimcount;
}

/*
 * Converts a Python input into a non-normalized list of holidays.
 *
 * IMPORTANT: This function can't do the normalization, because it doesn't
 *            know the weekmask. You must call 'normalize_holiday_list'
 *            on the result before using it.
 */
static int
PyArray_HolidaysConverter(PyObject *dates_in, npy_holidayslist *holidays)
{
    PyArrayObject *dates = NULL;
    PyArray_Descr *date_dtype = NULL;
    npy_intp count;

    /* Make 'dates' into an array */
    if (PyArray_Check(dates_in)) {
        dates = (PyArrayObject *)dates_in;
        Py_INCREF(dates);
    }
    else {
        PyArray_Descr *datetime_dtype;

        /* Use the datetime dtype with generic units so it fills it in */
        datetime_dtype = PyArray_DescrFromType(NPY_DATETIME);
        if (datetime_dtype == NULL) {
            goto fail;
        }

        /* This steals the datetime_dtype reference */
        dates = (PyArrayObject *)PyArray_FromAny(dates_in, datetime_dtype,
                                                0, 0, 0, dates_in);
        if (dates == NULL) {
            goto fail;
        }
    }

    date_dtype = create_datetime_dtype_with_unit(NPY_DATETIME, NPY_FR_D);
    if (date_dtype == NULL) {
        goto fail;
    }

    if (!PyArray_CanCastTypeTo(PyArray_DESCR(dates),
                                    date_dtype, NPY_SAFE_CASTING)) {
        PyErr_SetString(PyExc_ValueError, "Cannot safely convert "
                        "provided holidays input into an array of dates");
        goto fail;
    }
    if (PyArray_NDIM(dates) != 1) {
        PyErr_SetString(PyExc_ValueError, "holidays must be a provided "
                        "as a one-dimensional array");
        goto fail;
    }

    /* Allocate the memory for the dates */
    count = PyArray_DIM(dates, 0);
    holidays->begin = PyArray_malloc(sizeof(npy_datetime) * count);
    if (holidays->begin == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    holidays->end = holidays->begin + count;

    /* Cast the data into a raw date array */
    if (PyArray_CastRawArrays(count,
                            PyArray_BYTES(dates), (char *)holidays->begin,
                            PyArray_STRIDE(dates, 0), sizeof(npy_datetime),
                            PyArray_DESCR(dates), date_dtype,
                            0) != NPY_SUCCEED) {
        goto fail;
    }

    Py_DECREF(date_dtype);
    date_dtype = NULL;

fail:
    Py_XDECREF(dates);
    Py_XDECREF(date_dtype);
    return 0;
}

/*
 * This is the 'busday_offset' function exposed for calling
 * from Python.
 */
NPY_NO_EXPORT PyObject *
array_busday_offset(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds)
{
    char *kwlist[] = {"dates", "offsets", "roll",
                      "weekmask", "holidays", NULL};

    PyObject *dates_in = NULL, *offsets_in = NULL,
            *busdaydef_in = NULL, *out_in = NULL;

    PyArrayObject *dates = NULL, *offsets = NULL, *out = NULL, *ret;
    NPY_BUSDAY_ROLL roll = NPY_BUSDAY_RAISE;
    npy_bool weekmask[7] = {1, 1, 1, 1, 1, 0, 0};
    int i, busdays_in_weekmask;
    npy_holidayslist holidays = {NULL, NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                                    "OO|O&O&O&OO:busday_offset", kwlist,
                                    &dates_in,
                                    &offsets_in,
                                    &PyArray_BusDayRollConverter, &roll,
                                    &PyArray_WeekMaskConverter, &weekmask[0],
                                    &PyArray_HolidaysConverter, &holidays,
                                    &busdaydef_in,
                                    &out_in)) {
        return NULL;
    }

    /* Make 'dates' into an array */
    if (PyArray_Check(dates_in)) {
        dates = (PyArrayObject *)dates_in;
        Py_INCREF(dates);
    }
    else {
        PyArray_Descr *date_dtype;

        /* Use the datetime dtype with generic units so it fills it in */
        date_dtype = PyArray_DescrFromType(NPY_DATETIME);
        if (date_dtype == NULL) {
            goto fail;
        }

        /* This steals the date_dtype reference */
        dates = (PyArrayObject *)PyArray_FromAny(dates_in, date_dtype,
                                                0, 0, 0, dates_in);
        if (dates == NULL) {
            goto fail;
        }
    }

    /* Make 'offsets' into an array */
    offsets = (PyArrayObject *)PyArray_FromAny(offsets_in,
                            PyArray_DescrFromType(NPY_INT64),
                            0, 0, 0, offsets_in);
    if (offsets == NULL) {
        goto fail;
    }

    /* Make sure 'out' is an array if it's provided */
    if (out_in != NULL) {
        if (!PyArray_Check(out_in)) {
            PyErr_SetString(PyExc_ValueError,
                    "busday_offset: must provide a NumPy array for 'out'");
            goto fail;
        }
        out = (PyArrayObject *)out_in;
    }

    /* Count the number of business days in a week */
    busdays_in_weekmask = 0;
    for (i = 0; i < 7; ++i) {
        busdays_in_weekmask += weekmask[i];
    }

    /* The holidays list must be normalized before using it */
    normalize_holidays_list(&holidays, weekmask);

    ret = business_day_offset(dates, offsets, out, roll,
                    weekmask, busdays_in_weekmask,
                    holidays.begin, holidays.end);

    Py_DECREF(dates);
    Py_DECREF(offsets);
    if (holidays.begin != NULL) {
        PyArray_free(holidays.begin);
    }

    return out == NULL ? PyArray_Return(ret) : (PyObject *)ret;
fail:
    Py_XDECREF(dates);
    Py_XDECREF(offsets);
    if (holidays.begin != NULL) {
        PyArray_free(holidays.begin);
    }

    return NULL;
}
