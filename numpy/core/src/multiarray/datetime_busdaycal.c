/*
 * This file implements an object encapsulating a business day
 * calendar object for accelerating NumPy datetime business day functions.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/arrayobject.h>

#include "npy_config.h"
#include "npy_pycompat.h"

#include "numpy/arrayscalars.h"
#include "lowlevel_strided_loops.h"
#include "_datetime.h"
#include "datetime_busday.h"
#include "datetime_busdaycal.h"

NPY_NO_EXPORT int
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
        int i;

        if (PyBytes_AsStringAndSize(obj, &str, &len) < 0) {
            Py_DECREF(obj);
            return 0;
        }

        /* Length 7 is a string like "1111100" */
        if (len == 7) {
            for (i = 0; i < 7; ++i) {
                switch(str[i]) {
                    case '0':
                        weekmask[i] = 0;
                        break;
                    case '1':
                        weekmask[i] = 1;
                        break;
                    default:
                        goto general_weekmask_string;
                }
            }

            goto finish;
        }

general_weekmask_string:
        /* a string like "SatSun" or "Mon Tue Wed" */
        memset(weekmask, 0, 7);
        for (i = 0; i < len; i += 3) {
            while (isspace(str[i]))
                ++i;

            if (i == len) {
                goto finish;
            }
            else if (i + 2 >= len) {
                goto invalid_weekmask_string;
            }

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
                default:
                    goto invalid_weekmask_string;
            }
        }

        goto finish;

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
                        (PyArray_Check(obj) &&
                         PyArray_NDIM((PyArrayObject *)obj) != 1)) {
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
    npy_datetime e1 = *(const npy_datetime *)elem1;
    npy_datetime e2 = *(const npy_datetime *)elem2;

    return (e1 < e2) ? -1 : (e1 == e2) ? 0 : 1;
}

/*
 * Sorts the the array of dates provided in place and removes
 * NaT, duplicates and any date which is already excluded on account
 * of the weekmask.
 *
 * Returns the number of dates left after removing weekmask-excluded
 * dates.
 */
NPY_NO_EXPORT void
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
NPY_NO_EXPORT int
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

    Py_DECREF(dates);
    Py_DECREF(date_dtype);

    return 1;

fail:
    Py_XDECREF(dates);
    Py_XDECREF(date_dtype);
    return 0;
}

static PyObject *
busdaycalendar_new(PyTypeObject *subtype,
                    PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
    NpyBusDayCalendar *self;

    self = (NpyBusDayCalendar *)subtype->tp_alloc(subtype, 0);
    if (self != NULL) {
        /* Start with an empty holidays list */
        self->holidays.begin = NULL;
        self->holidays.end = NULL;

        /* Set the weekmask to the default */
        self->busdays_in_weekmask = 5;
        self->weekmask[0] = 1;
        self->weekmask[1] = 1;
        self->weekmask[2] = 1;
        self->weekmask[3] = 1;
        self->weekmask[4] = 1;
        self->weekmask[5] = 0;
        self->weekmask[6] = 0;
    }

    return (PyObject *)self;
}

static int
busdaycalendar_init(NpyBusDayCalendar *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"weekmask", "holidays", NULL};
    int i, busdays_in_weekmask;

    /* Clear the holidays if necessary */
    if (self->holidays.begin != NULL) {
        PyArray_free(self->holidays.begin);
        self->holidays.begin = NULL;
        self->holidays.end = NULL;
    }

    /* Reset the weekmask to the default */
    self->busdays_in_weekmask = 5;
    self->weekmask[0] = 1;
    self->weekmask[1] = 1;
    self->weekmask[2] = 1;
    self->weekmask[3] = 1;
    self->weekmask[4] = 1;
    self->weekmask[5] = 0;
    self->weekmask[6] = 0;

    /* Parse the parameters */
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                        "|O&O&:busdaycal", kwlist,
                        &PyArray_WeekMaskConverter, &self->weekmask[0],
                        &PyArray_HolidaysConverter, &self->holidays)) {
        return -1;
    }

    /* Count the number of business days in a week */
    busdays_in_weekmask = 0;
    for (i = 0; i < 7; ++i) {
        busdays_in_weekmask += self->weekmask[i];
    }
    self->busdays_in_weekmask = busdays_in_weekmask;

    /* Normalize the holidays list */
    normalize_holidays_list(&self->holidays, self->weekmask);

    if (self->busdays_in_weekmask == 0) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot construct a numpy.busdaycal with a weekmask of "
                "all zeros");
        return -1;
    }

    return 0;
}

static void
busdaycalendar_dealloc(NpyBusDayCalendar *self)
{
    /* Clear the holidays */
    if (self->holidays.begin != NULL) {
        PyArray_free(self->holidays.begin);
        self->holidays.begin = NULL;
        self->holidays.end = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
busdaycalendar_weekmask_get(NpyBusDayCalendar *self)
{
    PyArrayObject *ret;
    npy_intp size = 7;

    /* Allocate a 7-element boolean array */
    ret = (PyArrayObject *)PyArray_SimpleNew(1, &size, NPY_BOOL);
    if (ret == NULL) {
        return NULL;
    }

    /* Copy the weekmask data */
    memcpy(PyArray_DATA(ret), self->weekmask, 7);

    return (PyObject *)ret;
}

static PyObject *
busdaycalendar_holidays_get(NpyBusDayCalendar *self)
{
    PyArrayObject *ret;
    PyArray_Descr *date_dtype;
    npy_intp size = self->holidays.end - self->holidays.begin;

    /* Create a date dtype */
    date_dtype = create_datetime_dtype_with_unit(NPY_DATETIME, NPY_FR_D);
    if (date_dtype == NULL) {
        return NULL;
    }

    /* Allocate a date array (this steals the date_dtype reference) */
    ret = (PyArrayObject *)PyArray_SimpleNewFromDescr(1, &size, date_dtype);
    if (ret == NULL) {
        return NULL;
    }

    /* Copy the holidays */
    if (size > 0) {
        memcpy(PyArray_DATA(ret), self->holidays.begin,
                    size * sizeof(npy_datetime));
    }

    return (PyObject *)ret;
}

static PyGetSetDef busdaycalendar_getsets[] = {
    {"weekmask",
        (getter)busdaycalendar_weekmask_get,
        NULL, NULL, NULL},
    {"holidays",
        (getter)busdaycalendar_holidays_get,
        NULL, NULL, NULL},

    {NULL, NULL, NULL, NULL, NULL}
};

NPY_NO_EXPORT PyTypeObject NpyBusDayCalendar_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.busdaycalendar",                     /* tp_name */
    sizeof(NpyBusDayCalendar),                  /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)busdaycalendar_dealloc,         /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    busdaycalendar_getsets,                     /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)busdaycalendar_init,              /* tp_init */
    0,                                          /* tp_alloc */
    busdaycalendar_new,                         /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
};
