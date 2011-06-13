/*
 * This file implements an object encapsulating a business day
 * definition for accelerating NumPy datetime business day functions.
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
#include "datetime_busdaydef.h"

static PyObject *
busdaydef_new(PyTypeObject *subtype,
                    PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds))
{
    PyArray_BusinessDayDef *self;

    self = (PyArray_BusinessDayDef *)subtype->tp_alloc(subtype, 0);
    if (self != NULL) {
         /* Set the weekmask to the default */
        self->weekmask[0] = 1;
        self->weekmask[1] = 1;
        self->weekmask[2] = 1;
        self->weekmask[3] = 1;
        self->weekmask[4] = 1;
        self->weekmask[5] = 0;
        self->weekmask[6] = 0;

        /* Start with an empty holidays list */
        self->holidays.begin = NULL;
        self->holidays.end = NULL;
    }

    return (PyObject *)self;
}

static int
busdaydef_init(PyArray_BusinessDayDef *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"weekmask", "holidays", NULL};

    /* Reset the weekmask to the default */
    self->weekmask[0] = 1;
    self->weekmask[1] = 1;
    self->weekmask[2] = 1;
    self->weekmask[3] = 1;
    self->weekmask[4] = 1;
    self->weekmask[5] = 0;
    self->weekmask[6] = 0;

    /* Clear the holidays if necessary */
    if (self->holidays.begin != NULL) {
        PyArray_free(self->holidays.begin);
        self->holidays.begin = NULL;
        self->holidays.end = NULL;
    }

    /* Parse the parameters */
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                        "|O&O&", kwlist,
                        &PyArray_WeekMaskConverter, &self->weekmask[0],
                        &PyArray_HolidaysConverter, &self->holidays)) {
        return -1;
    }
 
    /* Normalize the holidays list */
    normalize_holidays_list(&self->holidays, self->weekmask);

    return 0;
}

static void
busdaydef_dealloc(PyArray_BusinessDayDef *self)
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
busdaydef_weekmask_get(PyArray_BusinessDayDef *self)
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
busdaydef_holidays_get(PyArray_BusinessDayDef *self)
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

static PyGetSetDef busdaydef_getsets[] = {
    {"weekmask",
        (getter)busdaydef_weekmask_get,
        NULL, NULL, NULL},
    {"holidays",
        (getter)busdaydef_holidays_get,
        NULL, NULL, NULL},

    {NULL, NULL, NULL, NULL, NULL}
};

NPY_NO_EXPORT PyTypeObject NpyBusinessDayDef_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.busdaydef",                          /* tp_name */
    sizeof(PyArray_BusinessDayDef),             /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)busdaydef_dealloc,              /* tp_dealloc */
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
    busdaydef_getsets,                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)busdaydef_init,                   /* tp_init */
    0,                                          /* tp_alloc */
    busdaydef_new,                              /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0,                                          /* tp_version_tag */
#endif
};

