/*
 * This file implements the missing value NA singleton object for NumPy.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API
#define _MULTIARRAYMODULE
#include <numpy/arrayobject.h>

#include "npy_config.h"
#include "numpy/npy_3kcompat.h"

#include "na_singleton.h"

static PyObject *
na_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    NpyNA_fieldaccess *self;

    self = (NpyNA_fieldaccess *)subtype->tp_alloc(subtype, 0);
    if (self != NULL) {
        self->payload = 0;
        self->dtype = NULL;
        self->is_singleton = 0;
    }

    return (PyObject *)self;
}

static int
na_init(NpyNA_fieldaccess *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"payload", "dtype", NULL};
    int payload = 0;
    PyArray_Descr *dtype = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO&:NA", kwlist,
                        &payload,
                        &PyArray_DescrConverter, &dtype)) {
        Py_XDECREF(dtype);
        return -1;
    }

    if (payload < 0 || payload > 127) {
        PyErr_Format(PyExc_ValueError,
                    "out of bounds payload for NumPy NA, "
                    "%d is not in the range [0,127]", payload);
        Py_XDECREF(dtype);
        return -1;
    }
    else {
        self->payload = (npy_uint8)payload;
    }

    Py_XDECREF(self->dtype);
    self->dtype = dtype;

    return 0;
}

static void
na_dealloc(NpyNA_fieldaccess *self)
{
    Py_XDECREF(self->dtype);
    self->dtype = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
na_payload_get(NpyNA_fieldaccess *self)
{
    return PyInt_FromLong(self->payload);
}

static int
na_payload_set(NpyNA_fieldaccess *self, PyObject *value)
{
    long payload;

    /* Don't allow changing the static singleton instance */
    if (self->is_singleton) {
        PyErr_SetString(PyExc_RuntimeError,
                    "cannot change the payload of the NumPy NA singleton, "
                    "make a new copy like 'numpy.NA(payload)'");
        return -1;
    }
    /* Deleting the payload sets it to 0 */
    else if (value == NULL) {
        self->payload = 0;
    }
    else {
        /* Use PyNumber_Index to ensure an integer in Python >= 2.5*/
#if PY_VERSION_HEX >= 0x02050000
        value = PyNumber_Index(value);
        if (value == NULL) {
            return -1;
        }
#else
        Py_INCREF(value);
#endif
        payload = PyInt_AsLong(value);
        Py_DECREF(value);
        if (payload == -1 && PyErr_Occurred()) {
            return -1;
        }
        else if (payload < 0 || payload > 127) {
            PyErr_Format(PyExc_ValueError,
                        "out of bounds payload for NumPy NA, "
                        "%ld is not in the range [0,127]", payload);
            return -1;
        }
        else {
            self->payload = (npy_uint8)payload;
        }
    }

    return 0;
}

static PyObject *
na_dtype_get(NpyNA_fieldaccess *self)
{
    if (self->dtype == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    else {
        Py_INCREF(self->dtype);
        return (PyObject *)self->dtype;
    }
}

static int
na_dtype_set(NpyNA_fieldaccess *self, PyObject *value)
{
    PyArray_Descr *dtype = NULL;

    /* Don't allow changing the static singleton instance */
    if (self->is_singleton) {
        PyErr_SetString(PyExc_RuntimeError,
                    "cannot change the dtype of the NumPy NA singleton, "
                    "make a new copy like 'numpy.NA(dtype=val)'");
        return -1;
    }
    /* Convert the input into a dtype object */
    else if (!PyArray_DescrConverter(value, &dtype)) {
        return -1;
    }

    /* Replace the existing dtype in self */
    Py_XDECREF(self->dtype);
    self->dtype = dtype;

    return 0;
}

static PyGetSetDef na_getsets[] = {
    {"payload",
        (getter)na_payload_get,
        (setter)na_payload_set,
        NULL, NULL},
    {"dtype",
        (getter)na_dtype_get,
        (setter)na_dtype_set,
        NULL, NULL},

    {NULL, NULL, NULL, NULL, NULL}
};

NPY_NO_EXPORT PyTypeObject NpyNA_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.NAType",                             /* tp_name */
    sizeof(NpyNA_fieldaccess),                  /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)na_dealloc,                     /* tp_dealloc */
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
    na_getsets,                                 /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)na_init,                          /* tp_init */
    0,                                          /* tp_alloc */
    na_new,                                     /* tp_new */
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

NPY_NO_EXPORT NpyNA_fieldaccess _Npy_NASingleton = {
    PyObject_HEAD_INIT(&NpyNA_Type)
    0,    /* payload */
    NULL, /* dtype */
    1     /* is_singleton */
};

