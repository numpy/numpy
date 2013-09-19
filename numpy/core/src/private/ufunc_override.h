#ifndef __UFUNC_OVERRIDE_H
#define __UFUNC_OVERRIDE_H
#include <npy_config.h>
#include "numpy/arrayobject.h"
#include "common.h"

/*
 * Check a set of args for the `__numpy_ufunc__` method.  If more than one of
 * the input arguments implements `__numpy_ufunc__`, they are tried in the
 * order: subclasses before superclasses, otherwise left to right. The first
 * routine returning something other than `NotImplemented` determines the
 * result. If all of the `__numpy_ufunc__` operations returns `NotImplemented`,
 * a `TypeError` is raised.
 */
static int
PyUFunc_CheckOverride(PyObject *ufunc, char *method,
                      PyObject *args, PyObject *kwds, 
                      PyObject **result,
                      int nin)
{
    int i;
    int override_pos; /* Position of override in args.*/
    int j;
    int pos_in_with_override; /* Position of override in with_override.*/

    int nargs = PyTuple_GET_SIZE(args);
    int noa = 0; /* Number of overriding args.*/
    int normalized = 0; /* Is normalized flag.*/

    PyObject *obj;
    PyObject *other_obj;
    PyObject *override_args;

    PyObject *method_name = PyUString_FromString(method);
    PyObject *normal_args = NULL; /* normal_* holds normalized arguments. */
    PyObject *normal_kwds = NULL;
    PyObject *override_obj = NULL; /* overriding object */
    PyObject *numpy_ufunc = NULL; /* the __numpy_ufunc__ method */

    PyObject *with_override[NPY_MAXARGS]; 
    /* Pos of each override in args */
    int with_override_pos[NPY_MAXARGS];

    /* Checks */
    if (!PyTuple_Check(args)) {
        goto fail;
    }
    if (PyTuple_GET_SIZE(args) > NPY_MAXARGS) {
        goto fail;
    }

    for (i = 0; i < nargs; ++i) {
        obj = PyTuple_GET_ITEM(args, i);
        if (PyArray_CheckExact(obj) || PyArray_IsAnyScalar(obj)) {
            continue;
        }
        if (PyObject_HasAttrString(obj, "__numpy_ufunc__")) {
            with_override[noa] = obj;
            with_override_pos[noa] = i;
            ++noa;
        }
    }

    /* No overrides, bail out.*/
    if (noa == 0) {
        Py_DECREF(method_name);
        return 0;
    }

    while (1) {
        obj = NULL;
        override_obj = NULL;
        *result = NULL;

        /* Choose an overriding argument */
        for (i = 0; i < noa; i++) {
            obj = with_override[i];
            if (obj == NULL) {
                continue;
            }
            /* Get the first instance of an overriding arg.*/
            override_pos = with_override_pos[i];
            override_obj = obj;
            pos_in_with_override = i;

            /* Check for sub-types to the right of obj. */
            for (j = i + 1; j < noa; j++) {
                other_obj = with_override[j];
                if (PyObject_Type(other_obj) != PyObject_Type(obj) &&
                    PyObject_IsInstance(other_obj, 
                                        PyObject_Type(override_obj))) {
                    override_obj = NULL;
                    break;
                }
            }
            /* override_obj had no subtypes to the right. */
            if (override_obj) {
                break;
            }
        }
        /* No good override_obj */
        if (!override_obj) {
            break;
        }
        /* 
         * Normalize the ufuncs arguments. Returns a tuple of 
         * (args, kwds).
         *
         * Test with and without kwds.
         */
        if (!normalized) {
            PyObject *out_arg;

            /* If we have more args than nin, the last one must be `out`.*/
            if (nargs > nin) {
                out_arg = PyTuple_GET_ITEM(args, nargs - 1);

                /* Build new args.*/
                normal_args = PyTuple_GetSlice(args, 0, nin);

                /* Build new kwds with out arg.*/
                if (kwds && PyDict_CheckExact(kwds)) {
                    normal_kwds = PyDict_Copy(kwds);
                    PyDict_SetItemString(normal_kwds, "out", out_arg);
                }
                else {
                    normal_kwds = PyDict_New();
                    PyDict_SetItemString(normal_kwds, "out", out_arg);
                }

                normalized = 1;
            }
            else {
                /* Copy args */
                normal_args = PyTuple_GetSlice(args, 0, nin);
                if (kwds && PyDict_CheckExact(kwds)) {
                    normal_kwds = PyDict_Copy(kwds);
                }
                else {
                    normal_kwds = PyDict_New();
                }

                normalized = 1;
            }
        }

        /* Calculate a result if we have a override. */
        if (override_obj) {
            numpy_ufunc = PyObject_GetAttrString(override_obj, 
                                                   "__numpy_ufunc__");
            override_args = Py_BuildValue("OOiO", ufunc, method_name, 
                                          override_pos, normal_args);
            *result = PyObject_Call(numpy_ufunc, override_args, normal_kwds);
            
            Py_DECREF(numpy_ufunc);
            Py_DECREF(override_args);

            /* Remove this arg if it gives not implemented */
            if (*result == Py_NotImplemented) {
                with_override[pos_in_with_override] = NULL;
                Py_DECREF(*result);
                continue;
            }
            /* Good result. */
            else {
                break;
            }
        }

        /* All overrides checked. */
        else {
            break;
        }
    }
    /* No acceptable override found. */
    if (!*result) {
        PyErr_SetString(PyExc_TypeError, 
                        "__numpy_ufunc__ not implemented for this type.");
        Py_XDECREF(normal_args);
        Py_XDECREF(normal_kwds);
        goto fail;
    }
    /* Override found, return it. */
    Py_DECREF(method_name);
    Py_XDECREF(normal_args);
    Py_XDECREF(normal_kwds);
    return 0;

fail:
    Py_DECREF(method_name);
    return 1;
    
}

#endif
