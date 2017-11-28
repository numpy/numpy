/*
  Provide multidimensional arrays as a basic object type in python.

  Based on Original Numeric implementation
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  with contributions from many Numeric Python developers 1995-2004

  Heavily modified in 2005 with inspiration from Numarray

  by

  Travis Oliphant,  oliphant@ee.byu.edu
  Brigham Young University


maintainer email:  oliphant.travis@ieee.org

  Numarray design (which provided guidance) by
  Space Science Telescope Institute
  (J. Todd Miller, Perry Greenfield, Rick White)
*/
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

/*#include <stdio.h>*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "common.h"

#include "npy_pycompat.h"

#include "usertypes.h"

NPY_NO_EXPORT PyArray_Descr **userdescrs=NULL;

static int *
_append_new(int *types, int insert)
{
    int n = 0;
    int *newtypes;

    while (types[n] != NPY_NOTYPE) {
        n++;
    }
    newtypes = (int *)realloc(types, (n + 2)*sizeof(int));
    newtypes[n] = insert;
    newtypes[n + 1] = NPY_NOTYPE;
    return newtypes;
}

static npy_bool
_default_nonzero(void *ip, void *arr)
{
    int elsize = PyArray_ITEMSIZE(arr);
    char *ptr = ip;
    while (elsize--) {
        if (*ptr++ != 0) {
            return NPY_TRUE;
        }
    }
    return NPY_FALSE;
}

static void
_default_copyswapn(void *dst, npy_intp dstride, void *src,
                   npy_intp sstride, npy_intp n, int swap, void *arr)
{
    npy_intp i;
    PyArray_CopySwapFunc *copyswap;
    char *dstptr = dst;
    char *srcptr = src;

    copyswap = PyArray_DESCR(arr)->f->copyswap;

    for (i = 0; i < n; i++) {
        copyswap(dstptr, srcptr, swap, arr);
        dstptr += dstride;
        srcptr += sstride;
    }
}

/*NUMPY_API
  Initialize arrfuncs to NULL
*/
NPY_NO_EXPORT void
PyArray_InitArrFuncs(PyArray_ArrFuncs *f)
{
    int i;

    for(i = 0; i < NPY_NTYPES_ABI_COMPATIBLE; i++) {
        f->cast[i] = NULL;
    }
    f->getitem = NULL;
    f->setitem = NULL;
    f->copyswapn = NULL;
    f->copyswap = NULL;
    f->compare = NULL;
    f->argmax = NULL;
    f->argmin = NULL;
    f->dotfunc = NULL;
    f->scanfunc = NULL;
    f->fromstr = NULL;
    f->nonzero = NULL;
    f->fill = NULL;
    f->fillwithscalar = NULL;
    for(i = 0; i < NPY_NSORTS; i++) {
        f->sort[i] = NULL;
        f->argsort[i] = NULL;
    }
    f->castdict = NULL;
    f->scalarkind = NULL;
    f->cancastscalarkindto = NULL;
    f->cancastto = NULL;
}

/*
  returns typenum to associate with this type >=NPY_USERDEF.
  needs the userdecrs table and PyArray_NUMUSER variables
  defined in arraytypes.inc
*/
/*NUMPY_API
  Register Data type
  Does not change the reference count of descr
*/
NPY_NO_EXPORT int
PyArray_RegisterDataType(PyArray_Descr *descr)
{
    PyArray_Descr *descr2;
    int typenum;
    int i;
    PyArray_ArrFuncs *f;

    /* See if this type is already registered */
    for (i = 0; i < NPY_NUMUSERTYPES; i++) {
        descr2 = userdescrs[i];
        if (descr2 == descr) {
            return descr->type_num;
        }
    }
    typenum = NPY_USERDEF + NPY_NUMUSERTYPES;
    descr->type_num = typenum;
    if (PyDataType_ISUNSIZED(descr)) {
        PyErr_SetString(PyExc_ValueError, "cannot register a" \
                        "flexible data-type");
        return -1;
    }
    f = descr->f;
    if (f->nonzero == NULL) {
        f->nonzero = _default_nonzero;
    }
    if (f->copyswapn == NULL) {
        f->copyswapn = _default_copyswapn;
    }
    if (f->copyswap == NULL || f->getitem == NULL ||
        f->setitem == NULL) {
        PyErr_SetString(PyExc_ValueError, "a required array function"   \
                        " is missing.");
        return -1;
    }
    if (descr->typeobj == NULL) {
        PyErr_SetString(PyExc_ValueError, "missing typeobject");
        return -1;
    }
    userdescrs = realloc(userdescrs,
                         (NPY_NUMUSERTYPES+1)*sizeof(void *));
    if (userdescrs == NULL) {
        PyErr_SetString(PyExc_MemoryError, "RegisterDataType");
        return -1;
    }
    userdescrs[NPY_NUMUSERTYPES++] = descr;
    return typenum;
}

/*NUMPY_API
  Register Casting Function
  Replaces any function currently stored.
*/
NPY_NO_EXPORT int
PyArray_RegisterCastFunc(PyArray_Descr *descr, int totype,
                         PyArray_VectorUnaryFunc *castfunc)
{
    PyObject *cobj, *key;
    int ret;

    if (totype < NPY_NTYPES_ABI_COMPATIBLE) {
        descr->f->cast[totype] = castfunc;
        return 0;
    }
    if (totype >= NPY_NTYPES && !PyTypeNum_ISUSERDEF(totype)) {
        PyErr_SetString(PyExc_TypeError, "invalid type number.");
        return -1;
    }
    if (descr->f->castdict == NULL) {
        descr->f->castdict = PyDict_New();
        if (descr->f->castdict == NULL) {
            return -1;
        }
    }
    key = PyInt_FromLong(totype);
    if (PyErr_Occurred()) {
        return -1;
    }
    cobj = NpyCapsule_FromVoidPtr((void *)castfunc, NULL);
    if (cobj == NULL) {
        Py_DECREF(key);
        return -1;
    }
    ret = PyDict_SetItem(descr->f->castdict, key, cobj);
    Py_DECREF(key);
    Py_DECREF(cobj);
    return ret;
}

/*NUMPY_API
 * Register a type number indicating that a descriptor can be cast
 * to it safely
 */
NPY_NO_EXPORT int
PyArray_RegisterCanCast(PyArray_Descr *descr, int totype,
                        NPY_SCALARKIND scalar)
{
    /*
     * If we were to allow this, the casting lookup table for
     * built-in types needs to be modified, as cancastto is
     * not checked for them.
     */
    if (!PyTypeNum_ISUSERDEF(descr->type_num) &&
                                        !PyTypeNum_ISUSERDEF(totype)) {
        PyErr_SetString(PyExc_ValueError,
                        "At least one of the types provided to"
                        "RegisterCanCast must be user-defined.");
        return -1;
    }

    if (scalar == NPY_NOSCALAR) {
        /*
         * register with cancastto
         * These lists won't be freed once created
         * -- they become part of the data-type
         */
        if (descr->f->cancastto == NULL) {
            descr->f->cancastto = (int *)malloc(1*sizeof(int));
            descr->f->cancastto[0] = NPY_NOTYPE;
        }
        descr->f->cancastto = _append_new(descr->f->cancastto,
                                          totype);
    }
    else {
        /* register with cancastscalarkindto */
        if (descr->f->cancastscalarkindto == NULL) {
            int i;
            descr->f->cancastscalarkindto =
                (int **)malloc(NPY_NSCALARKINDS* sizeof(int*));
            for (i = 0; i < NPY_NSCALARKINDS; i++) {
                descr->f->cancastscalarkindto[i] = NULL;
            }
        }
        if (descr->f->cancastscalarkindto[scalar] == NULL) {
            descr->f->cancastscalarkindto[scalar] =
                (int *)malloc(1*sizeof(int));
            descr->f->cancastscalarkindto[scalar][0] =
                NPY_NOTYPE;
        }
        descr->f->cancastscalarkindto[scalar] =
            _append_new(descr->f->cancastscalarkindto[scalar], totype);
    }
    return 0;
}
