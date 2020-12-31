#ifndef _NPY_DTYPEMETA_H
#define _NPY_DTYPEMETA_H

#define NPY_DTYPE(descr) ((PyArray_DTypeMeta *)Py_TYPE(descr))
/*
 * This function will hopefully be phased out or replaced, but was convenient
 * for incremental implementation of new DTypes based on DTypeMeta.
 * (Error checking is not required for DescrFromType, assuming that the
 * type is valid.)
 */
static NPY_INLINE PyArray_DTypeMeta *
PyArray_DTypeFromTypeNum(int typenum)
{
    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    PyArray_DTypeMeta *dtype = NPY_DTYPE(descr);
    Py_INCREF(dtype);
    Py_DECREF(descr);
    return dtype;
}


NPY_NO_EXPORT int
dtypemeta_wrap_legacy_descriptor(PyArray_Descr *dtypem);

#endif  /*_NPY_DTYPEMETA_H */
