#ifndef _NPY_PRIVATE__NA_SINGLETON_H_
#define _NPY_PRIVATE__NA_SINGLETON_H_

/* Direct access to the fields of the NA object is just internal to NumPy. */
typedef struct {
    PyObject_HEAD
    /* NA payload, 0 by default */
    npy_uint8 payload;
    /* NA dtype, NULL by default */
    PyArray_Descr *dtype;
    /* Internal flag, whether this is the singleton numpy.NA or not */
    int is_singleton;
} NpyNA_fieldaccess;

NPY_NO_EXPORT NpyNA_fieldaccess _Npy_NASingleton;

#define Npy_NA ((PyObject *)&_Npy_NASingleton)

#endif
