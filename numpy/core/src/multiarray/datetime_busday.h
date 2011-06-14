#ifndef _NPY_PRIVATE__DATETIME_BUSDAY_H_
#define _NPY_PRIVATE__DATETIME_BUSDAY_H_

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
 * This is the 'busday_offset' function exposed for calling
 * from Python.
 */
NPY_NO_EXPORT PyObject *
array_busday_offset(PyObject *NPY_UNUSED(self),
                      PyObject *args, PyObject *kwds);


#endif
