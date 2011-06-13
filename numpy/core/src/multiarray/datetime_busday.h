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

/*
 * Converts a Python input into a 7-element weekmask, where 0 means
 * weekend and 1 means business day.
 */
NPY_NO_EXPORT int
PyArray_WeekMaskConverter(PyObject *weekmask_in, npy_bool *weekmask);

/*
 * Sorts the the array of dates provided in place and removes
 * NaT, duplicates and any date which is already excluded on account
 * of the weekmask.
 *
 * Returns the number of dates left after removing weekmask-excluded
 * dates.
 */
NPY_NO_EXPORT void
normalize_holidays_list(npy_holidayslist *holidays, npy_bool *weekmask);

/*
 * Converts a Python input into a non-normalized list of holidays.
 *
 * IMPORTANT: This function can't do the normalization, because it doesn't
 *            know the weekmask. You must call 'normalize_holiday_list'
 *            on the result before using it.
 */
NPY_NO_EXPORT int
PyArray_HolidaysConverter(PyObject *dates_in, npy_holidayslist *holidays);



#endif
