#ifndef _NPY_PRIVATE__DATETIME_BUSDAYDEF_H_
#define _NPY_PRIVATE__DATETIME_BUSDAYDEF_H_

typedef struct {
    PyObject_HEAD
    npy_holidayslist holidays;
    int busdays_in_weekmask;
    npy_bool weekmask[7];
} NpyBusinessDayDef;

NPY_NO_EXPORT PyTypeObject NpyBusinessDayDef_Type;

#define NpyBusinessDayDef_Check(op) PyObject_TypeCheck(op, \
                        &NpyBusinessDayDef_Type)

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
