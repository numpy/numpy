#ifndef _NPY_PRIVATE__DATETIME_BUSDAYDEF_H_
#define _NPY_PRIVATE__DATETIME_BUSDAYDEF_H_

typedef struct {
    PyObject_HEAD
    npy_holidayslist holidays;
    int busdays_in_weekmask;
    npy_bool weekmask[7];
} PyArray_BusinessDayDef;

NPY_NO_EXPORT PyTypeObject NpyBusinessDayDef_Type;

#define NpyBusinessDayDef_Check(op) PyObject_TypeCheck(op, \
                        &NpyBusinessDayDef_Type)

#endif
