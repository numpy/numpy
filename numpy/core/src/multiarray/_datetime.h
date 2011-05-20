#ifndef _NPY_PRIVATE__DATETIME_H_
#define _NPY_PRIVATE__DATETIME_H_

NPY_NO_EXPORT void
PyArray_DatetimeToDatetimeStruct(npy_datetime val, NPY_DATETIMEUNIT fr,
                                 npy_datetimestruct *result);

NPY_NO_EXPORT void
PyArray_TimedeltaToTimedeltaStruct(npy_timedelta val, NPY_DATETIMEUNIT fr,
                                 npy_timedeltastruct *result);

NPY_NO_EXPORT npy_datetime
PyArray_DatetimeStructToDatetime(NPY_DATETIMEUNIT fr, npy_datetimestruct *d);

NPY_NO_EXPORT npy_datetime
PyArray_TimedeltaStructToTimedelta(NPY_DATETIMEUNIT fr, npy_timedeltastruct *d);

/*
 * This function returns a pointer to the DateTimeMetaData
 * contained within the provided datetime dtype.
 */
NPY_NO_EXPORT PyArray_DatetimeMetaData *
get_datetime_metadata_from_dtype(PyArray_Descr *dtype);

/*
 * This function returns a reference to a PyCObject/Capsule
 * which contains the datetime metadata parsed from a metadata
 * string. 'metastr' should be NULL-terminated, and len should
 * contain its string length.
 */
NPY_NO_EXPORT PyObject *
parse_datetime_metacobj_from_metastr(char *metastr, Py_ssize_t len);

/*
 * Converts a datetype dtype string into a dtype descr object.
 * The "type" string should be NULL-terminated, and len should
 * contain its string length.
 */
NPY_NO_EXPORT PyArray_Descr *
parse_dtype_from_datetime_typestr(char *typestr, Py_ssize_t len);

/*
 * Converts a substring given by 'str' and 'len' into
 * a date time unit enum value. The 'metastr' parameter
 * is used for error messages, and may be NULL.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT NPY_DATETIMEUNIT
parse_datetime_unit_from_string(char *str, Py_ssize_t len, char *metastr);

/*
 * Translate divisors into multiples of smaller units.
 * 'metastr' is used for the error message if the divisor doesn't work,
 * and can be NULL if the metadata didn't come from a string.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_datetime_divisor_to_multiple(PyArray_DatetimeMetaData *meta,
                                    int den, char *metastr);

/*
 * Given an the CObject/Capsule datetime metadata object,
 * returns a tuple for pickling and other purposes.
 */
NPY_NO_EXPORT PyObject *
convert_datetime_metadata_to_tuple(PyArray_DatetimeMetaData *meta);

/*
 * Given a tuple representing datetime metadata,
 * returns a CObject/Capsule datetime metadata object.
 */
NPY_NO_EXPORT PyObject *
convert_datetime_metadata_tuple_to_metacobj(PyObject *tuple);

/*
 * 'ret' is a PyUString containing the datetime string, and this
 * function appends the metadata string to it.
 *
 * This function steals the reference 'ret'
 */
NPY_NO_EXPORT PyObject *
append_metastr_to_datetime_typestr(PyArray_Descr *self, PyObject *ret);

#endif
