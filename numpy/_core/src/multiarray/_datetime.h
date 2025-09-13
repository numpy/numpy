#ifndef NUMPY_CORE_SRC_MULTIARRAY__DATETIME_H_
#define NUMPY_CORE_SRC_MULTIARRAY__DATETIME_H_

extern NPY_NO_EXPORT char const *_datetime_strings[NPY_DATETIME_NUMUNITS];
extern NPY_NO_EXPORT int _days_per_month_table[2][12];

NPY_NO_EXPORT void
numpy_pydatetime_import(void);

/*
 * Returns 1 if the given year is a leap year, 0 otherwise.
 */
NPY_NO_EXPORT int
is_leapyear(npy_int64 year);

/*
 * Calculates the days offset from the 1970 epoch.
 */
NPY_NO_EXPORT npy_int64
get_datetimestruct_days(const npy_datetimestruct *dts);

/*
 * Creates a datetime or timedelta dtype using a copy of the provided metadata.
 */
NPY_NO_EXPORT PyArray_Descr *
create_datetime_dtype(int type_num, PyArray_DatetimeMetaData *meta);

/*
 * Creates a datetime or timedelta dtype using the given unit.
 */
NPY_NO_EXPORT PyArray_Descr *
create_datetime_dtype_with_unit(int type_num, NPY_DATETIMEUNIT unit);

/*
 * This function returns a pointer to the DateTimeMetaData
 * contained within the provided datetime dtype.
 */
NPY_NO_EXPORT PyArray_DatetimeMetaData *
get_datetime_metadata_from_dtype(PyArray_Descr *dtype);

NPY_NO_EXPORT int
find_string_array_datetime64_type(PyArrayObject *arr,
        PyArray_DatetimeMetaData *meta);

/*
 * Both type1 and type2 must be either NPY_DATETIME or NPY_TIMEDELTA.
 * Applies the type promotion rules between the two types, returning
 * the promoted type.
 */
NPY_NO_EXPORT PyArray_Descr *
datetime_type_promotion(PyArray_Descr *type1, PyArray_Descr *type2);

/*
 * Extracts the month number, within the current year,
 * from a 'datetime64[D]' value. January is 1, etc.
 */
NPY_NO_EXPORT int
days_to_month_number(npy_datetime days);

/*
 * Parses the metadata string into the metadata C structure.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
parse_datetime_metadata_from_metastr(char const *metastr, Py_ssize_t len,
                                    PyArray_DatetimeMetaData *out_meta);


/*
 * Converts a datetype dtype string into a dtype descr object.
 * The "type" string should be NULL-terminated, and len should
 * contain its string length.
 */
NPY_NO_EXPORT PyArray_Descr *
parse_dtype_from_datetime_typestr(char const *typestr, Py_ssize_t len);

/*
 * Converts a substring given by 'str' and 'len' into
 * a date time unit enum value. The 'metastr' parameter
 * is used for error messages, and may be NULL.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT NPY_DATETIMEUNIT
parse_datetime_unit_from_string(char const *str, Py_ssize_t len, char const *metastr);

/*
 * Translate divisors into multiples of smaller units.
 * 'metastr' is used for the error message if the divisor doesn't work,
 * and can be NULL if the metadata didn't come from a string.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_datetime_divisor_to_multiple(PyArray_DatetimeMetaData *meta,
                                    int den, char const *metastr);

/*
 * Determines whether the 'divisor' metadata divides evenly into
 * the 'dividend' metadata.
 */
NPY_NO_EXPORT npy_bool
datetime_metadata_divides(
                        PyArray_DatetimeMetaData *dividend,
                        PyArray_DatetimeMetaData *divisor,
                        int strict_with_nonlinear_units);

/*
 * This provides the casting rules for the DATETIME data type units.
 *
 * Notably, there is a barrier between 'date units' and 'time units'
 * for all but 'unsafe' casting.
 */
NPY_NO_EXPORT npy_bool
can_cast_datetime64_units(NPY_DATETIMEUNIT src_unit,
                          NPY_DATETIMEUNIT dst_unit,
                          NPY_CASTING casting);

/*
 * This provides the casting rules for the DATETIME data type metadata.
 */
NPY_NO_EXPORT npy_bool
can_cast_datetime64_metadata(PyArray_DatetimeMetaData *src_meta,
                             PyArray_DatetimeMetaData *dst_meta,
                             NPY_CASTING casting);

/*
 * This provides the casting rules for the TIMEDELTA data type units.
 *
 * Notably, there is a barrier between the nonlinear years and
 * months units, and all the other units.
 */
NPY_NO_EXPORT npy_bool
can_cast_timedelta64_units(NPY_DATETIMEUNIT src_unit,
                          NPY_DATETIMEUNIT dst_unit,
                          NPY_CASTING casting);

/*
 * This provides the casting rules for the TIMEDELTA data type metadata.
 */
NPY_NO_EXPORT npy_bool
can_cast_timedelta64_metadata(PyArray_DatetimeMetaData *src_meta,
                             PyArray_DatetimeMetaData *dst_meta,
                             NPY_CASTING casting);

/*
 * Computes the conversion factor to convert data with 'src_meta' metadata
 * into data with 'dst_meta' metadata.
 *
 * If overflow occurs, both out_num and out_denom are set to 0, but
 * no error is set.
 */
NPY_NO_EXPORT void
get_datetime_conversion_factor(PyArray_DatetimeMetaData *src_meta,
                                PyArray_DatetimeMetaData *dst_meta,
                                npy_int64 *out_num, npy_int64 *out_denom);

/*
 * Given a pointer to datetime metadata,
 * returns a tuple for pickling and other purposes.
 */
NPY_NO_EXPORT PyObject *
convert_datetime_metadata_to_tuple(PyArray_DatetimeMetaData *meta);

/*
 * Converts a metadata tuple into a datetime metadata C struct.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_datetime_metadata_tuple_to_datetime_metadata(PyObject *tuple,
                                        PyArray_DatetimeMetaData *out_meta,
                                        npy_bool from_pickle);

/*
 * Gets a tzoffset in minutes by calling the astimezone() function on
 * the Python datetime.datetime object.
 */
NPY_NO_EXPORT int
get_tzoffset_from_pytzinfo(PyObject *timezone, npy_datetimestruct *dts);

/*
 * Converts an input object into datetime metadata. The input
 * may be either a string or a tuple.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_pyobject_to_datetime_metadata(PyObject *obj,
                                        PyArray_DatetimeMetaData *out_meta);

/*
 * Returns datetime metadata as a new reference a Unicode object.
 * Returns NULL on error.
 *
 * If 'skip_brackets' is true, skips the '[]'.
 *
 */
NPY_NO_EXPORT PyObject *
metastr_to_unicode(PyArray_DatetimeMetaData *meta, int skip_brackets);


/*
 * Converts a PyObject * into a datetime, in any of the forms supported.
 *
 * If the units metadata isn't known ahead of time, set meta->base
 * to -1, and this function will populate meta with either default
 * values or values from the input object.
 *
 * The 'casting' parameter is used to control what kinds of inputs
 * are accepted, and what happens. For example, with 'unsafe' casting,
 * unrecognized inputs are converted to 'NaT' instead of throwing an error,
 * while with 'safe' casting an error will be thrown if any precision
 * from the input will be thrown away.
 *
 * Returns -1 on error, 0 on success.
 */
NPY_NO_EXPORT int
convert_pyobject_to_datetime(PyArray_DatetimeMetaData *meta, PyObject *obj,
                                NPY_CASTING casting, npy_datetime *out);

/*
 * Converts a PyObject * into a timedelta, in any of the forms supported
 *
 * If the units metadata isn't known ahead of time, set meta->base
 * to -1, and this function will populate meta with either default
 * values or values from the input object.
 *
 * The 'casting' parameter is used to control what kinds of inputs
 * are accepted, and what happens. For example, with 'unsafe' casting,
 * unrecognized inputs are converted to 'NaT' instead of throwing an error,
 * while with 'safe' casting an error will be thrown if any precision
 * from the input will be thrown away.
 *
 * Returns -1 on error, 0 on success.
 */
NPY_NO_EXPORT int
convert_pyobject_to_timedelta(PyArray_DatetimeMetaData *meta, PyObject *obj,
                                NPY_CASTING casting, npy_timedelta *out);

/*
 * Converts a datetime into a PyObject *.
 *
 * NaT (Not-a-time) is returned as None.
 * For D/W/Y/M (days or coarser), returns a datetime.date.
 * For μs/ms/s/m/h/D/W (microseconds or coarser), returns a datetime.datetime.
 * For ns/ps/fs/as (units shorter than microseconds), returns an integer.
 */
NPY_NO_EXPORT PyObject *
convert_datetime_to_pyobject(npy_datetime dt, PyArray_DatetimeMetaData *meta);

/*
 * Converts a timedelta into a PyObject *.
 *
 * NaT (Not-a-time) is returned as None.
 * For μs/ms/s/m/h/D/W (microseconds or coarser), returns a datetime.timedelta.
 * For Y/M (non-linear units), generic units and ns/ps/fs/as (units shorter than microseconds), returns an integer.
 */
NPY_NO_EXPORT PyObject *
convert_timedelta_to_pyobject(npy_timedelta td, PyArray_DatetimeMetaData *meta);

/*
 * Adjusts a datetimestruct based on a seconds offset. Assumes
 * the current values are valid.
 */
NPY_NO_EXPORT void
add_seconds_to_datetimestruct(npy_datetimestruct *dts, int seconds);

/*
 * Adjusts a datetimestruct based on a minutes offset. Assumes
 * the current values are valid.
 */
NPY_NO_EXPORT void
add_minutes_to_datetimestruct(npy_datetimestruct *dts, int minutes);

/*
 * Returns true if the datetime metadata matches
 */
NPY_NO_EXPORT npy_bool
has_equivalent_datetime_metadata(PyArray_Descr *type1, PyArray_Descr *type2);

/*
 * Casts a single datetime from having src_meta metadata into
 * dst_meta metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
cast_datetime_to_datetime(PyArray_DatetimeMetaData *src_meta,
                          PyArray_DatetimeMetaData *dst_meta,
                          npy_datetime src_dt,
                          npy_datetime *dst_dt);

/*
 * Casts a single timedelta from having src_meta metadata into
 * dst_meta metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
cast_timedelta_to_timedelta(PyArray_DatetimeMetaData *src_meta,
                          PyArray_DatetimeMetaData *dst_meta,
                          npy_timedelta src_dt,
                          npy_timedelta *dst_dt);

/*
 * Returns true if the object is something that is best considered
 * a Datetime or Timedelta, false otherwise.
 */
NPY_NO_EXPORT npy_bool
is_any_numpy_datetime_or_timedelta(PyObject *obj);

/*
 * Implements a datetime-specific arange
 */
NPY_NO_EXPORT PyArrayObject *
datetime_arange(PyObject *start, PyObject *stop, PyObject *step,
                PyArray_Descr *dtype);

/*
 * Examines all the objects in the given Python object by
 * recursively descending the sequence structure. Returns a
 * datetime or timedelta type with metadata based on the data.
 */
NPY_NO_EXPORT PyArray_Descr *
find_object_datetime_type(PyObject *obj, int type_num);

NPY_NO_EXPORT int
PyArray_InitializeDatetimeCasts(void);

NPY_NO_EXPORT npy_hash_t
datetime_hash(PyArray_DatetimeMetaData *meta, npy_datetime dt);

NPY_NO_EXPORT npy_hash_t
timedelta_hash(PyArray_DatetimeMetaData *meta, npy_timedelta td);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY__DATETIME_H_ */
