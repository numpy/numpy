Datetime API
============

NumPy represents dates internally using an int64 counter and a unit metadata
struct. Time differences are represented similarly using an int64 and a unit
metadata struct. The functions described below are available to to facilitate
converting between ISO 8601 date strings, NumPy datetimes, and Python datetime
objects in C.

Data types
----------

In addition to the `npy_datetime` and `npy_timedelta` typedefs for `npy_int64`,
NumPy defines two additional structs that represent time unit metadata and
an "exploded" view of a datetime.

.. c:type:: PyArray_DatetimeMetaData

   Represents datetime unit metadata.

   .. code-block:: c

       typedef struct {
           NPY_DATETIMEUNIT base;
           int num;
       } PyArray_DatetimeMetaData;

   .. c:member:: NPY_DATETIMEUNIT base

       The unit of the datetime.

   .. c:member:: int num

       A multiplier for the unit.

.. c:type:: npy_datetimestruct

   An "exploded" view of a datetime value

   .. code-block:: c

       typedef struct {
           npy_int64 year;
           npy_int32 month, day, hour, min, sec, us, ps, as;
       } npy_datetimestruct;

.. c:enum:: NPY_DATETIMEUNIT

   Time units supported by NumPy. The "FR" in the names of the enum variants
   is short for frequency.

   .. c:enumerator:: NPY_FR_ERROR

       Error or undetermined units.

   .. c:enumerator:: NPY_FR_Y

       Years

   .. c:enumerator:: NPY_FR_M

       Months

   .. c:enumerator:: NPY_FR_W

       Weeks

   .. c:enumerator:: NPY_FR_D

       Days

   .. c:enumerator:: NPY_FR_h

       Hours

   .. c:enumerator:: NPY_FR_m

       Minutes

   .. c:enumerator:: NPY_FR_s

       Seconds

   .. c:enumerator:: NPY_FR_ms

       Milliseconds

   .. c:enumerator:: NPY_FR_us

       Microseconds

   .. c:enumerator:: NPY_FR_ns

       Nanoseconds

   .. c:enumerator:: NPY_FR_ps

       Picoseconds

   .. c:enumerator:: NPY_FR_fs

       Femtoseconds

   .. c:enumerator:: NPY_FR_as

       Attoseconds

   .. c:enumerator:: NPY_FR_GENERIC

       Unbound units, can convert to anything


Conversion functions
--------------------

.. c:function:: int NpyDatetime_ConvertDatetimeStructToDatetime64( \
        PyArray_DatetimeMetaData *meta, const npy_datetimestruct *dts, \
        npy_datetime *out)

    Converts a datetime from a datetimestruct to a datetime in the units
    specified by the unit metadata. The date is assumed to be valid.

    If the ``num`` member of the metadata struct is large, there may
    be integer overflow in this function.

    Returns 0 on success and -1 on failure.

.. c:function:: int NpyDatetime_ConvertDatetime64ToDatetimeStruct( \
        PyArray_DatetimeMetaData *meta, npy_datetime dt, \
        npy_datetimestruct *out)

    Converts a datetime with units specified by the unit metadata to an
    exploded datetime struct.

    Returns 0 on success and -1 on failure.

.. c:function:: int NpyDatetime_ConvertPyDateTimeToDatetimeStruct( \
        PyObject *obj, npy_datetimestruct *out, \
        NPY_DATETIMEUNIT *out_bestunit, int apply_tzinfo)

    Tests for and converts a Python ``datetime.datetime`` or ``datetime.date``
    object into a NumPy ``npy_datetimestruct``.

    ``out_bestunit`` gives a suggested unit based on whether the object
    was a ``datetime.date`` or ``datetime.datetime`` object.

    If ``apply_tzinfo`` is 1, this function uses the tzinfo to convert
    to UTC time, otherwise it returns the struct with the local time.

    Returns -1 on error, 0 on success, and 1 (with no error set)
    if obj doesn't have the needed date or datetime attributes.

.. c:function:: int NpyDatetime_ParseISO8601Datetime( \
        char const *str, Py_ssize_t len, NPY_DATETIMEUNIT unit, \
        NPY_CASTING casting, npy_datetimestruct *out, \
        NPY_DATETIMEUNIT *out_bestunit, npy_bool *out_special)

    Parses (almost) standard ISO 8601 date strings. The differences are:

    * The date "20100312" is parsed as the year 20100312, not as
      equivalent to "2010-03-12". The '-' in the dates are not optional.
    * Only seconds may have a decimal point, with up to 18 digits after it
      (maximum attoseconds precision).
    * Either a 'T' as in ISO 8601 or a ' ' may be used to separate
      the date and the time. Both are treated equivalently.
    * Doesn't (yet) handle the "YYYY-DDD" or "YYYY-Www" formats.
    * Doesn't handle leap seconds (seconds value has 60 in these cases).
    * Doesn't handle 24:00:00 as synonym for midnight (00:00:00) tomorrow
    * Accepts special values "NaT" (not a time), "Today", (current
      day according to local time) and "Now" (current time in UTC).

    ``str`` must be a NULL-terminated string, and ``len`` must be its length.

    ``unit`` should contain -1 if the unit is unknown, or the unit
    which will be used if it is.

    ``casting`` controls how the detected unit from the string is allowed
    to be cast to the 'unit' parameter.

    ``out`` gets filled with the parsed date-time.

    ``out_bestunit`` gives a suggested unit based on the amount of
    resolution provided in the string, or -1 for NaT.
    
    ``out_special`` gets set to 1 if the parsed time was 'today',
    'now', empty string, or 'NaT'. For 'today', the unit recommended is
    'D', for 'now', the unit recommended is 's', and for 'NaT'
    the unit recommended is 'Y'.

    Returns 0 on success, -1 on failure.

.. c:function:: int NpyDatetime_GetDatetimeISO8601StrLen(\
        int local, NPY_DATETIMEUNIT base)

    Returns the string length to use for converting datetime
    objects with the given local time and unit settings to strings.
    Use this when constructings strings to supply to
    ``NpyDatetime_MakeISO8601Datetime``.

.. c:function:: int NpyDatetime_MakeISO8601Datetime(\
        npy_datetimestruct *dts, char *outstr, npy_intp outlen, \
        int local, int utc, NPY_DATETIMEUNIT base, int tzoffset, \
        NPY_CASTING casting)

    Converts an ``npy_datetimestruct`` to an (almost) ISO 8601
    NULL-terminated string. If the string fits in the space exactly,
    it leaves out the NULL terminator and returns success.

    The differences from ISO 8601 are the 'NaT' string, and
    the number of year digits is >= 4 instead of strictly 4.

    If ``local`` is non-zero, it produces a string in local time with
    a +-#### timezone offset. If ``local`` is zero and ``utc`` is non-zero,
    produce a string ending with 'Z' to denote UTC. By default, no time
    zone information is attached.

    ``base`` restricts the output to that unit. Set ``base`` to
    -1 to auto-detect a base after which all the values are zero.

    ``tzoffset`` is used if ``local`` is enabled, and ``tzoffset`` is
    set to a value other than -1. This is a manual override for
    the local time zone to use, as an offset in minutes.

    ``casting`` controls whether data loss is allowed by truncating
    the data to a coarser unit. This interacts with ``local``, slightly,
    in order to form a date unit string as a local time, the casting
    must be unsafe.

    Returns 0 on success, -1 on failure (for example if the output
    string was too short).
