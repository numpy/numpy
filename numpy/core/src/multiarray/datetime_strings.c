/*
 * This file implements string parsing and creation for NumPy datetime.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <time.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/arrayobject.h>

#include "npy_config.h"
#include "npy_pycompat.h"

#include "numpy/arrayscalars.h"
#include "methods.h"
#include "_datetime.h"
#include "datetime_strings.h"

/*
 * Platform-specific time_t typedef. Some platforms use 32 bit, some use 64 bit
 * and we just use the default with the exception of mingw, where we must use
 * 64 bit because MSVCRT version 9 does not have the (32 bit) localtime()
 * symbol, so we need to use the 64 bit version [1].
 *
 * [1] http://thread.gmane.org/gmane.comp.gnu.mingw.user/27011
 */
#if defined(NPY_MINGW_USE_CUSTOM_MSVCR)
 typedef __time64_t NPY_TIME_T;
#else
 typedef time_t NPY_TIME_T;
#endif

/*
 * Wraps `localtime` functionality for multiple platforms. This
 * converts a time value to a time structure in the local timezone.
 * If size(NPY_TIME_T) == 4, then years must be between 1970 and 2038. If
 * size(NPY_TIME_T) == 8, then years must be later than 1970. If the years are
 * not in this range, then get_localtime() will fail on some platforms.
 *
 * Returns 0 on success, -1 on failure.
 *
 * Notes:
 * 1) If NPY_TIME_T is 32 bit (i.e. sizeof(NPY_TIME_T) == 4), then the
 *    maximum year it can represent is 2038 (see [1] for more details). Trying
 *    to use a higher date like 2041 in the 32 bit "ts" variable below will
 *    typically result in "ts" being a negative number (corresponding roughly
 *    to a year ~ 1905). If NPY_TIME_T is 64 bit, then there is no such
 *    problem in practice.
 * 2) If the "ts" argument to localtime() is negative, it represents
 *    years < 1970 both for 32 and 64 bits (for 32 bits the earliest year it can
 *    represent is 1901, while 64 bits can represent much earlier years).
 * 3) On Linux, localtime() works for negative "ts". On Windows and in Wine,
 *    localtime() as well as the localtime_s() and _localtime64_s() functions
 *    will fail for any negative "ts" and return a nonzero exit number
 *    (localtime_s, _localtime64_s) or NULL (localtime). This behavior is the
 *    same for both 32 and 64 bits.
 *
 * From this it follows that get_localtime() is only guaranteed to work
 * correctly on all platforms for years between 1970 and 2038 for 32bit
 * NPY_TIME_T and years higher than 1970 for 64bit NPY_TIME_T. For
 * multiplatform code, get_localtime() should never be used outside of this
 * range.
 *
 * [1] https://en.wikipedia.org/wiki/Year_2038_problem
 */
static int
get_localtime(NPY_TIME_T *ts, struct tm *tms)
{
    char *func_name = "<unknown>";
#if defined(_WIN32)
 #if defined(_MSC_VER) && (_MSC_VER >= 1400)
    if (localtime_s(tms, ts) != 0) {
        func_name = "localtime_s";
        goto fail;
    }
 #elif defined(NPY_MINGW_USE_CUSTOM_MSVCR)
    if (_localtime64_s(tms, ts) != 0) {
        func_name = "_localtime64_s";
        goto fail;
    }
 #else
    struct tm *tms_tmp;
    tms_tmp = localtime(ts);
    if (tms_tmp == NULL) {
        func_name = "localtime";
        goto fail;
    }
    memcpy(tms, tms_tmp, sizeof(struct tm));
 #endif
#else
    if (localtime_r(ts, tms) == NULL) {
        func_name = "localtime_r";
        goto fail;
    }
#endif

    return 0;

fail:
    PyErr_Format(PyExc_OSError, "Failed to use '%s' to convert "
                                "to a local time", func_name);
    return -1;
}

/*
 * Converts a datetimestruct in UTC to a datetimestruct in local time,
 * also returning the timezone offset applied. This function works for any year
 * > 1970 on all platforms and both 32 and 64 bits. If the year < 1970, then it
 * will fail on some platforms.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
convert_datetimestruct_utc_to_local(npy_datetimestruct *out_dts_local,
                const npy_datetimestruct *dts_utc, int *out_timezone_offset)
{
    NPY_TIME_T rawtime = 0, localrawtime;
    struct tm tm_;
    npy_int64 year_correction = 0;

    /* Make a copy of the input 'dts' to modify */
    *out_dts_local = *dts_utc;

    /*
     * For 32 bit NPY_TIME_T, the get_localtime() function does not work for
     * years later than 2038, see the comments above get_localtime(). So if the
     * year >= 2038, we instead call get_localtime() for the year 2036 or 2037
     * (depending on the leap year) which must work and at the end we add the
     * 'year_correction' back.
     */
    if (sizeof(NPY_TIME_T) == 4 && out_dts_local->year >= 2038) {
        if (is_leapyear(out_dts_local->year)) {
            /* 2036 is a leap year */
            year_correction = out_dts_local->year - 2036;
            out_dts_local->year -= year_correction; /* = 2036 */
        }
        else {
            /* 2037 is not a leap year */
            year_correction = out_dts_local->year - 2037;
            out_dts_local->year -= year_correction; /* = 2037 */
        }
    }

    /*
     * Convert everything in 'dts' to a time_t, to minutes precision.
     * This is POSIX time, which skips leap-seconds, but because
     * we drop the seconds value from the npy_datetimestruct, everything
     * is ok for this operation.
     */
    rawtime = (NPY_TIME_T)get_datetimestruct_days(out_dts_local) * 24 * 60 * 60;
    rawtime += dts_utc->hour * 60 * 60;
    rawtime += dts_utc->min * 60;

    /* localtime converts a 'time_t' into a local 'struct tm' */
    if (get_localtime(&rawtime, &tm_) < 0) {
        /* This should only fail if year < 1970 on some platforms. */
        return -1;
    }

    /* Copy back all the values except seconds */
    out_dts_local->min = tm_.tm_min;
    out_dts_local->hour = tm_.tm_hour;
    out_dts_local->day = tm_.tm_mday;
    out_dts_local->month = tm_.tm_mon + 1;
    out_dts_local->year = tm_.tm_year + 1900;

    /* Extract the timezone offset that was applied */
    rawtime /= 60;
    localrawtime = (NPY_TIME_T)get_datetimestruct_days(out_dts_local) * 24 * 60;
    localrawtime += out_dts_local->hour * 60;
    localrawtime += out_dts_local->min;

    *out_timezone_offset = localrawtime - rawtime;

    /* Reapply the year 2038 year correction */
    out_dts_local->year += year_correction;

    return 0;
}

/*
 * Parses (almost) standard ISO 8601 date strings. The differences are:
 *
 * + The date "20100312" is parsed as the year 20100312, not as
 *   equivalent to "2010-03-12". The '-' in the dates are not optional.
 * + Only seconds may have a decimal point, with up to 18 digits after it
 *   (maximum attoseconds precision).
 * + Either a 'T' as in ISO 8601 or a ' ' may be used to separate
 *   the date and the time. Both are treated equivalently.
 * + Doesn't (yet) handle the "YYYY-DDD" or "YYYY-Www" formats.
 * + Doesn't handle leap seconds (seconds value has 60 in these cases).
 * + Doesn't handle 24:00:00 as synonym for midnight (00:00:00) tomorrow
 * + Accepts special values "NaT" (not a time), "Today", (current
 *   day according to local time) and "Now" (current time in UTC).
 *
 * 'str' must be a NULL-terminated string, and 'len' must be its length.
 * 'unit' should contain -1 if the unit is unknown, or the unit
 *      which will be used if it is.
 * 'casting' controls how the detected unit from the string is allowed
 *           to be cast to the 'unit' parameter.
 *
 * 'out' gets filled with the parsed date-time.
 * 'out_bestunit' gives a suggested unit based on the amount of
 *      resolution provided in the string, or -1 for NaT.
 * 'out_special' gets set to 1 if the parsed time was 'today',
 *      'now', or ''/'NaT'. For 'today', the unit recommended is
 *      'D', for 'now', the unit recommended is 's', and for 'NaT'
 *      the unit recommended is 'Y'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
parse_iso_8601_datetime(char *str, Py_ssize_t len,
                    NPY_DATETIMEUNIT unit,
                    NPY_CASTING casting,
                    npy_datetimestruct *out,
                    NPY_DATETIMEUNIT *out_bestunit,
                    npy_bool *out_special)
{
    int year_leap = 0;
    int i, numdigits;
    char *substr;
    Py_ssize_t sublen;
    NPY_DATETIMEUNIT bestunit;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(npy_datetimestruct));
    out->month = 1;
    out->day = 1;

    /*
     * Convert the empty string and case-variants of "NaT" to not-a-time.
     * Tried to use PyOS_stricmp, but that function appears to be broken,
     * not even matching the strcmp function signature as it should.
     */
    if (len <= 0 || (len == 3 &&
                        tolower(str[0]) == 'n' &&
                        tolower(str[1]) == 'a' &&
                        tolower(str[2]) == 't')) {
        out->year = NPY_DATETIME_NAT;

        /*
         * Indicate that this was a special value, and
         * recommend generic units.
         */
        if (out_bestunit != NULL) {
            *out_bestunit = NPY_FR_GENERIC;
        }
        if (out_special != NULL) {
            *out_special = 1;
        }

        return 0;
    }

    if (unit == NPY_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
                    "Cannot create a NumPy datetime other than NaT "
                    "with generic units");
        return -1;
    }

    /*
     * The string "today" means take today's date in local time, and
     * convert it to a date representation. This date representation, if
     * forced into a time unit, will be at midnight UTC.
     * This is perhaps a little weird, but done so that the
     * 'datetime64[D]' type produces the date you expect, rather than
     * switching to an adjacent day depending on the current time and your
     * timezone.
     */
    if (len == 5 && tolower(str[0]) == 't' &&
                    tolower(str[1]) == 'o' &&
                    tolower(str[2]) == 'd' &&
                    tolower(str[3]) == 'a' &&
                    tolower(str[4]) == 'y') {
        NPY_TIME_T rawtime = 0;
        struct tm tm_;

        time(&rawtime);
        if (get_localtime(&rawtime, &tm_) < 0) {
            return -1;
        }
        out->year = tm_.tm_year + 1900;
        out->month = tm_.tm_mon + 1;
        out->day = tm_.tm_mday;

        bestunit = NPY_FR_D;

        /*
         * Indicate that this was a special value, and
         * is a date (unit 'D').
         */
        if (out_bestunit != NULL) {
            *out_bestunit = bestunit;
        }
        if (out_special != NULL) {
            *out_special = 1;
        }

        /* Check the casting rule */
        if (unit != NPY_FR_ERROR &&
                !can_cast_datetime64_units(bestunit, unit, casting)) {
            PyErr_Format(PyExc_TypeError, "Cannot parse \"%s\" as unit "
                         "'%s' using casting rule %s",
                         str, _datetime_strings[unit],
                         npy_casting_to_string(casting));
            return -1;
        }

        return 0;
    }

    /* The string "now" resolves to the current UTC time */
    if (len == 3 && tolower(str[0]) == 'n' &&
                    tolower(str[1]) == 'o' &&
                    tolower(str[2]) == 'w') {
        NPY_TIME_T rawtime = 0;
        PyArray_DatetimeMetaData meta;

        time(&rawtime);

        /* Set up a dummy metadata for the conversion */
        meta.base = NPY_FR_s;
        meta.num = 1;

        bestunit = NPY_FR_s;

        /*
         * Indicate that this was a special value, and
         * use 's' because the time() function has resolution
         * seconds.
         */
        if (out_bestunit != NULL) {
            *out_bestunit = bestunit;
        }
        if (out_special != NULL) {
            *out_special = 1;
        }

        /* Check the casting rule */
        if (unit != NPY_FR_ERROR &&
                !can_cast_datetime64_units(bestunit, unit, casting)) {
            PyErr_Format(PyExc_TypeError, "Cannot parse \"%s\" as unit "
                         "'%s' using casting rule %s",
                         str, _datetime_strings[unit],
                         npy_casting_to_string(casting));
            return -1;
        }

        return convert_datetime_to_datetimestruct(&meta, rawtime, out);
    }

    /* Anything else isn't a special value */
    if (out_special != NULL) {
        *out_special = 0;
    }

    substr = str;
    sublen = len;

    /* Skip leading whitespace */
    while (sublen > 0 && isspace(*substr)) {
        ++substr;
        --sublen;
    }

    /* Leading '-' sign for negative year */
    if (*substr == '-' || *substr == '+') {
        ++substr;
        --sublen;
    }

    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE YEAR (digits until the '-' character) */
    out->year = 0;
    while (sublen > 0 && isdigit(*substr)) {
        out->year = 10 * out->year + (*substr - '0');
        ++substr;
        --sublen;
    }

    /* Negate the year if necessary */
    if (str[0] == '-') {
        out->year = -out->year;
    }
    /* Check whether it's a leap-year */
    year_leap = is_leapyear(out->year);

    /* Next character must be a '-' or the end of the string */
    if (sublen == 0) {
        bestunit = NPY_FR_Y;
        goto finish;
    }
    else if (*substr == '-') {
        ++substr;
        --sublen;
    }
    else {
        goto parse_error;
    }

    /* Can't have a trailing '-' */
    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE MONTH (2 digits) */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        out->month = 10 * (substr[0] - '0') + (substr[1] - '0');

        if (out->month < 1 || out->month > 12) {
            PyErr_Format(PyExc_ValueError,
                        "Month out of range in datetime string \"%s\"", str);
            goto error;
        }
        substr += 2;
        sublen -= 2;
    }
    else {
        goto parse_error;
    }

    /* Next character must be a '-' or the end of the string */
    if (sublen == 0) {
        bestunit = NPY_FR_M;
        goto finish;
    }
    else if (*substr == '-') {
        ++substr;
        --sublen;
    }
    else {
        goto parse_error;
    }

    /* Can't have a trailing '-' */
    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE DAY (2 digits) */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        out->day = 10 * (substr[0] - '0') + (substr[1] - '0');

        if (out->day < 1 ||
                    out->day > _days_per_month_table[year_leap][out->month-1]) {
            PyErr_Format(PyExc_ValueError,
                        "Day out of range in datetime string \"%s\"", str);
            goto error;
        }
        substr += 2;
        sublen -= 2;
    }
    else {
        goto parse_error;
    }

    /* Next character must be a 'T', ' ', or end of string */
    if (sublen == 0) {
        bestunit = NPY_FR_D;
        goto finish;
    }
    else if (*substr != 'T' && *substr != ' ') {
        goto parse_error;
    }
    else {
        ++substr;
        --sublen;
    }

    /* PARSE THE HOURS (2 digits) */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        out->hour = 10 * (substr[0] - '0') + (substr[1] - '0');

        if (out->hour >= 24) {
            PyErr_Format(PyExc_ValueError,
                        "Hours out of range in datetime string \"%s\"", str);
            goto error;
        }
        substr += 2;
        sublen -= 2;
    }
    else {
        goto parse_error;
    }

    /* Next character must be a ':' or the end of the string */
    if (sublen > 0 && *substr == ':') {
        ++substr;
        --sublen;
    }
    else {
        bestunit = NPY_FR_h;
        goto parse_timezone;
    }

    /* Can't have a trailing ':' */
    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE MINUTES (2 digits) */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        out->min = 10 * (substr[0] - '0') + (substr[1] - '0');

        if (out->min >= 60) {
            PyErr_Format(PyExc_ValueError,
                        "Minutes out of range in datetime string \"%s\"", str);
            goto error;
        }
        substr += 2;
        sublen -= 2;
    }
    else {
        goto parse_error;
    }

    /* Next character must be a ':' or the end of the string */
    if (sublen > 0 && *substr == ':') {
        ++substr;
        --sublen;
    }
    else {
        bestunit = NPY_FR_m;
        goto parse_timezone;
    }

    /* Can't have a trailing ':' */
    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE SECONDS (2 digits) */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        out->sec = 10 * (substr[0] - '0') + (substr[1] - '0');

        if (out->sec >= 60) {
            PyErr_Format(PyExc_ValueError,
                        "Seconds out of range in datetime string \"%s\"", str);
            goto error;
        }
        substr += 2;
        sublen -= 2;
    }
    else {
        goto parse_error;
    }

    /* Next character may be a '.' indicating fractional seconds */
    if (sublen > 0 && *substr == '.') {
        ++substr;
        --sublen;
    }
    else {
        bestunit = NPY_FR_s;
        goto parse_timezone;
    }

    /* PARSE THE MICROSECONDS (0 to 6 digits) */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        out->us *= 10;
        if (sublen > 0  && isdigit(*substr)) {
            out->us += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    if (sublen == 0 || !isdigit(*substr)) {
        if (numdigits > 3) {
            bestunit = NPY_FR_us;
        }
        else {
            bestunit = NPY_FR_ms;
        }
        goto parse_timezone;
    }

    /* PARSE THE PICOSECONDS (0 to 6 digits) */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        out->ps *= 10;
        if (sublen > 0 && isdigit(*substr)) {
            out->ps += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    if (sublen == 0 || !isdigit(*substr)) {
        if (numdigits > 3) {
            bestunit = NPY_FR_ps;
        }
        else {
            bestunit = NPY_FR_ns;
        }
        goto parse_timezone;
    }

    /* PARSE THE ATTOSECONDS (0 to 6 digits) */
    numdigits = 0;
    for (i = 0; i < 6; ++i) {
        out->as *= 10;
        if (sublen > 0 && isdigit(*substr)) {
            out->as += (*substr - '0');
            ++substr;
            --sublen;
            ++numdigits;
        }
    }

    if (numdigits > 3) {
        bestunit = NPY_FR_as;
    }
    else {
        bestunit = NPY_FR_fs;
    }

parse_timezone:
    if (sublen == 0) {
        goto finish;
    }
    else {
        /* 2016-01-14, 1.11 */
        PyErr_Clear();
        if (DEPRECATE(
                "parsing timezone aware datetimes is deprecated; "
                "this will raise an error in the future") < 0) {
            return -1;
        }
    }

    /* UTC specifier */
    if (*substr == 'Z') {
        if (sublen == 1) {
            goto finish;
        }
        else {
            ++substr;
            --sublen;
        }
    }
    /* Time zone offset */
    else if (*substr == '-' || *substr == '+') {
        int offset_neg = 0, offset_hour = 0, offset_minute = 0;

        if (*substr == '-') {
            offset_neg = 1;
        }
        ++substr;
        --sublen;

        /* The hours offset */
        if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
            offset_hour = 10 * (substr[0] - '0') + (substr[1] - '0');
            substr += 2;
            sublen -= 2;
            if (offset_hour >= 24) {
                PyErr_Format(PyExc_ValueError,
                            "Timezone hours offset out of range "
                            "in datetime string \"%s\"", str);
                goto error;
            }
        }
        else {
            goto parse_error;
        }

        /* The minutes offset is optional */
        if (sublen > 0) {
            /* Optional ':' */
            if (*substr == ':') {
                ++substr;
                --sublen;
            }

            /* The minutes offset (at the end of the string) */
            if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
                offset_minute = 10 * (substr[0] - '0') + (substr[1] - '0');
                substr += 2;
                sublen -= 2;
                if (offset_minute >= 60) {
                    PyErr_Format(PyExc_ValueError,
                                "Timezone minutes offset out of range "
                                "in datetime string \"%s\"", str);
                    goto error;
                }
            }
            else {
                goto parse_error;
            }
        }

        /* Apply the time zone offset */
        if (offset_neg) {
            offset_hour = -offset_hour;
            offset_minute = -offset_minute;
        }
        add_minutes_to_datetimestruct(out, -60 * offset_hour - offset_minute);
    }

    /* Skip trailing whitespace */
    while (sublen > 0 && isspace(*substr)) {
        ++substr;
        --sublen;
    }

    if (sublen != 0) {
        goto parse_error;
    }

finish:
    if (out_bestunit != NULL) {
        *out_bestunit = bestunit;
    }

    /* Check the casting rule */
    if (unit != NPY_FR_ERROR &&
            !can_cast_datetime64_units(bestunit, unit, casting)) {
        PyErr_Format(PyExc_TypeError, "Cannot parse \"%s\" as unit "
                     "'%s' using casting rule %s",
                     str, _datetime_strings[unit],
                     npy_casting_to_string(casting));
        return -1;
    }

    return 0;

parse_error:
    PyErr_Format(PyExc_ValueError,
            "Error parsing datetime string \"%s\" at position %zd",
            str, substr - str);
    return -1;

error:
    return -1;
}

/*
 * Provides a string length to use for converting datetime
 * objects with the given local and unit settings.
 */
NPY_NO_EXPORT int
get_datetime_iso_8601_strlen(int local, NPY_DATETIMEUNIT base)
{
    int len = 0;

    switch (base) {
        case NPY_FR_ERROR:
            /* If no unit is provided, return the maximum length */
            return NPY_DATETIME_MAX_ISO8601_STRLEN;
        case NPY_FR_GENERIC:
            /* Generic units can only be used to represent NaT */
            return 4;
        case NPY_FR_as:
            len += 3;  /* "###" */
        case NPY_FR_fs:
            len += 3;  /* "###" */
        case NPY_FR_ps:
            len += 3;  /* "###" */
        case NPY_FR_ns:
            len += 3;  /* "###" */
        case NPY_FR_us:
            len += 3;  /* "###" */
        case NPY_FR_ms:
            len += 4;  /* ".###" */
        case NPY_FR_s:
            len += 3;  /* ":##" */
        case NPY_FR_m:
            len += 3;  /* ":##" */
        case NPY_FR_h:
            len += 3;  /* "T##" */
        case NPY_FR_D:
        case NPY_FR_W:
            len += 3;  /* "-##" */
        case NPY_FR_M:
            len += 3;  /* "-##" */
        case NPY_FR_Y:
            len += 21; /* 64-bit year */
            break;
    }

    if (base >= NPY_FR_h) {
        if (local) {
            len += 5;  /* "+####" or "-####" */
        }
        else {
            len += 1;  /* "Z" */
        }
    }

    len += 1; /* NULL terminator */

    return len;
}

/*
 * Finds the largest unit whose value is nonzero, and for which
 * the remainder for the rest of the units is zero.
 */
static NPY_DATETIMEUNIT
lossless_unit_from_datetimestruct(npy_datetimestruct *dts)
{
    if (dts->as % 1000 != 0) {
        return NPY_FR_as;
    }
    else if (dts->as != 0) {
        return NPY_FR_fs;
    }
    else if (dts->ps % 1000 != 0) {
        return NPY_FR_ps;
    }
    else if (dts->ps != 0) {
        return NPY_FR_ns;
    }
    else if (dts->us % 1000 != 0) {
        return NPY_FR_us;
    }
    else if (dts->us != 0) {
        return NPY_FR_ms;
    }
    else if (dts->sec != 0) {
        return NPY_FR_s;
    }
    else if (dts->min != 0) {
        return NPY_FR_m;
    }
    else if (dts->hour != 0) {
        return NPY_FR_h;
    }
    else if (dts->day != 1) {
        return NPY_FR_D;
    }
    else if (dts->month != 1) {
        return NPY_FR_M;
    }
    else {
        return NPY_FR_Y;
    }
}

/*
 * Converts an npy_datetimestruct to an (almost) ISO 8601
 * NULL-terminated string. If the string fits in the space exactly,
 * it leaves out the NULL terminator and returns success.
 *
 * The differences from ISO 8601 are the 'NaT' string, and
 * the number of year digits is >= 4 instead of strictly 4.
 *
 * If 'local' is non-zero, it produces a string in local time with
 * a +-#### timezone offset. If 'local' is zero and 'utc' is non-zero,
 * produce a string ending with 'Z' to denote UTC. By default, no time
 * zone information is attached.
 *
 * 'base' restricts the output to that unit. Set 'base' to
 * -1 to auto-detect a base after which all the values are zero.
 *
 *  'tzoffset' is used if 'local' is enabled, and 'tzoffset' is
 *  set to a value other than -1. This is a manual override for
 *  the local time zone to use, as an offset in minutes.
 *
 *  'casting' controls whether data loss is allowed by truncating
 *  the data to a coarser unit. This interacts with 'local', slightly,
 *  in order to form a date unit string as a local time, the casting
 *  must be unsafe.
 *
 *  Returns 0 on success, -1 on failure (for example if the output
 *  string was too short).
 */
NPY_NO_EXPORT int
make_iso_8601_datetime(npy_datetimestruct *dts, char *outstr, npy_intp outlen,
                    int local, int utc, NPY_DATETIMEUNIT base, int tzoffset,
                    NPY_CASTING casting)
{
    npy_datetimestruct dts_local;
    int timezone_offset = 0;

    char *substr = outstr;
    npy_intp sublen = outlen;
    npy_intp tmplen;

    /* Handle NaT, and treat a datetime with generic units as NaT */
    if (dts->year == NPY_DATETIME_NAT || base == NPY_FR_GENERIC) {
        if (outlen < 3) {
            goto string_too_short;
        }
        outstr[0] = 'N';
        outstr[1] = 'a';
        outstr[2] = 'T';
        if (outlen > 3) {
            outstr[3] = '\0';
        }

        return 0;
    }

    /*
     * Only do local time within a reasonable year range. The years
     * earlier than 1970 are not made local, because the Windows API
     * raises an error when they are attempted (see the comments above the
     * get_localtime() function). For consistency, this
     * restriction is applied to all platforms.
     *
     * Note that this only affects how the datetime becomes a string.
     * The result is still completely unambiguous, it only means
     * that datetimes outside this range will not include a time zone
     * when they are printed.
     */
    if ((dts->year < 1970 || dts->year >= 10000) && tzoffset == -1) {
        local = 0;
    }

    /* Automatically detect a good unit */
    if (base == NPY_FR_ERROR) {
        base = lossless_unit_from_datetimestruct(dts);
        /*
         * If there's a timezone, use at least minutes precision,
         * and never split up hours and minutes by default
         */
        if ((base < NPY_FR_m && local) || base == NPY_FR_h) {
            base = NPY_FR_m;
        }
        /* Don't split up dates by default */
        else if (base < NPY_FR_D) {
            base = NPY_FR_D;
        }
    }
    /*
     * Print weeks with the same precision as days.
     *
     * TODO: Could print weeks with YYYY-Www format if the week
     *       epoch is a Monday.
     */
    else if (base == NPY_FR_W) {
        base = NPY_FR_D;
    }

    /* Use the C API to convert from UTC to local time */
    if (local && tzoffset == -1) {
        if (convert_datetimestruct_utc_to_local(&dts_local, dts,
                                                &timezone_offset) < 0) {
            return -1;
        }

        /* Set dts to point to our local time instead of the UTC time */
        dts = &dts_local;
    }
    /* Use the manually provided tzoffset */
    else if (local) {
        /* Make a copy of the npy_datetimestruct we can modify */
        dts_local = *dts;
        dts = &dts_local;

        /* Set and apply the required timezone offset */
        timezone_offset = tzoffset;
        add_minutes_to_datetimestruct(dts, timezone_offset);
    }

    /*
     * Now the datetimestruct data is in the final form for
     * the string representation, so ensure that the data
     * is being cast according to the casting rule.
     */
    if (casting != NPY_UNSAFE_CASTING) {
        /* Producing a date as a local time is always 'unsafe' */
        if (base <= NPY_FR_D && local) {
            PyErr_SetString(PyExc_TypeError, "Cannot create a local "
                        "timezone-based date string from a NumPy "
                        "datetime without forcing 'unsafe' casting");
            return -1;
        }
        /* Only 'unsafe' and 'same_kind' allow data loss */
        else {
            NPY_DATETIMEUNIT unitprec;

            unitprec = lossless_unit_from_datetimestruct(dts);
            if (casting != NPY_SAME_KIND_CASTING && unitprec > base) {
                PyErr_Format(PyExc_TypeError, "Cannot create a "
                            "string with unit precision '%s' "
                            "from the NumPy datetime, which has data at "
                            "unit precision '%s', "
                            "requires 'unsafe' or 'same_kind' casting",
                             _datetime_strings[base],
                             _datetime_strings[unitprec]);
                return -1;
            }
        }
    }

    /* YEAR */
    /*
     * Can't use PyOS_snprintf, because it always produces a '\0'
     * character at the end, and NumPy string types are permitted
     * to have data all the way to the end of the buffer.
     */
#ifdef _WIN32
    tmplen = _snprintf(substr, sublen, "%04" NPY_INT64_FMT, dts->year);
#else
    tmplen = snprintf(substr, sublen, "%04" NPY_INT64_FMT, dts->year);
#endif
    /* If it ran out of space or there isn't space for the NULL terminator */
    if (tmplen < 0 || tmplen > sublen) {
        goto string_too_short;
    }
    substr += tmplen;
    sublen -= tmplen;

    /* Stop if the unit is years */
    if (base == NPY_FR_Y) {
        if (sublen > 0) {
            *substr = '\0';
        }
        return 0;
    }

    /* MONTH */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = '-';
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->month / 10) + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->month % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is months */
    if (base == NPY_FR_M) {
        if (sublen > 0) {
            *substr = '\0';
        }
        return 0;
    }

    /* DAY */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = '-';
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->day / 10) + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->day % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is days */
    if (base == NPY_FR_D) {
        if (sublen > 0) {
            *substr = '\0';
        }
        return 0;
    }

    /* HOUR */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = 'T';
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->hour / 10) + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->hour % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is hours */
    if (base == NPY_FR_h) {
        goto add_time_zone;
    }

    /* MINUTE */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = ':';
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->min / 10) + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->min % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is minutes */
    if (base == NPY_FR_m) {
        goto add_time_zone;
    }

    /* SECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = ':';
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->sec / 10) + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->sec % 10) + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is seconds */
    if (base == NPY_FR_s) {
        goto add_time_zone;
    }

    /* MILLISECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = '.';
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->us / 100000) % 10 + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->us / 10000) % 10 + '0');
    if (sublen < 4 ) {
        goto string_too_short;
    }
    substr[3] = (char)((dts->us / 1000) % 10 + '0');
    substr += 4;
    sublen -= 4;

    /* Stop if the unit is milliseconds */
    if (base == NPY_FR_ms) {
        goto add_time_zone;
    }

    /* MICROSECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->us / 100) % 10 + '0');
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->us / 10) % 10 + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)(dts->us % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is microseconds */
    if (base == NPY_FR_us) {
        goto add_time_zone;
    }

    /* NANOSECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->ps / 100000) % 10 + '0');
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->ps / 10000) % 10 + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->ps / 1000) % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is nanoseconds */
    if (base == NPY_FR_ns) {
        goto add_time_zone;
    }

    /* PICOSECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->ps / 100) % 10 + '0');
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->ps / 10) % 10 + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)(dts->ps % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is picoseconds */
    if (base == NPY_FR_ps) {
        goto add_time_zone;
    }

    /* FEMTOSECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->as / 100000) % 10 + '0');
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->as / 10000) % 10 + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->as / 1000) % 10 + '0');
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is femtoseconds */
    if (base == NPY_FR_fs) {
        goto add_time_zone;
    }

    /* ATTOSECOND */
    if (sublen < 1 ) {
        goto string_too_short;
    }
    substr[0] = (char)((dts->as / 100) % 10 + '0');
    if (sublen < 2 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->as / 10) % 10 + '0');
    if (sublen < 3 ) {
        goto string_too_short;
    }
    substr[2] = (char)(dts->as % 10 + '0');
    substr += 3;
    sublen -= 3;

add_time_zone:
    if (local) {
        /* Add the +/- sign */
        if (sublen < 1) {
            goto string_too_short;
        }
        if (timezone_offset < 0) {
            substr[0] = '-';
            timezone_offset = -timezone_offset;
        }
        else {
            substr[0] = '+';
        }
        substr += 1;
        sublen -= 1;

        /* Add the timezone offset */
        if (sublen < 1 ) {
            goto string_too_short;
        }
        substr[0] = (char)((timezone_offset / (10*60)) % 10 + '0');
        if (sublen < 2 ) {
            goto string_too_short;
        }
        substr[1] = (char)((timezone_offset / 60) % 10 + '0');
        if (sublen < 3 ) {
            goto string_too_short;
        }
        substr[2] = (char)(((timezone_offset % 60) / 10) % 10 + '0');
        if (sublen < 4 ) {
            goto string_too_short;
        }
        substr[3] = (char)((timezone_offset % 60) % 10 + '0');
        substr += 4;
        sublen -= 4;
    }
    /* UTC "Zulu" time */
    else if (utc) {
        if (sublen < 1) {
            goto string_too_short;
        }
        substr[0] = 'Z';
        substr += 1;
        sublen -= 1;
    }

    /* Add a NULL terminator, and return */
    if (sublen > 0) {
        substr[0] = '\0';
    }

    return 0;

string_too_short:
    PyErr_Format(PyExc_RuntimeError,
                "The string provided for NumPy ISO datetime formatting "
                "was too short, with length %"NPY_INTP_FMT,
                outlen);
    return -1;
}


/*
 * This is the Python-exposed datetime_as_string function.
 */
NPY_NO_EXPORT PyObject *
array_datetime_as_string(PyObject *NPY_UNUSED(self), PyObject *args,
                                PyObject *kwds)
{
    PyObject *arr_in = NULL, *unit_in = NULL, *timezone_obj = NULL;
    NPY_DATETIMEUNIT unit;
    NPY_CASTING casting = NPY_SAME_KIND_CASTING;

    int local = 0;
    int utc = 0;
    PyArray_DatetimeMetaData *meta;
    int strsize;

    PyArrayObject *ret = NULL;

    NpyIter *iter = NULL;
    PyArrayObject *op[2] = {NULL, NULL};
    PyArray_Descr *op_dtypes[2] = {NULL, NULL};
    npy_uint32 flags, op_flags[2];

    static char *kwlist[] = {"arr", "unit", "timezone", "casting", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds,
                                "O|OOO&:datetime_as_string", kwlist,
                                &arr_in,
                                &unit_in,
                                &timezone_obj,
                                &PyArray_CastingConverter, &casting)) {
        return NULL;
    }

    /* Claim a reference to timezone for later */
    Py_XINCREF(timezone_obj);

    op[0] = (PyArrayObject *)PyArray_FROM_O(arr_in);
    if (op[0] == NULL) {
        goto fail;
    }
    if (PyArray_DESCR(op[0])->type_num != NPY_DATETIME) {
        PyErr_SetString(PyExc_TypeError,
                    "input must have type NumPy datetime");
        goto fail;
    }

    /* Get the datetime metadata */
    meta = get_datetime_metadata_from_dtype(PyArray_DESCR(op[0]));
    if (meta == NULL) {
        goto fail;
    }

    /* Use the metadata's unit for printing by default */
    unit = meta->base;

    /* Parse the input unit if provided */
    if (unit_in != NULL && unit_in != Py_None) {
        PyObject *strobj;
        char *str = NULL;
        Py_ssize_t len = 0;

        if (PyUnicode_Check(unit_in)) {
            strobj = PyUnicode_AsASCIIString(unit_in);
            if (strobj == NULL) {
                goto fail;
            }
        }
        else {
            strobj = unit_in;
            Py_INCREF(strobj);
        }

        if (PyBytes_AsStringAndSize(strobj, &str, &len) < 0) {
            Py_DECREF(strobj);
            goto fail;
        }

        /*
         * unit == NPY_FR_ERROR means to autodetect the unit
         * from the datetime data
         * */
        if (strcmp(str, "auto") == 0) {
            unit = NPY_FR_ERROR;
        }
        else {
            unit = parse_datetime_unit_from_string(str, len, NULL);
            if (unit == NPY_FR_ERROR) {
                Py_DECREF(strobj);
                goto fail;
            }
        }
        Py_DECREF(strobj);

        if (unit != NPY_FR_ERROR &&
                !can_cast_datetime64_units(meta->base, unit, casting)) {
            PyErr_Format(PyExc_TypeError, "Cannot create a datetime "
                        "string as units '%s' from a NumPy datetime "
                        "with units '%s' according to the rule %s",
                        _datetime_strings[unit],
                        _datetime_strings[meta->base],
                         npy_casting_to_string(casting));
            goto fail;
        }
    }

    /* Get the input time zone */
    if (timezone_obj != NULL) {
        /* Convert to ASCII if it's unicode */
        if (PyUnicode_Check(timezone_obj)) {
            /* accept unicode input */
            PyObject *obj_str;
            obj_str = PyUnicode_AsASCIIString(timezone_obj);
            if (obj_str == NULL) {
                goto fail;
            }
            Py_DECREF(timezone_obj);
            timezone_obj = obj_str;
        }

        /* Check for the supported string inputs */
        if (PyBytes_Check(timezone_obj)) {
            char *str;
            Py_ssize_t len;

            if (PyBytes_AsStringAndSize(timezone_obj, &str, &len) < 0) {
                goto fail;
            }

            if (strcmp(str, "local") == 0) {
                local = 1;
                utc = 0;
                Py_DECREF(timezone_obj);
                timezone_obj = NULL;
            }
            else if (strcmp(str, "UTC") == 0) {
                local = 0;
                utc = 1;
                Py_DECREF(timezone_obj);
                timezone_obj = NULL;
            }
            else if (strcmp(str, "naive") == 0) {
                local = 0;
                utc = 0;
                Py_DECREF(timezone_obj);
                timezone_obj = NULL;
            }
            else {
                PyErr_Format(PyExc_ValueError, "Unsupported timezone "
                            "input string \"%s\"", str);
                goto fail;
            }
        }
        /* Otherwise assume it's a Python TZInfo, or acts like one */
        else {
            local = 1;
        }
    }

    /* Get a string size long enough for any datetimes we're given */
    strsize = get_datetime_iso_8601_strlen(local, unit);
#if defined(NPY_PY3K)
    /*
     * For Python3, allocate the output array as a UNICODE array, so
     * that it will behave as strings properly
     */
    op_dtypes[1] = PyArray_DescrNewFromType(NPY_UNICODE);
    if (op_dtypes[1] == NULL) {
        goto fail;
    }
    op_dtypes[1]->elsize = strsize * 4;
    /* This steals the UNICODE dtype reference in op_dtypes[1] */
    op[1] = (PyArrayObject *)PyArray_NewLikeArray(op[0],
                                        NPY_KEEPORDER, op_dtypes[1], 1);
    if (op[1] == NULL) {
        op_dtypes[1] = NULL;
        goto fail;
    }
#endif
    /* Create the iteration string data type (always ASCII string) */
    op_dtypes[1] = PyArray_DescrNewFromType(NPY_STRING);
    if (op_dtypes[1] == NULL) {
        goto fail;
    }
    op_dtypes[1]->elsize = strsize;

    flags = NPY_ITER_ZEROSIZE_OK|
            NPY_ITER_BUFFERED;
    op_flags[0] = NPY_ITER_READONLY|
                  NPY_ITER_ALIGNED;
    op_flags[1] = NPY_ITER_WRITEONLY|
                  NPY_ITER_ALLOCATE;

    iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                            op_flags, op_dtypes);
    if (iter == NULL) {
        goto fail;
    }

    if (NpyIter_GetIterSize(iter) != 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_datetime dt;
        npy_datetimestruct dts;

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);

        do {
            int tzoffset = -1;

            /* Get the datetime */
            dt = *(npy_datetime *)dataptr[0];

            /* Convert it to a struct */
            if (convert_datetime_to_datetimestruct(meta, dt, &dts) < 0) {
                goto fail;
            }

            /* Get the tzoffset from the timezone if provided */
            if (local && timezone_obj != NULL) {
                tzoffset = get_tzoffset_from_pytzinfo(timezone_obj, &dts);
                if (tzoffset == -1) {
                    goto fail;
                }
            }

            /* Zero the destination string completely */
            memset(dataptr[1], 0, strsize);
            /* Convert that into a string */
            if (make_iso_8601_datetime(&dts, (char *)dataptr[1], strsize,
                                local, utc, unit, tzoffset, casting) < 0) {
                goto fail;
            }
        } while(iternext(iter));
    }

    ret = NpyIter_GetOperandArray(iter)[1];
    Py_INCREF(ret);

    Py_XDECREF(timezone_obj);
    Py_XDECREF(op[0]);
    Py_XDECREF(op[1]);
    Py_XDECREF(op_dtypes[0]);
    Py_XDECREF(op_dtypes[1]);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    return PyArray_Return(ret);

fail:
    Py_XDECREF(timezone_obj);
    Py_XDECREF(op[0]);
    Py_XDECREF(op[1]);
    Py_XDECREF(op_dtypes[0]);
    Py_XDECREF(op_dtypes[1]);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    return NULL;
}
