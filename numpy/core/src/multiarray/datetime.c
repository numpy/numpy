#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <datetime.h>

#include <time.h>

#define _MULTIARRAYMODULE
#include <numpy/arrayobject.h>

#include "npy_config.h"

#include "numpy/npy_3kcompat.h"

#include "numpy/arrayscalars.h"
#include "_datetime.h"

/*
 * Imports the PyDateTime functions so we can create these objects.
 * This is called during module initialization
 */
NPY_NO_EXPORT void
numpy_pydatetime_import()
{
    PyDateTime_IMPORT;
}

static int
is_leapyear(npy_int64 year);

/* Exported as DATETIMEUNITS in multiarraymodule.c */
NPY_NO_EXPORT char *_datetime_strings[] = {
    NPY_STR_Y,
    NPY_STR_M,
    NPY_STR_W,
    NPY_STR_B,
    NPY_STR_D,
    NPY_STR_h,
    NPY_STR_m,
    NPY_STR_s,
    NPY_STR_ms,
    NPY_STR_us,
    NPY_STR_ns,
    NPY_STR_ps,
    NPY_STR_fs,
    NPY_STR_as,
    "generic"
};

/*
  ====================================================
    }
  == Beginning of section borrowed from mx.DateTime ==
  ====================================================
*/

/*
 * Functions in the following section are borrowed from mx.DateTime version
 * 2.0.6, and hence this code is subject to the terms of the egenix public
 * license version 1.0.0
 */

/* Table of number of days in a month (0-based, without and with leap) */
static int days_in_month[2][12] = {
    { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
    { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};

/*
 * Return the day of the week for the given absolute date.
 * Monday is 0 and Sunday is 6
 */
static int
day_of_week(npy_longlong absdate)
{
    /* Add in four for the Thursday on Jan 1, 1970 (epoch offset)*/
    absdate += 4;

    if (absdate >= 0) {
        return absdate % 7;
    }
    else {
        return 6 + (absdate + 1) % 7;
    }
}

/*
  ====================================================
  == End of section adapted from mx.DateTime       ==
  ====================================================
*/


static int
is_leapyear(npy_int64 year)
{
    return (year & 0x3) == 0 && /* year % 4 == 0 */
           ((year % 100) != 0 ||
            (year % 400) == 0);
}

/*
 * Calculates the days offset from the 1970 epoch.
 */
static npy_int64
get_datetimestruct_days(const npy_datetimestruct *dts)
{
    int i, month;
    npy_int64 year, days = 0;
    int *month_lengths;

    year = dts->year - 1970;
    days = year * 365;

    /* Adjust for leap years */
    if (days >= 0) {
        /*
         * 1968 is the closest leap year before 1970.
         * Exclude the current year, so add 1.
         */
        year += 1;
        /* Add one day for each 4 years */
        days += year / 4;
        /* 1900 is the closest previous year divisible by 100 */
        year += 68;
        /* Subtract one day for each 100 years */
        days -= year / 100;
        /* 1600 is the closest previous year divisible by 400 */
        year += 300;
        /* Add one day for each 400 years */
        days += year / 400;
    }
    else {
        /*
         * 1972 is the closest later year after 1970.
         * Include the current year, so subtract 2.
         */
        year -= 2;
        /* Subtract one day for each 4 years */
        days += year / 4;
        /* 2000 is the closest later year divisible by 100 */
        year -= 28;
        /* Add one day for each 100 years */
        days -= year / 100;
        /* 2000 is also the closest later year divisible by 400 */
        /* Subtract one day for each 400 years */
        days += year / 400;
    }

    month_lengths = days_in_month[is_leapyear(dts->year)];
    month = dts->month - 1;

    /* Add the months */
    for (i = 0; i < month; ++i) {
        days += month_lengths[i];
    }

    /* Add the days */
    days += dts->day - 1;

    return days;
}

/*
 * Modifies '*days_' to be the day offset within the year,
 * and returns the year.
 */
static npy_int64
days_to_yearsdays(npy_int64 *days_)
{
    const npy_int64 days_per_400years = (400*365 + 100 - 4 + 1);
    /* Adjust so it's relative to the year 2000 (divisible by 400) */
    npy_int64 days = (*days_) - (365*30 + 7), year;

    /* Break down the 400 year cycle to get the year and day within the year */
    if (days >= 0) {
        year = 400 * (days / days_per_400years);
        days = days % days_per_400years;
    }
    else {
        year = 400 * ((days - (days_per_400years - 1)) / days_per_400years);
        days = days % days_per_400years;
        if (days < 0) {
            days += days_per_400years;
        }
    }

    /* Work out the year/day within the 400 year cycle */
    if (days >= 366) {
        year += 100 * ((days-1) / (100*365 + 25 - 1));
        days = (days-1) % (100*365 + 25 - 1);
        if (days >= 365) {
            year += 4 * ((days+1) / (4*365 + 1));
            days = (days+1) % (4*365 + 1);
            if (days >= 366) {
                year += (days-1) / 365;
                days = (days-1) % 365;
            }
        }
    }

    *days_ = days;
    return year + 2000;
}

/*
 * Fills in the year, month, day in 'dts' based on the days
 * offset from 1970.
 */
static void
set_datetimestruct_days(npy_int64 days, npy_datetimestruct *dts)
{
    int *month_lengths, i;

    dts->year = days_to_yearsdays(&days);

    month_lengths = days_in_month[is_leapyear(dts->year)];
    for (i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            dts->month = i + 1;
            dts->day = days + 1;
            return;
        }
        else {
            days -= month_lengths[i];
        }
    }
}

/*
 * Converts a datetime from a datetimestruct to a datetime based
 * on some metadata. The date is assumed to be valid.
 *
 * TODO: If meta->num is really big, there could be overflow
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_datetimestruct_to_datetime(PyArray_DatetimeMetaData *meta,
                                    const npy_datetimestruct *dts,
                                    npy_datetime *out)
{
    npy_datetime ret;
    NPY_DATETIMEUNIT base = meta->base;

    /* If the datetimestruct is NaT, return NaT */
    if (dts->year == NPY_DATETIME_NAT) {
        *out = NPY_DATETIME_NAT;
        return 0;
    }

    /* Cannot instantiate a datetime with generic units */
    if (meta->base == NPY_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
                    "Cannot create a NumPy datetime other than NaT "
                    "with generic units");
        return -1;
    }

    if (dts->event < 0 || dts->event >= meta->events) {
        PyErr_Format(PyExc_ValueError,
                    "NumPy datetime event %d is outside range [0,%d)",
                    (int)dts->event, (int)meta->events);
        return -1;
    }

    if (base == NPY_FR_Y) {
        /* Truncate to the year */
        ret = dts->year - 1970;
    }
    else if (base == NPY_FR_M) {
        /* Truncate to the month */
        ret = 12 * (dts->year - 1970) + (dts->month - 1);
    }
    else {
        /* Otherwise calculate the number of days to start */
        npy_int64 days = get_datetimestruct_days(dts);
        int dotw;

        switch (base) {
            case NPY_FR_W:
                /* Truncate to weeks */
                if (days >= 0) {
                    ret = days / 7;
                }
                else {
                    ret = (days - 6) / 7;
                }
                break;
            case NPY_FR_B:
                /* TODO: this needs work... */
                dotw = day_of_week(days);

                if (dotw > 4) {
                    /* Invalid business day */
                    ret = 0;
                }
                else {
                    npy_int64 x;
                    if (days >= 0) {
                        /* offset to adjust first week */
                        x = days - 4;
                    }
                    else {
                        x = days - 2;
                    }
                    ret = 2 + (x / 7) * 5 + x % 7;
                }
                break;
            case NPY_FR_D:
                ret = days;
                break;
            case NPY_FR_h:
                ret = days * 24 +
                      dts->hour;
                break;
            case NPY_FR_m:
                ret = (days * 24 +
                      dts->hour) * 60 +
                      dts->min;
                break;
            case NPY_FR_s:
                ret = ((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec;
                break;
            case NPY_FR_ms:
                ret = (((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000 +
                      dts->us / 1000;
                break;
            case NPY_FR_us:
                ret = (((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us;
                break;
            case NPY_FR_ns:
                ret = ((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000 +
                      dts->ps / 1000;
                break;
            case NPY_FR_ps:
                ret = ((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps;
                break;
            case NPY_FR_fs:
                /* only 2.6 hours */
                ret = (((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps) * 1000 +
                      dts->as / 1000;
                break;
            case NPY_FR_as:
                /* only 9.2 secs */
                ret = (((((days * 24 +
                      dts->hour) * 60 +
                      dts->min) * 60 +
                      dts->sec) * 1000000 +
                      dts->us) * 1000000 +
                      dts->ps) * 1000000 +
                      dts->as;
                break;
            default:
                /* Something got corrupted */
                PyErr_SetString(PyExc_ValueError,
                        "NumPy datetime metadata with corrupt unit value");
                return -1;
        }
    }

    /* Divide by the multiplier */
    if (meta->num > 1) {
        if (ret >= 0) {
            ret /= meta->num;
        }
        else {
            ret = (ret - meta->num + 1) / meta->num;
        }
    }

    /* Add in the event number if needed */
    if (meta->events > 1) {
        /* Multiply by the number of events and put in the event number */
        ret = ret * meta->events + dts->event;
    }

    *out = ret;

    return 0;
}

/*NUMPY_API
 * Create a datetime value from a filled datetime struct and resolution unit.
 *
 * TO BE REMOVED - NOT USED INTERNALLY.
 */
NPY_NO_EXPORT npy_datetime
PyArray_DatetimeStructToDatetime(NPY_DATETIMEUNIT fr, npy_datetimestruct *d)
{
    npy_datetime ret;
    PyArray_DatetimeMetaData meta;

    /* Set up a dummy metadata for the conversion */
    meta.base = fr;
    meta.num = 1;
    meta.events = 1;

    if (convert_datetimestruct_to_datetime(&meta, d, &ret) < 0) {
        /* The caller then needs to check PyErr_Occurred() */
        return -1;
    }

    return ret;
}

/* Uses Average values when frequency is Y, M, or B */

#define _DAYS_PER_MONTH 30.436875
#define _DAYS_PER_YEAR  365.2425

/*NUMPY_API
 * Create a timdelta value from a filled timedelta struct and resolution unit.
 *
 * TO BE REMOVED - NOT USED INTERNALLY.
 */
NPY_NO_EXPORT npy_datetime
PyArray_TimedeltaStructToTimedelta(NPY_DATETIMEUNIT fr, npy_timedeltastruct *d)
{
    npy_datetime ret;

    if (fr == NPY_FR_Y) {
        ret = d->day / _DAYS_PER_YEAR;
    }
    else if (fr == NPY_FR_M) {
        ret = d->day / _DAYS_PER_MONTH;
    }
    else if (fr == NPY_FR_W) {
        /* This is just 7-days for now. */
        if (d->day >= 0) {
            ret = d->day / 7;
        }
        else {
            ret = (d->day - 6) / 7;
        }
    }
    else if (fr == NPY_FR_B) {
        /*
         * What is the meaning of a relative Business day?
         *
         * This assumes you want to take the day difference and
         * convert it to business-day difference (removing 2 every 7).
         */
        ret = (d->day / 7) * 5 + d->day % 7;
    }
    else if (fr == NPY_FR_D) {
        ret = d->day;
    }
    else if (fr == NPY_FR_h) {
        ret = d->day + d->sec / 3600;
    }
    else if (fr == NPY_FR_m) {
        ret = d->day * (npy_int64)(1440) + d->sec / 60;
    }
    else if (fr == NPY_FR_s) {
        ret = d->day * (npy_int64)(86400) + d->sec;
    }
    else if (fr == NPY_FR_ms) {
        ret = d->day * (npy_int64)(86400000) + d->sec * 1000 + d->us / 1000;
    }
    else if (fr == NPY_FR_us) {
        npy_int64 num = 86400000;
        num *= (npy_int64)(1000);
        ret = d->day * num + d->sec * (npy_int64)(1000000) + d->us;
    }
    else if (fr == NPY_FR_ns) {
        npy_int64 num = 86400000;
        num *= (npy_int64)(1000000);
        ret = d->day * num + d->sec * (npy_int64)(1000000000)
            + d->us * (npy_int64)(1000) + (d->ps / 1000);
    }
    else if (fr == NPY_FR_ps) {
        npy_int64 num2, num1;

        num2 = 1000000;
        num2 *= (npy_int64)(1000000);
        num1 = (npy_int64)(86400) * num2;

        ret = d->day * num1 + d->sec * num2 + d->us * (npy_int64)(1000000)
            + d->ps;
    }
    else if (fr == NPY_FR_fs) {
        /* only 2.6 hours */
        npy_int64 num2 = 1000000000;
        num2 *= (npy_int64)(1000000);
        ret = d->sec * num2 + d->us * (npy_int64)(1000000000)
            + d->ps * (npy_int64)(1000) + (d->as / 1000);
    }
    else if (fr == NPY_FR_as) {
        /* only 9.2 secs */
        npy_int64 num1, num2;

        num1 = 1000000;
        num1 *= (npy_int64)(1000000);
        num2 = num1 * (npy_int64)(1000000);
        ret = d->sec * num2 + d->us * num1 + d->ps * (npy_int64)(1000000)
            + d->as;
    }
    else {
        /* Shouldn't get here */
        PyErr_SetString(PyExc_ValueError, "invalid internal frequency");
        ret = -1;
    }

    return ret;
}

/*
 * Converts a datetime based on the given metadata into a datetimestruct
 */
NPY_NO_EXPORT int
convert_datetime_to_datetimestruct(PyArray_DatetimeMetaData *meta,
                                    npy_datetime dt,
                                    npy_datetimestruct *out)
{
    npy_int64 absdays;
    npy_int64 perday;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(npy_datetimestruct));
    out->year = 1970;
    out->month = 1;
    out->day = 1;

    /* NaT is signaled in the year */
    if (dt == NPY_DATETIME_NAT) {
        out->year = NPY_DATETIME_NAT;
        return 0;
    }

    /* Datetimes can't be in generic units */
    if (meta->base == NPY_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
                    "Cannot convert a NumPy datetime value other than NaT "
                    "with generic units");
        return -1;
    }
    
    /* Extract the event number */
    if (meta->events > 1) {
        out->event = dt % meta->events;
        dt = dt / meta->events;
        if (out->event < 0) {
            out->event += meta->events;
            --dt;
        }
    }

    /* TODO: Change to a mechanism that avoids the potential overflow */
    dt *= meta->num;

    /*
     * Note that care must be taken with the / and % operators
     * for negative values.
     */
    switch (meta->base) {
        case NPY_FR_Y:
            out->year = 1970 + dt;
            break;

        case NPY_FR_M:
            if (dt >= 0) {
                out->year  = 1970 + dt / 12;
                out->month = dt % 12 + 1;
            }
            else {
                out->year  = 1969 + (dt + 1) / 12;
                out->month = 12 + (dt + 1)% 12;
            }
            break;

        case NPY_FR_W:
            /* A week is 7 days */
            set_datetimestruct_days(dt * 7, out);
            break;

        case NPY_FR_B:
            /* TODO: fix up business days */
            /* Number of business days since Thursday, 1-1-70 */
            /*
             * A business day is M T W Th F (i.e. all but Sat and Sun.)
             * Convert the business day to the number of actual days.
             *
             * Must convert [0,1,2,3,4,5,6,7,...] to
             *                  [0,1,4,5,6,7,8,11,...]
             * and  [...,-9,-8,-7,-6,-5,-4,-3,-2,-1,0] to
             *        [...,-13,-10,-9,-8,-7,-6,-3,-2,-1,0]
             */
            if (dt >= 0) {
                absdays = 7 * ((dt + 3) / 5) + ((dt + 3) % 5) - 3;
            }
            else {
                /* Recall how C computes / and % with negative numbers */
                absdays = 7 * ((dt - 1) / 5) + ((dt - 1) % 5) + 1;
            }
            set_datetimestruct_days(absdays, out);
            break;

        case NPY_FR_D:
            set_datetimestruct_days(dt, out);
            break;

        case NPY_FR_h:
            perday = 24LL;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt;
            break;

        case NPY_FR_m:
            perday = 24LL * 60;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt / 60;
            out->min = dt % 60;
            break;

        case NPY_FR_s:
            perday = 24LL * 60 * 60;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt / (60*60);
            out->min = (dt / 60) % 60;
            out->sec = dt % 60;
            break;

        case NPY_FR_ms:
            perday = 24LL * 60 * 60 * 1000;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt / (60*60*1000LL);
            out->min = (dt / (60*1000LL)) % 60;
            out->sec = (dt / 1000LL) % 60;
            out->us = (dt % 1000LL) * 1000;
            break;

        case NPY_FR_us:
            perday = 24LL * 60LL * 60LL * 1000LL * 1000LL;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt / (60*60*1000000LL);
            out->min = (dt / (60*1000000LL)) % 60;
            out->sec = (dt / 1000000LL) % 60;
            out->us = dt % 1000000LL;
            break;

        case NPY_FR_ns:
            perday = 24LL * 60LL * 60LL * 1000LL * 1000LL * 1000LL;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt / (60*60*1000000000LL);
            out->min = (dt / (60*1000000000LL)) % 60;
            out->sec = (dt / 1000000000LL) % 60;
            out->us = (dt / 1000LL) % 1000000LL;
            out->ps = (dt % 1000LL) * 1000;
            break;

        case NPY_FR_ps:
            perday = 24LL * 60 * 60 * 1000 * 1000 * 1000 * 1000;

            if (dt >= 0) {
                set_datetimestruct_days(dt / perday, out);
                dt  = dt % perday;
            }
            else {
                set_datetimestruct_days((dt - (perday-1)) / perday, out);
                dt = (perday-1) + (dt + 1) % perday;
            }
            out->hour = dt / (60*60*1000000000000LL);
            out->min = (dt / (60*1000000000000LL)) % 60;
            out->sec = (dt / 1000000000000LL) % 60;
            out->us = (dt / 1000000LL) % 1000000LL;
            out->ps = dt % 1000000LL;
            break;

        case NPY_FR_fs:
            /* entire range is only +- 2.6 hours */
            if (dt >= 0) {
                out->hour = dt / (60*60*1000000000000000LL);
                out->min = (dt / (60*1000000000000000LL)) % 60;
                out->sec = (dt / 1000000000000000LL) % 60;
                out->us = (dt / 1000000000LL) % 1000000LL;
                out->ps = (dt / 1000LL) % 1000000LL;
                out->as = (dt % 1000LL) * 1000;
            }
            else {
                npy_datetime minutes;

                minutes = dt / (60*1000000000000000LL);
                dt = dt % (60*1000000000000000LL);
                if (dt < 0) {
                    dt += (60*1000000000000000LL);
                    --minutes;
                }
                /* Offset the negative minutes */
                add_minutes_to_datetimestruct(out, minutes);
                out->sec = (dt / 1000000000000000LL) % 60;
                out->us = (dt / 1000000000LL) % 1000000LL;
                out->ps = (dt / 1000LL) % 1000000LL;
                out->as = (dt % 1000LL) * 1000;
            }
            break;

        case NPY_FR_as:
            /* entire range is only +- 9.2 seconds */
            if (dt >= 0) {
                out->sec = (dt / 1000000000000000000LL) % 60;
                out->us = (dt / 1000000000000LL) % 1000000LL;
                out->ps = (dt / 1000000LL) % 1000000LL;
                out->as = dt % 1000000LL;
            }
            else {
                npy_datetime seconds;

                seconds = dt / 1000000000000000000LL;
                dt = dt % 1000000000000000000LL;
                if (dt < 0) {
                    dt += 1000000000000000000LL;
                    --seconds;
                }
                /* Offset the negative seconds */
                add_seconds_to_datetimestruct(out, seconds);
                out->us = (dt / 1000000000000LL) % 1000000LL;
                out->ps = (dt / 1000000LL) % 1000000LL;
                out->as = dt % 1000000LL;
            }
            break;

        default:
            PyErr_SetString(PyExc_RuntimeError,
                        "NumPy datetime metadata is corrupted with invalid "
                        "base unit");
            return -1;
    }

    return 0;
}


/*NUMPY_API
 * Fill the datetime struct from the value and resolution unit.
 *
 * TO BE REMOVED - NOT USED INTERNALLY.
 */
NPY_NO_EXPORT void
PyArray_DatetimeToDatetimeStruct(npy_datetime val, NPY_DATETIMEUNIT fr,
                                 npy_datetimestruct *result)
{
    PyArray_DatetimeMetaData meta;

    /* Set up a dummy metadata for the conversion */
    meta.base = fr;
    meta.num = 1;
    meta.events = 1;

    if (convert_datetime_to_datetimestruct(&meta, val, result) < 0) {
        /* The caller needs to check PyErr_Occurred() */
        return;
    }

    return;
}

/*
 * FIXME: Overflow is not handled at all
 *   To convert from Years, Months, and Business Days,
 *   multiplication by the average is done
 */

/*NUMPY_API
 * Fill the timedelta struct from the timedelta value and resolution unit.
 *
 * TO BE REMOVED - NOT USED INTERNALLY.
 */
NPY_NO_EXPORT void
PyArray_TimedeltaToTimedeltaStruct(npy_timedelta val, NPY_DATETIMEUNIT fr,
                                 npy_timedeltastruct *result)
{
    npy_longlong day=0;
    int sec=0, us=0, ps=0, as=0;
    npy_bool negative=0;

    /*
     * Note that what looks like val / N and val % N for positive
     * numbers maps to [val - (N-1)] / N and [N-1 + (val+1) % N]
     * for negative numbers (with the 2nd value, the remainder,
     * being positive in both cases).
     */

    if (val < 0) {
        val = -val;
        negative = 1;
    }
    if (fr == NPY_FR_Y) {
        day = val * _DAYS_PER_YEAR;
    }
    else if (fr == NPY_FR_M) {
        day = val * _DAYS_PER_MONTH;
    }
    else if (fr == NPY_FR_W) {
        day = val * 7;
    }
    else if (fr == NPY_FR_B) {
        /* Number of business days since Thursday, 1-1-70 */
        day = (val * 7) / 5;
    }
    else if (fr == NPY_FR_D) {
        day = val;
    }
    else if (fr == NPY_FR_h) {
        day = val / 24;
        sec = (val % 24)*3600;
    }
    else if (fr == NPY_FR_m) {
        day = val / 1440;
        sec = (val % 1440)*60;
    }
    else if (fr == NPY_FR_s) {
        day = val / (86400);
        sec = val % 86400;
    }
    else if (fr == NPY_FR_ms) {
        day = val / 86400000;
        val = val % 86400000;
        sec = val / 1000;
        us = (val % 1000)*1000;
    }
    else if (fr == NPY_FR_us) {
        npy_int64 num1;
        num1 = 86400000;
        num1 *= 1000;
        day = val / num1;
        us = val % num1;
        sec = us / 1000000;
        us = us % 1000000;
    }
    else if (fr == NPY_FR_ns) {
        npy_int64 num1;
        num1 = 86400000;
        num1 *= 1000000;
        day = val / num1;
        val = val % num1;
        sec = val / 1000000000;
        val = val % 1000000000;
        us  = val / 1000;
        ps  = (val % 1000) * (npy_int64)(1000);
    }
    else if (fr == NPY_FR_ps) {
        npy_int64 num1, num2;
        num2 = 1000000000;
        num2 *= (npy_int64)(1000);
        num1 = (npy_int64)(86400) * num2;

        day = val / num1;
        ps = val % num1;
        sec = ps / num2;
        ps = ps % num2;
        us = ps / 1000000;
        ps = ps % 1000000;
    }
    else if (fr == NPY_FR_fs) {
        /* entire range is only += 9.2 hours */
        npy_int64 num1, num2;
        num1 = 1000000000;
        num2 = num1 * (npy_int64)(1000000);

        day = 0;
        sec = val / num2;
        val = val % num2;
        us = val / num1;
        val = val % num1;
        ps = val / 1000;
        as = (val % 1000) * (npy_int64)(1000);
    }
    else if (fr == NPY_FR_as) {
        /* entire range is only += 2.6 seconds */
        npy_int64 num1, num2, num3;
        num1 = 1000000;
        num2 = num1 * (npy_int64)(1000000);
        num3 = num2 * (npy_int64)(1000000);
        day = 0;
        sec = val / num3;
        as = val % num3;
        us = as / num2;
        as = as % num2;
        ps = as / num1;
        as = as % num1;
    }
    else {
        PyErr_SetString(PyExc_RuntimeError, "invalid internal time resolution");
    }

    if (negative) {
        result->day = -day;
        result->sec = -sec;
        result->us = -us;
        result->ps = -ps;
        result->as = -as;
    }
    else {
        result->day   = day;
        result->sec   = sec;
        result->us    = us;
        result->ps    = ps;
        result->as    = as;
    }
    return;
}

/*
 * This function returns the a new reference to the
 * capsule with the datetime metadata.
 */
NPY_NO_EXPORT PyObject *
get_datetime_metacobj_from_dtype(PyArray_Descr *dtype)
{
    PyObject *metacobj;

    /* Check that the dtype has metadata */
    if (dtype->metadata == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Datetime type object is invalid, lacks metadata");
        return NULL;
    }

    /* Check that the dtype has unit metadata */
    metacobj = PyDict_GetItemString(dtype->metadata, NPY_METADATA_DTSTR);
    if (metacobj == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Datetime type object is invalid, lacks unit metadata");
        return NULL;
    }

    Py_INCREF(metacobj);
    return metacobj;
}

/*
 * This function returns a pointer to the DateTimeMetaData
 * contained within the provided datetime dtype.
 */
NPY_NO_EXPORT PyArray_DatetimeMetaData *
get_datetime_metadata_from_dtype(PyArray_Descr *dtype)
{
    PyObject *metacobj;
    PyArray_DatetimeMetaData *meta = NULL;

    metacobj = get_datetime_metacobj_from_dtype(dtype);
    if (metacobj == NULL) {
        return NULL;
    }

    /* Check that the dtype has an NpyCapsule for the metadata */
    meta = (PyArray_DatetimeMetaData *)NpyCapsule_AsVoidPtr(metacobj);
    if (meta == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Datetime type object is invalid, unit metadata is corrupt");
        return NULL;
    }

    return meta;
}

/*
 * Converts a substring given by 'str' and 'len' into
 * a date time unit multiplier + enum value, which are populated
 * into out_meta. Other metadata is left along.
 *
 * 'metastr' is only used in the error message, and may be NULL.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
parse_datetime_extended_unit_from_string(char *str, Py_ssize_t len,
                                    char *metastr,
                                    PyArray_DatetimeMetaData *out_meta)
{
    char *substr = str, *substrend = NULL;
    int den = 1;

    /* First comes an optional integer multiplier */
    out_meta->num = (int)strtol(substr, &substrend, 10);
    if (substr == substrend) {
        out_meta->num = 1;
    }
    substr = substrend;

    /* Next comes the unit itself, followed by either '/' or the string end */
    substrend = substr;
    while (substrend-str < len && *substrend != '/') {
        ++substrend;
    }
    if (substr == substrend) {
        goto bad_input;
    }
    out_meta->base = parse_datetime_unit_from_string(substr,
                                        substrend-substr, metastr);
    if (out_meta->base == -1) {
        return -1;
    }
    substr = substrend;

    /* Next comes an optional integer denominator */
    if (substr-str < len && *substr == '/') {
        substr++;
        den = (int)strtol(substr, &substrend, 10);
        /* If the '/' exists, there must be a number followed by ']' */
        if (substr == substrend || *substrend != ']') {
            goto bad_input;
        }
        substr = substrend + 1;
    }
    else if (substr-str != len) {
        goto bad_input;
    }

    if (den != 1) {
        if (convert_datetime_divisor_to_multiple(
                                out_meta, den, metastr) < 0) {
            return -1;
        }
    }

    return 0;

bad_input:
    if (metastr != NULL) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\" at position %d",
                metastr, (int)(substr-metastr));
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\"",
                str);
    }

    return -1;
}

/*
 * Parses the metadata string into the metadata C structure.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
parse_datetime_metadata_from_metastr(char *metastr, Py_ssize_t len,
                                    PyArray_DatetimeMetaData *out_meta)
{
    char *substr = metastr, *substrend = NULL;

    /* Treat the empty string as generic units */
    if (len == 0) {
        out_meta->base = NPY_FR_GENERIC;
        out_meta->num = 1;
        out_meta->events = 1;

        return 0;
    }

    /* The metadata string must start with a '[' */
    if (len < 3 || *substr++ != '[') {
        goto bad_input;
    }

    substrend = substr;
    while (*substrend != '\0' && *substrend != ']') {
        ++substrend;
    }
    if (*substrend == '\0' || substr == substrend) {
        substr = substrend;
        goto bad_input;
    }

    /* Parse the extended unit inside the [] */
    if (parse_datetime_extended_unit_from_string(substr, substrend-substr,
                                                    metastr, out_meta) < 0) {
        return -1;
    }

    substr = substrend+1;

    /* Finally comes an optional number of events */
    if (substr[0] == '/' && substr[1] == '/') {
        substr += 2;

        out_meta->events = (int)strtol(substr, &substrend, 10);
        if (substr == substrend || *substrend != '\0') {
            goto bad_input;
        }
    }
    else if (*substr != '\0') {
        goto bad_input;
    }
    else {
        out_meta->events = 1;
    }

    return 0;

bad_input:
    if (substr != metastr) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\" at position %d",
                metastr, (int)(substr-metastr));
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime metadata string \"%s\"",
                metastr);
    }

    return -1;
}

NPY_NO_EXPORT PyObject *
parse_datetime_metacobj_from_metastr(char *metastr, Py_ssize_t len)
{
    PyArray_DatetimeMetaData *dt_data;

    dt_data = PyArray_malloc(sizeof(PyArray_DatetimeMetaData));
    if (dt_data == NULL) {
        return PyErr_NoMemory();
    }

    /* If there's no metastr, use the default */
    if (len == 0) {
        dt_data->base = NPY_DATETIME_DEFAULTUNIT;
        dt_data->num = 1;
        dt_data->events = 1;
    }
    else {
        if (parse_datetime_metadata_from_metastr(metastr, len, dt_data) < 0) {
            PyArray_free(dt_data);
            return NULL;
        }
    }

    return NpyCapsule_FromVoidPtr((void *)dt_data, simple_capsule_dtor);
}

/*
 * Converts a datetype dtype string into a dtype descr object.
 * The "type" string should be NULL-terminated.
 */
NPY_NO_EXPORT PyArray_Descr *
parse_dtype_from_datetime_typestr(char *typestr, Py_ssize_t len)
{
    PyArray_Descr *dtype = NULL;
    char *metastr = NULL;
    int is_timedelta = 0;
    Py_ssize_t metalen = 0;
    PyObject *metacobj = NULL;

    if (len < 2) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime typestr \"%s\"",
                typestr);
        return NULL;
    }

    /*
     * First validate that the root is correct,
     * and get the metadata string address
     */
    if (typestr[0] == 'm' && typestr[1] == '8') {
        is_timedelta = 1;
        metastr = typestr + 2;
        metalen = len - 2;
    }
    else if (typestr[0] == 'M' && typestr[1] == '8') {
        is_timedelta = 0;
        metastr = typestr + 2;
        metalen = len - 2;
    }
    else if (len >= 11 && strncmp(typestr, "timedelta64", 11) == 0) {
        is_timedelta = 1;
        metastr = typestr + 11;
        metalen = len - 11;
    }
    else if (len >= 10 && strncmp(typestr, "datetime64", 10) == 0) {
        is_timedelta = 0;
        metastr = typestr + 10;
        metalen = len - 10;
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime typestr \"%s\"",
                typestr);
        return NULL;
    }

    /* Create a default datetime or timedelta */
    if (is_timedelta) {
        dtype = PyArray_DescrNewFromType(NPY_TIMEDELTA);
    }
    else {
        dtype = PyArray_DescrNewFromType(NPY_DATETIME);
    }
    if (dtype == NULL) {
        return NULL;
    }

    /*
     * Remove any reference to old metadata dictionary
     * And create a new one for this new dtype
     */
    Py_XDECREF(dtype->metadata);
    dtype->metadata = PyDict_New();
    if (dtype->metadata == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }

    /* Parse the metadata string into a metadata capsule */
    metacobj = parse_datetime_metacobj_from_metastr(metastr, metalen);
    if (metacobj == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }

    /* Set the metadata object in the dictionary. */
    if (PyDict_SetItemString(dtype->metadata, NPY_METADATA_DTSTR,
                                                    metacobj) < 0) {
        Py_DECREF(dtype);
        Py_DECREF(metacobj);
        return NULL;
    }
    Py_DECREF(metacobj);

    return dtype;
}

static NPY_DATETIMEUNIT _multiples_table[16][4] = {
    {12, 52, 365},                            /* NPY_FR_Y */
    {NPY_FR_M, NPY_FR_W, NPY_FR_D},
    {4,  30, 720},                            /* NPY_FR_M */
    {NPY_FR_W, NPY_FR_D, NPY_FR_h},
    {5,  7,  168, 10080},                     /* NPY_FR_W */
    {NPY_FR_B, NPY_FR_D, NPY_FR_h, NPY_FR_m},
    {24, 1440, 86400},                        /* NPY_FR_B */
    {NPY_FR_h, NPY_FR_m, NPY_FR_s},
    {24, 1440, 86400},                        /* NPY_FR_D */
    {NPY_FR_h, NPY_FR_m, NPY_FR_s},
    {60, 3600},                               /* NPY_FR_h */
    {NPY_FR_m, NPY_FR_s},
    {60, 60000},                              /* NPY_FR_m */
    {NPY_FR_s, NPY_FR_ms},
    {1000, 1000000},                          /* >=NPY_FR_s */
    {0, 0}
};



/*
 * Translate divisors into multiples of smaller units.
 * 'metastr' is used for the error message if the divisor doesn't work,
 * and can be NULL if the metadata didn't come from a string.
 *
 * This function only affects the 'base' and 'num' values in the metadata.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_datetime_divisor_to_multiple(PyArray_DatetimeMetaData *meta,
                                    int den, char *metastr)
{
    int i, num, ind;
    NPY_DATETIMEUNIT *totry;
    NPY_DATETIMEUNIT *baseunit;
    int q, r;

    if (meta->base == NPY_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
            "Can't use 'den' divisor with generic units");
        return -1;
    }

    ind = ((int)meta->base - (int)NPY_FR_Y)*2;
    totry = _multiples_table[ind];
    baseunit = _multiples_table[ind + 1];

    num = 3;
    if (meta->base == NPY_FR_W) {
        num = 4;
    }
    else if (meta->base > NPY_FR_D) {
        num = 2;
    }
    if (meta->base >= NPY_FR_s) {
        ind = ((int)NPY_FR_s - (int)NPY_FR_Y)*2;
        totry = _multiples_table[ind];
        baseunit = _multiples_table[ind + 1];
        baseunit[0] = meta->base + 1;
        baseunit[1] = meta->base + 2;
        if (meta->base == NPY_FR_as - 1) {
            num = 1;
        }
        if (meta->base == NPY_FR_as) {
            num = 0;
        }
    }

    for (i = 0; i < num; i++) {
        q = totry[i] / den;
        r = totry[i] % den;
        if (r == 0) {
            break;
        }
    }
    if (i == num) {
        if (metastr == NULL) {
            PyErr_Format(PyExc_ValueError,
                    "divisor (%d) is not a multiple of a lower-unit "
                    "in datetime metadata", den);
        }
        else {
            PyErr_Format(PyExc_ValueError,
                    "divisor (%d) is not a multiple of a lower-unit "
                    "in datetime metadata \"%s\"", den, metastr);
        }
        return -1;
    }
    meta->base = baseunit[i];
    meta->num *= q;

    return 0;
}

/*
 * Lookup table for factors between datetime units, except
 * for years, months, and business days.
 */
static npy_uint32
_datetime_factors[] = {
    1,  /* Years - not used */
    1,  /* Months - not used */
    7,  /* Weeks -> Days */
    1,  /* Business days - not used */
    24, /* Days -> Hours */
    60, /* Hours -> Minutes */
    60, /* Minutes -> Seconds */
    1000,
    1000,
    1000,
    1000,
    1000,
    1000,
    1,   /* Attoseconds are the smallest base unit */
    0    /* Generic units don't have a conversion */
};

/*
 * Returns the scale factor between the units. Does not validate
 * that bigbase represents larger units than littlebase, or that
 * the units are not generic.
 *
 * Returns 0 if there is an overflow.
 */
static npy_uint64
get_datetime_units_factor(NPY_DATETIMEUNIT bigbase, NPY_DATETIMEUNIT littlebase)
{
    npy_uint64 factor = 1;
    int unit = (int)bigbase;
    while (littlebase > unit) {
        factor *= _datetime_factors[unit];
        /*
         * Detect overflow by disallowing the top 16 bits to be 1.
         * That alows a margin of error much bigger than any of
         * the datetime factors.
         */
        if (factor&0xff00000000000000ULL) {
            return 0;
        }
        ++unit;
    }
    return factor;
}

/* Euclidean algorithm on two positive numbers */
static npy_uint64
_uint64_euclidean_gcd(npy_uint64 x, npy_uint64 y)
{
    npy_uint64 tmp;

    if (x > y) {
        tmp = x;
        x = y;
        y = tmp;
    }
    while (x != y && y != 0) {
        tmp = x % y;
        x = y;
        y = tmp;
    }

    return x;
}

/*
 * Computes the conversion factor to convert data with 'src_meta' metadata
 * into data with 'dst_meta' metadata, not taking into account the events.
 *
 * To convert a npy_datetime or npy_timedelta, first the event number needs
 * to be divided away, then it needs to be scaled by num/denom, and
 * finally the event number can be added back in.
 *
 * If overflow occurs, both out_num and out_denom are set to 0, but
 * no error is set.
 */
NPY_NO_EXPORT void
get_datetime_conversion_factor(PyArray_DatetimeMetaData *src_meta,
                                PyArray_DatetimeMetaData *dst_meta,
                                npy_int64 *out_num, npy_int64 *out_denom)
{
    int src_base, dst_base, swapped;
    npy_uint64 num = 1, denom = 1, tmp, gcd;

    /* Generic units change to the destination with no conversion factor */
    if (src_meta->base == NPY_FR_GENERIC) {
        *out_num = 1;
        *out_denom = 1;
        return;
    }
    /*
     * Converting to a generic unit from something other than a generic
     * unit is an error.
     */
    else if (dst_meta->base == NPY_FR_GENERIC) {
        PyErr_SetString(PyExc_ValueError,
                    "Cannot convert from specific units to generic "
                    "units in NumPy datetimes or timedeltas");
        *out_num = 0;
        *out_denom = 0;
        return;
    }

    if (src_meta->base <= dst_meta->base) {
        src_base = src_meta->base;
        dst_base = dst_meta->base;
        swapped = 0;
    }
    else {
        src_base = dst_meta->base;
        dst_base = src_meta->base;
        swapped = 1;
    }

    if (src_base != dst_base) {
        /*
         * Conversions between years/months and other units use
         * the factor averaged over the 400 year leap year cycle.
         */
        if (src_base == NPY_FR_Y) {
            if (dst_base == NPY_FR_M) {
                num *= 12;
            }
            else if (dst_base == NPY_FR_W) {
                num *= (97 + 400*365);
                denom *= 400*7;
            }
            else {
                /* Year -> Day */
                num *= (97 + 400*365);
                denom *= 400;
                /* Day -> dst_base */
                num *= get_datetime_units_factor(NPY_FR_D, dst_base);
            }
        }
        else if (src_base == NPY_FR_M) {
            if (dst_base == NPY_FR_W) {
                num *= (97 + 400*365);
                denom *= 400*12*7;
            }
            else {
                /* Month -> Day */
                num *= (97 + 400*365);
                denom *= 400*12;
                /* Day -> dst_base */
                num *= get_datetime_units_factor(NPY_FR_D, dst_base);
            }
        }
        else {
            num *= get_datetime_units_factor(src_base, dst_base);
        }
    }

    /* If something overflowed, make both num and denom 0 */
    if (denom == 0) {
        PyErr_Format(PyExc_OverflowError,
                    "Integer overflow getting a conversion factor between "
                    "NumPy datetime unit %s and %s",
                    _datetime_strings[src_base],
                    _datetime_strings[dst_base]);
        *out_num = 0;
        *out_denom = 0;
        return;
    }

    /* Swap the numerator and denominator if necessary */
    if (swapped) {
        tmp = num;
        num = denom;
        denom = tmp;
    }

    num *= src_meta->num;
    denom *= dst_meta->num;

    /* Return as a fraction in reduced form */
    gcd = _uint64_euclidean_gcd(num, denom);
    *out_num = (npy_int64)(num / gcd);
    *out_denom = (npy_int64)(denom / gcd);
}

/*
 * Determines whether the 'divisor' metadata divides evenly into
 * the 'dividend' metadata.
 */
NPY_NO_EXPORT npy_bool
datetime_metadata_divides(
                        PyArray_Descr *dividend,
                        PyArray_Descr *divisor,
                        int strict_with_nonlinear_units)
{
    PyArray_DatetimeMetaData *meta1, *meta2;
    npy_uint64 num1, num2;

    /* Must be datetime types */
    if ((dividend->type_num != NPY_DATETIME &&
                        dividend->type_num != NPY_TIMEDELTA) ||
                    (divisor->type_num != NPY_DATETIME &&
                        divisor->type_num != NPY_TIMEDELTA)) {
        return 0;
    }

    meta1 = get_datetime_metadata_from_dtype(dividend);
    if (meta1 == NULL) {
        PyErr_Clear();
        return 0;
    }
    meta2 = get_datetime_metadata_from_dtype(divisor);
    if (meta2 == NULL) {
        PyErr_Clear();
        return 0;
    }

    /* Generic units divide into anything */
    if (meta2->base == NPY_FR_GENERIC) {
        return 1;
    }
    /* Non-generic units never divide into generic units */
    else if (meta1->base == NPY_FR_GENERIC) {
        return 0;
    }

    /* Events must match */
    if (meta1->events != meta2->events) {
        return 0;
    }

    num1 = (npy_uint64)meta1->num;
    num2 = (npy_uint64)meta2->num;

    /* If the bases are different, factor in a conversion */
    if (meta1->base != meta2->base) {
        /*
         * Years, Months, and Business days are incompatible with
         * all other units (except years and months are compatible
         * with each other).
         */
        if (meta1->base == NPY_FR_B || meta2->base == NPY_FR_B) {
            return 0;
        }
        else if (meta1->base == NPY_FR_Y) {
            if (meta2->base == NPY_FR_M) {
                num1 *= 12;
            }
            else if (strict_with_nonlinear_units) {
                return 0;
            }
            else {
                /* Could do something complicated here */
                return 1;
            }
        }
        else if (meta2->base == NPY_FR_Y) {
            if (meta1->base == NPY_FR_M) {
                num2 *= 12;
            }
            else if (strict_with_nonlinear_units) {
                return 0;
            }
            else {
                /* Could do something complicated here */
                return 1;
            }
        }
        else if (meta1->base == NPY_FR_M || meta2->base == NPY_FR_M) {
            if (strict_with_nonlinear_units) {
                return 0;
            }
            else {
                /* Could do something complicated here */
                return 1;
            }
        }

        /* Take the greater base (unit sizes are decreasing in enum) */
        if (meta1->base > meta2->base) {
            num2 *= get_datetime_units_factor(meta2->base, meta1->base);
            if (num2 == 0) {
                return 0;
            }
        }
        else {
            num1 *= get_datetime_units_factor(meta1->base, meta2->base);
            if (num1 == 0) {
                return 0;
            }
        }
    }

    /* Crude, incomplete check for overflow */
    if (num1&0xff00000000000000LL || num2&0xff00000000000000LL ) {
        return 0;
    }

    return (num1 % num2) == 0;
}

/*
 * Computes the GCD of the two date-time metadata values. Raises
 * an exception if there is no reasonable GCD, such as with
 * years and days.
 *
 * The result is placed in 'out_meta'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
compute_datetime_metadata_greatest_common_divisor(
                        PyArray_DatetimeMetaData *meta1,
                        PyArray_DatetimeMetaData *meta2,
                        PyArray_DatetimeMetaData *out_meta,
                        int strict_with_nonlinear_units1,
                        int strict_with_nonlinear_units2)
{
    NPY_DATETIMEUNIT base;
    npy_uint64 num1, num2, num;
    int events = 1;

    /* If either unit is generic, adopt the metadata from the other one */
    if (meta1->base == NPY_FR_GENERIC) {
        *out_meta = *meta2;
        return 0;
    }
    else if (meta2->base == NPY_FR_GENERIC) {
        *out_meta = *meta1;
        return 0;
    }

    /* Take the maximum of the events */
    if (meta1->events > meta2->events) {
        events = meta1->events;
    }
    else {
        events = meta2->events;
    }

    num1 = (npy_uint64)meta1->num;
    num2 = (npy_uint64)meta2->num;

    /* First validate that the units have a reasonable GCD */
    if (meta1->base == meta2->base) {
        base = meta1->base;
    }
    else {
        /*
         * Years, Months, and Business days are incompatible with
         * all other units (except years and months are compatible
         * with each other).
         */
        if (meta1->base == NPY_FR_Y) {
            if (meta2->base == NPY_FR_M) {
                base = NPY_FR_M;
                num1 *= 12;
            }
            else if (strict_with_nonlinear_units1) {
                goto incompatible_units;
            }
            else {
                base = meta2->base;
                /* Don't multiply num1 since there is no even factor */
            }
        }
        else if (meta2->base == NPY_FR_Y) {
            if (meta1->base == NPY_FR_M) {
                base = NPY_FR_M;
                num2 *= 12;
            }
            else if (strict_with_nonlinear_units2) {
                goto incompatible_units;
            }
            else {
                base = meta1->base;
                /* Don't multiply num2 since there is no even factor */
            }
        }
        else if (meta1->base == NPY_FR_M) {
            if (strict_with_nonlinear_units1) {
                goto incompatible_units;
            }
            else {
                base = meta2->base;
                /* Don't multiply num1 since there is no even factor */
            }
        }
        else if (meta2->base == NPY_FR_M) {
            if (strict_with_nonlinear_units2) {
                goto incompatible_units;
            }
            else {
                base = meta1->base;
                /* Don't multiply num2 since there is no even factor */
            }
        }
        else if (meta1->base == NPY_FR_B || meta2->base == NPY_FR_B) {
            if (strict_with_nonlinear_units1 || strict_with_nonlinear_units2) {
                goto incompatible_units;
            }
            else {
                if (meta1->base > meta2->base) {
                    base = meta1->base;
                }
                else {
                    base = meta2->base;
                }

                /*
                 * When combining business days with other units, end
                 * up with days instead of business days.
                 */
                if (base == NPY_FR_B) {
                    base = NPY_FR_D;
                }
            }
        }

        /* Take the greater base (unit sizes are decreasing in enum) */
        if (meta1->base > meta2->base) {
            base = meta1->base;
            num2 *= get_datetime_units_factor(meta2->base, meta1->base);
            if (num2 == 0) {
                goto units_overflow;
            }
        }
        else {
            base = meta2->base;
            num1 *= get_datetime_units_factor(meta1->base, meta2->base);
            if (num1 == 0) {
                goto units_overflow;
            }
        }
    }

    /* Compute the GCD of the resulting multipliers */
    num = _uint64_euclidean_gcd(num1, num2);

    /* Fill the 'out_meta' values */
    out_meta->base = base;
    out_meta->num = (int)num;
    if (out_meta->num <= 0 || num != (npy_uint64)out_meta->num) {
        goto units_overflow;
    }
    out_meta->events = events;

    return 0;

incompatible_units: {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Cannot get "
                    "a common metadata divisor for "
                    "NumPy datetime metadata ");
        errmsg = append_metastr_to_string(meta1, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" and "));
        errmsg = append_metastr_to_string(meta2, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" because they have "
                    "incompatible nonlinear base time units"));
        PyErr_SetObject(PyExc_TypeError, errmsg);
        return -1;
    }
units_overflow: {
        PyObject *errmsg;
        errmsg = PyUString_FromString("Integer overflow "
                    "getting a common metadata divisor for "
                    "NumPy datetime metadata ");
        errmsg = append_metastr_to_string(meta1, 0, errmsg);
        PyUString_ConcatAndDel(&errmsg,
                PyUString_FromString(" and "));
        errmsg = append_metastr_to_string(meta2, 0, errmsg);
        PyErr_SetObject(PyExc_OverflowError, errmsg);
        return -1;
    }
}

/*
 * Computes the GCD of the two date-time metadata values. Raises
 * an exception if there is no reasonable GCD, such as with
 * years and days.
 *
 * Returns a capsule with the GCD metadata.
 */
NPY_NO_EXPORT PyObject *
compute_datetime_metadata_greatest_common_divisor_capsule(
                        PyArray_Descr *type1,
                        PyArray_Descr *type2,
                        int strict_with_nonlinear_units1,
                        int strict_with_nonlinear_units2)
{
    PyArray_DatetimeMetaData *meta1, *meta2, *dt_data;

    if ((type1->type_num != NPY_DATETIME &&
                        type1->type_num != NPY_TIMEDELTA) ||
                    (type2->type_num != NPY_DATETIME &&
                        type2->type_num != NPY_TIMEDELTA)) {
        PyErr_SetString(PyExc_TypeError,
                "Require datetime types for metadata "
                "greatest common divisor operation");
        return NULL;
    }

    meta1 = get_datetime_metadata_from_dtype(type1);
    if (meta1 == NULL) {
        return NULL;
    }
    meta2 = get_datetime_metadata_from_dtype(type2);
    if (meta2 == NULL) {
        return NULL;
    }

    /* Create and return the metadata capsule */
    dt_data = PyArray_malloc(sizeof(PyArray_DatetimeMetaData));
    if (dt_data == NULL) {
        return PyErr_NoMemory();
    }

    if (compute_datetime_metadata_greatest_common_divisor(meta1, meta2,
                            dt_data, strict_with_nonlinear_units1,
                            strict_with_nonlinear_units2) < 0) {
        PyArray_free(dt_data);
        return NULL;
    }

    return NpyCapsule_FromVoidPtr((void *)dt_data, simple_capsule_dtor);
}

/*
 * Both type1 and type2 must be either NPY_DATETIME or NPY_TIMEDELTA.
 * Applies the type promotion rules between the two types, returning
 * the promoted type.
 */
NPY_NO_EXPORT PyArray_Descr *
datetime_type_promotion(PyArray_Descr *type1, PyArray_Descr *type2)
{
    int type_num1, type_num2;
    PyObject *gcdmeta;
    PyArray_Descr *dtype;
    int is_datetime;

    type_num1 = type1->type_num;
    type_num2 = type2->type_num;

    is_datetime = (type_num1 == NPY_DATETIME || type_num2 == NPY_DATETIME);

    /*
     * Get the metadata GCD, being strict about nonlinear units for
     * timedelta and relaxed for datetime.
     */
    gcdmeta = compute_datetime_metadata_greatest_common_divisor_capsule(
                                            type1, type2,
                                            type_num1 == NPY_TIMEDELTA,
                                            type_num2 == NPY_TIMEDELTA);
    if (gcdmeta == NULL) {
        return NULL;
    }

    /* Create a DATETIME or TIMEDELTA dtype */
    dtype = PyArray_DescrNewFromType(is_datetime ? NPY_DATETIME :
                                                   NPY_TIMEDELTA);
    if (dtype == NULL) {
        Py_DECREF(gcdmeta);
        return NULL;
    }

    /* Replace the metadata dictionary */
    Py_XDECREF(dtype->metadata);
    dtype->metadata = PyDict_New();
    if (dtype->metadata == NULL) {
        Py_DECREF(dtype);
        Py_DECREF(gcdmeta);
        return NULL;
    }

    /* Set the metadata object in the dictionary. */
    if (PyDict_SetItemString(dtype->metadata, NPY_METADATA_DTSTR,
                                                gcdmeta) < 0) {
        Py_DECREF(dtype);
        Py_DECREF(gcdmeta);
        return NULL;
    }
    Py_DECREF(gcdmeta);
    
    return dtype;


    PyErr_SetString(PyExc_RuntimeError,
            "Called datetime_type_promotion on non-datetype type");
    return NULL;
}

/*
 * Converts a substring given by 'str' and 'len' into
 * a date time unit enum value. The 'metastr' parameter
 * is used for error messages, and may be NULL.
 *
 * Generic units have no representation as a string in this form.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT NPY_DATETIMEUNIT
parse_datetime_unit_from_string(char *str, Py_ssize_t len, char *metastr)
{
    /* Use switch statements so the compiler can make it fast */
    if (len == 1) {
        switch (str[0]) {
            case 'Y':
                return NPY_FR_Y;
            case 'M':
                return NPY_FR_M;
            case 'W':
                return NPY_FR_W;
            case 'B':
                return NPY_FR_B;
            case 'D':
                return NPY_FR_D;
            case 'h':
                return NPY_FR_h;
            case 'm':
                return NPY_FR_m;
            case 's':
                return NPY_FR_s;
        }
    }
    /* All the two-letter units are variants of seconds */
    else if (len == 2 && str[1] == 's') {
        switch (str[0]) {
            case 'm':
                return NPY_FR_ms;
            case 'u':
                return NPY_FR_us;
            case 'n':
                return NPY_FR_ns;
            case 'p':
                return NPY_FR_ps;
            case 'f':
                return NPY_FR_fs;
            case 'a':
                return NPY_FR_as;
        }
    }

    /* If nothing matched, it's an error */
    if (metastr == NULL) {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime unit \"%s\" in metadata",
                str);
    }
    else {
        PyErr_Format(PyExc_TypeError,
                "Invalid datetime unit in metadata string \"%s\"",
                metastr);
    }
    return -1;
}


NPY_NO_EXPORT PyObject *
convert_datetime_metadata_to_tuple(PyArray_DatetimeMetaData *meta)
{
    PyObject *dt_tuple;

    dt_tuple = PyTuple_New(3);
    if (dt_tuple == NULL) {
        return NULL;
    }

    PyTuple_SET_ITEM(dt_tuple, 0,
            PyBytes_FromString(_datetime_strings[meta->base]));
    PyTuple_SET_ITEM(dt_tuple, 1,
            PyInt_FromLong(meta->num));
    PyTuple_SET_ITEM(dt_tuple, 2,
            PyInt_FromLong(meta->events));

    return dt_tuple;
}

/*
 * Converts a metadata tuple into a datetime metadata C struct.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_datetime_metadata_tuple_to_datetime_metadata(PyObject *tuple,
                                        PyArray_DatetimeMetaData *out_meta)
{
    char *basestr = NULL;
    Py_ssize_t len = 0, tuple_size;
    int den = 1;

    if (!PyTuple_Check(tuple)) {
        PyObject_Print(tuple, stderr, 0);
        PyErr_SetString(PyExc_TypeError,
                        "Require tuple for tuple to NumPy datetime "
                        "metadata conversion");
        return -1;
    }

    tuple_size = PyTuple_GET_SIZE(tuple);
    if (tuple_size < 3 || tuple_size > 4) {
        PyErr_SetString(PyExc_TypeError,
                        "Require tuple of size 3 or 4 for "
                        "tuple to NumPy datetime metadata conversion");
        return -1;
    }

    if (PyBytes_AsStringAndSize(PyTuple_GET_ITEM(tuple, 0),
                                        &basestr, &len) < 0) {
        return -1;
    }

    out_meta->base = parse_datetime_unit_from_string(basestr, len, NULL);
    if (out_meta->base == -1) {
        return -1;
    }

    /* Convert the values to longs */
    out_meta->num = PyInt_AsLong(PyTuple_GET_ITEM(tuple, 1));
    if (out_meta->num == -1 && PyErr_Occurred()) {
        return -1;
    }

    if (tuple_size == 3) {
        out_meta->events = PyInt_AsLong(PyTuple_GET_ITEM(tuple, 2));
        if (out_meta->events == -1 && PyErr_Occurred()) {
            return -1;
        }
    }
    else {
        den = PyInt_AsLong(PyTuple_GET_ITEM(tuple, 2));
        if (den == -1 && PyErr_Occurred()) {
            return -1;
        }
        out_meta->events = PyInt_AsLong(PyTuple_GET_ITEM(tuple, 3));
        if (out_meta->events == -1 && PyErr_Occurred()) {
            return -1;
        }
    }

    if (out_meta->num <= 0 || out_meta->events <= 0 || den <= 0) {
        PyErr_SetString(PyExc_TypeError,
                        "Invalid tuple values for "
                        "tuple to NumPy datetime metadata conversion");
        return -1;
    }

    if (den != 1) {
        if (convert_datetime_divisor_to_multiple(out_meta, den, NULL) < 0) {
            return -1;
        }
    }

    return 0;
}

/*
 * Converts a metadata tuple into a datetime metadata capsule.
 */
NPY_NO_EXPORT PyObject *
convert_datetime_metadata_tuple_to_metacobj(PyObject *tuple)
{
    PyArray_DatetimeMetaData *dt_data;

    dt_data = PyArray_malloc(sizeof(PyArray_DatetimeMetaData));

    if (convert_datetime_metadata_tuple_to_datetime_metadata(
                                                tuple, dt_data) < 0) {
        PyArray_free(dt_data);
        return NULL;
    }

    return NpyCapsule_FromVoidPtr((void *)dt_data, simple_capsule_dtor);
}

/*
 * Converts an input object into datetime metadata. The input
 * may be either a string or a tuple.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
convert_pyobject_to_datetime_metadata(PyObject *obj,
                                      PyArray_DatetimeMetaData *out_meta)
{
    PyObject *ascii = NULL;
    char *str = NULL;
    Py_ssize_t len = 0;

    if (PyTuple_Check(obj)) {
        return convert_datetime_metadata_tuple_to_datetime_metadata(obj,
                                                                out_meta);
    }

    /* Get an ASCII string */
    if (PyUnicode_Check(obj)) {
        /* Allow unicode format strings: convert to bytes */
        ascii = PyUnicode_AsASCIIString(obj);
        if (ascii == NULL) {
            return -1;
        }
    }
    else if (PyBytes_Check(obj)) {
        ascii = obj;
        Py_INCREF(ascii);
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "Invalid object for specifying NumPy datetime metadata");
        return -1;
    }

    if (PyBytes_AsStringAndSize(ascii, &str, &len) < 0) {
        return -1;
    }

    if (len > 0 && str[0] == '[') {
        return parse_datetime_metadata_from_metastr(str, len, out_meta);
    }
    else {
        if (parse_datetime_extended_unit_from_string(str, len,
                                                NULL, out_meta) < 0) {
            return -1;
        }

        /* extended_unit is only 'num' and 'base', we have to fill the rest */
        out_meta->events = 1;

        return 0;
    }

}

/*
 * 'ret' is a PyUString containing the datetime string, and this
 * function appends the metadata string to it.
 *
 * If 'skip_brackets' is true, skips the '[]' when events == 1.
 *
 * This function steals the reference 'ret'
 */
NPY_NO_EXPORT PyObject *
append_metastr_to_string(PyArray_DatetimeMetaData *meta,
                                    int skip_brackets,
                                    PyObject *ret)
{
    PyObject *res;
    int num, events;
    char *basestr;

    if (ret == NULL) {
        return NULL;
    }

    if (meta->base == NPY_FR_GENERIC) {
        /* Without brackets, give a string "generic" */
        if (skip_brackets) {
            PyUString_ConcatAndDel(&ret, PyUString_FromString("generic"));
            return ret;
        }
        /* But with brackets, append nothing */
        else {
            return ret;
        }
    }

    num = meta->num;
    events = meta->events;
    if (meta->base >= 0 && meta->base < NPY_DATETIME_NUMUNITS) {
        basestr = _datetime_strings[meta->base];
    }
    else {
        PyErr_SetString(PyExc_RuntimeError,
                "NumPy datetime metadata is corrupted");
        return NULL;
    }

    if (num == 1) {
        if (skip_brackets && events == 1) {
            res = PyUString_FromFormat("%s", basestr);
        }
        else {
            res = PyUString_FromFormat("[%s]", basestr);
        }
    }
    else {
        if (skip_brackets && events == 1) {
            res = PyUString_FromFormat("%d%s", num, basestr);
        }
        else {
            res = PyUString_FromFormat("[%d%s]", num, basestr);
        }
    }

    if (events != 1) {
        PyUString_ConcatAndDel(&res,
                    PyUString_FromFormat("//%d", events));
    }
    PyUString_ConcatAndDel(&ret, res);
    return ret;
}

/*
 * Adjusts a datetimestruct based on a seconds offset. Assumes
 * the current values are valid.
 */
NPY_NO_EXPORT void
add_seconds_to_datetimestruct(npy_datetimestruct *dts, int seconds)
{
    int minutes;

    dts->sec += seconds;
    if (dts->sec < 0) {
        minutes = dts->sec / 60;
        dts->sec = dts->sec % 60;
        if (dts->sec < 0) {
            --minutes;
            dts->sec += 60;
        }
        add_minutes_to_datetimestruct(dts, minutes);
    }
    else if (dts->sec >= 60) {
        minutes = dts->sec / 60;
        dts->sec = dts->sec % 60;
        add_minutes_to_datetimestruct(dts, minutes);
    }
}

/*
 * Adjusts a datetimestruct based on a minutes offset. Assumes
 * the current values are valid.
 */
NPY_NO_EXPORT void
add_minutes_to_datetimestruct(npy_datetimestruct *dts, int minutes)
{
    int isleap;

    /* MINUTES */
    dts->min += minutes;
    while (dts->min < 0) {
        dts->min += 60;
        dts->hour--;
    }
    while (dts->min >= 60) {
        dts->min -= 60;
        dts->hour++;
    }

    /* HOURS */
    while (dts->hour < 0) {
        dts->hour += 24;
        dts->day--;
    }
    while (dts->hour >= 24) {
        dts->hour -= 24;
        dts->day++;
    }

    /* DAYS */
    if (dts->day < 1) {
        dts->month--;
        if (dts->month < 1) {
            dts->year--;
            dts->month = 12;
        }
        isleap = is_leapyear(dts->year);
        dts->day += days_in_month[isleap][dts->month-1];
    }
    else if (dts->day > 28) {
        isleap = is_leapyear(dts->year);
        if (dts->day > days_in_month[isleap][dts->month-1]) {
            dts->day -= days_in_month[isleap][dts->month-1];
            dts->month++;
            if (dts->month > 12) {
                dts->year++;
                dts->month = 1;
            }
        }
    }
}

/*
 * Parses (almost) standard ISO 8601 date strings. The differences are:
 *
 * + After the date and time, may place a ' ' followed by an event number.
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
 *
 * 'out' gets filled with the parsed date-time.
 * 'out_local' gets set to 1 if the parsed time was in local time,
 *      to 0 otherwise. The values 'now' and 'today' don't get counted
 *      as local, and neither do UTC +/-#### timezone offsets, because
 *      they aren't using the computer's local timezone offset.
 * 'out_bestunit' gives a suggested unit based on the amount of
 *      resolution provided in the string, or -1 for NaT.
 * 'out_special' gets set to 1 if the parsed time was 'today',
 *      'now', or ''/'NaT'. For 'today', the unit recommended is
 *      'D', for 'now', the unit recommended is 's', and for 'NaT'
 *      the unit recommended is 'Y'.
 *
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
parse_iso_8601_date(char *str, int len,
                    NPY_DATETIMEUNIT unit,
                    npy_datetimestruct *out,
                    npy_bool *out_local,
                    NPY_DATETIMEUNIT *out_bestunit,
                    npy_bool *out_special)
{
    int year_leap = 0;
    int i, numdigits;
    char *substr, sublen;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(npy_datetimestruct));
    out->month = 1;
    out->day = 1;
    
    /* The empty string and case-variants of "NaT" parse to not-a-time */
    if (len <= 0 || (len == 3 &&
                        tolower(str[0]) == 'n' &&
                        tolower(str[1]) == 'a' &&
                        tolower(str[2]) == 't')) {
        out->year = NPY_DATETIME_NAT;

        /*
         * Indicate that this was a special value, and
         * recommend generic units.
         */
        if (out_local != NULL) {
            *out_local = 0;
        }
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
     * The string "today" resolves to midnight of today's local date in UTC.
     * This is perhaps a little weird, but done so that further truncation
     * to a 'datetime64[D]' type produces the date you expect, rather than
     * switching to an adjacent day depending on the current time and your
     * timezone.
     */
    if (len == 5 && tolower(str[0]) == 't' &&
                    tolower(str[1]) == 'o' &&
                    tolower(str[2]) == 'd' &&
                    tolower(str[3]) == 'a' &&
                    tolower(str[4]) == 'y') {
        time_t rawtime = 0;
        struct tm tm_;

        /* 'today' only works for units of days or larger */
        if (unit != -1 && unit > NPY_FR_D) {
            PyErr_SetString(PyExc_ValueError,
                        "Special value 'today' can only be converted "
                        "to a NumPy datetime with 'D' or larger units");
            return -1;
        }

        time(&rawtime);
#if defined(_WIN32)
        if (localtime_s(&tm_, &rawtime) != 0) {
            PyErr_SetString(PyExc_OSError, "Failed to use localtime_s to "
                                        "get local time");
            return -1;
        }
#else
        /* Other platforms may require something else */
        if (localtime_r(&rawtime, &tm_) == NULL) {
            PyErr_SetString(PyExc_OSError, "Failed to use localtime_r to "
                                        "get local time");
            return -1;
        }
#endif
        out->year = tm_.tm_year + 1900;
        out->month = tm_.tm_mon + 1;
        out->day = tm_.tm_mday;

        /*
         * Indicate that this was a special value, and
         * is a date (unit 'D').
         */
        if (out_local != NULL) {
            *out_local = 0;
        }
        if (out_bestunit != NULL) {
            *out_bestunit = NPY_FR_D;
        }
        if (out_special != NULL) {
            *out_special = 1;
        }

        return 0;
    }

    /* The string "now" resolves to the current UTC time */
    if (len == 3 && tolower(str[0]) == 'n' &&
                    tolower(str[1]) == 'o' &&
                    tolower(str[2]) == 'w') {
        time_t rawtime = 0;
        PyArray_DatetimeMetaData meta;

        /* 'now' only works for units of hours or smaller */
        if (unit != -1 && unit < NPY_FR_h) {
            PyErr_SetString(PyExc_ValueError,
                        "Special value 'now' can only be converted "
                        "to a NumPy datetime with 'h' or smaller units");
            return -1;
        }

        time(&rawtime);

        /* Set up a dummy metadata for the conversion */
        meta.base = NPY_FR_s;
        meta.num = 1;
        meta.events = 1;

        /*
         * Indicate that this was a special value, and
         * use 's' because the time() function has resolution
         * seconds.
         */
        if (out_local != NULL) {
            *out_local = 0;
        }
        if (out_bestunit != NULL) {
            *out_bestunit = NPY_FR_s;
        }
        if (out_special != NULL) {
            *out_special = 1;
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
    if (*substr == '-') {
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
        if (out_local != NULL) {
            *out_local = 0;
        }
        if (out_bestunit != NULL) {
            *out_bestunit = NPY_FR_Y;
        }
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
        if (out_local != NULL) {
            *out_local = 0;
        }
        if (out_bestunit != NULL) {
            *out_bestunit = NPY_FR_M;
        }
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
                        out->day > days_in_month[year_leap][out->month-1]) {
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
        if (out_local != NULL) {
            *out_local = 0;
        }
        if (out_bestunit != NULL) {
            *out_bestunit = NPY_FR_D;
        }
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

        if (out->hour < 0 || out->hour >= 24) {
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
        if (out_bestunit != NULL) {
            *out_bestunit = NPY_FR_h;
        }
        goto parse_timezone;
    }

    /* Can't have a trailing ':' */
    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE MINUTES (2 digits) */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        out->min = 10 * (substr[0] - '0') + (substr[1] - '0');

        if (out->hour < 0 || out->min >= 60) {
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
        if (out_bestunit != NULL) {
            *out_bestunit = NPY_FR_m;
        }
        goto parse_timezone;
    }

    /* Can't have a trailing ':' */
    if (sublen == 0) {
        goto parse_error;
    }

    /* PARSE THE SECONDS (2 digits) */
    if (sublen >= 2 && isdigit(substr[0]) && isdigit(substr[1])) {
        out->sec = 10 * (substr[0] - '0') + (substr[1] - '0');

        if (out->sec < 0 || out->sec >= 60) {
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
        if (out_bestunit != NULL) {
            *out_bestunit = NPY_FR_s;
        }
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
        if (out_bestunit != NULL) {
            if (numdigits > 3) {
                *out_bestunit = NPY_FR_us;
            }
            else {
                *out_bestunit = NPY_FR_ms;
            }
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
        if (out_bestunit != NULL) {
            if (numdigits > 3) {
                *out_bestunit = NPY_FR_ps;
            }
            else {
                *out_bestunit = NPY_FR_ns;
            }
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

    if (out_bestunit != NULL) {
        if (numdigits > 3) {
            *out_bestunit = NPY_FR_as;
        }
        else {
            *out_bestunit = NPY_FR_fs;
        }
    }

parse_timezone:
    if (sublen == 0) {
        /*
         * ISO 8601 states to treat date-times without a timezone offset
         * or 'Z' for UTC as local time. The C standard libary functions
         * mktime and gmtime allow us to do this conversion.
         *
         * Only do this timezone adjustment for recent and future years.
         */
        if (out->year > 1900 && out->year < 10000) {
            time_t rawtime = 0;
            struct tm tm_;

            tm_.tm_sec = out->sec;
            tm_.tm_min = out->min;
            tm_.tm_hour = out->hour;
            tm_.tm_mday = out->day;
            tm_.tm_mon = out->month - 1;
            tm_.tm_year = out->year - 1900;
            tm_.tm_isdst = -1;

            /* mktime converts a local 'struct tm' into a time_t */
            rawtime = mktime(&tm_);
            if (rawtime == -1) {
                PyErr_SetString(PyExc_OSError, "Failed to use mktime to "
                                            "convert local time to UTC");
                goto error;
            }

            /* gmtime converts a 'time_t' into a UTC 'struct tm' */
#if defined(_WIN32)
            if (gmtime_s(&tm_, &rawtime) != 0) {
                PyErr_SetString(PyExc_OSError, "Failed to use gmtime_s to "
                                            "get a UTC time");
                goto error;
            }
#else
            /* Other platforms may require something else */
            if (gmtime_r(&rawtime, &tm_) == NULL) {
                PyErr_SetString(PyExc_OSError, "Failed to use gmtime_r to "
                                            "get a UTC time");
                goto error;
            }
#endif
            out->sec = tm_.tm_sec;
            out->min = tm_.tm_min;
            out->hour = tm_.tm_hour;
            out->day = tm_.tm_mday;
            out->month = tm_.tm_mon + 1;
            out->year = tm_.tm_year + 1900;
        }

        /* Since neither "Z" nor a time-zone was specified, it's local */
        if (out_local != NULL) {
            *out_local = 1;
        }

        goto finish;
    }

    /* UTC specifier */
    if (*substr == 'Z') {
        /* "Z" means not local */
        if (out_local != NULL) {
            *out_local = 0;
        }

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

        /*
         * Since "local" means local with respect to the current
         * machine, we say this is non-local.
         */
        if (out_local != NULL) {
            *out_local = 0;
        }

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
    return 0;

parse_error:
    PyErr_Format(PyExc_ValueError,
            "Error parsing datetime string \"%s\" at position %d",
            str, (int)(substr-str));
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

    /* If no unit is provided, return the maximum length */
    if (base == -1) {
        return NPY_DATETIME_MAX_ISO8601_STRLEN;
    }

    switch (base) {
        /* Generic units can only be used to represent NaT */
        case NPY_FR_GENERIC:
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
        case NPY_FR_B:
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
 * Converts an npy_datetimestruct to an (almost) ISO 8601
 * NULL-terminated string.
 *
 * If 'local' is non-zero, it produces a string in local time with
 * a +-#### timezone offset, otherwise it uses timezone Z (UTC).
 *
 * 'base' restricts the output to that unit. Set 'base' to
 * -1 to auto-detect a base after which all the values are zero.
 *
 *  'tzoffset' is used if 'local' is enabled, and 'tzoffset' is
 *  set to a value other than -1. This is a manual override for
 *  the local time zone to use, as an offset in minutes.
 *
 *  Returns 0 on success, -1 on failure (for example if the output
 *  string was too short).
 */
NPY_NO_EXPORT int
make_iso_8601_date(npy_datetimestruct *dts, char *outstr, int outlen,
                    int local, NPY_DATETIMEUNIT base, int tzoffset)
{
    npy_datetimestruct dts_local;
    int timezone_offset = 0;

    char *substr = outstr, sublen = outlen;
    int tmplen;

    /* Handle NaT, and treat a datetime with generic units as NaT */
    if (dts->year == NPY_DATETIME_NAT || base == NPY_FR_GENERIC) {
        if (outlen < 4) {
            goto string_too_short;
        }
        outstr[0] = 'N';
        outstr[0] = 'a';
        outstr[0] = 'T';
        outstr[0] = '\0';

        return 0;
    }

    /* Only do local time within a reasonable year range */
    if ((dts->year <= 1900 || dts->year >= 10000) && tzoffset == -1) {
        local = 0;
    }

    /* Automatically detect a good unit */
    if (base == -1) {
        if (dts->as % 1000 != 0) {
            base = NPY_FR_as;
        }
        else if (dts->as != 0) {
            base = NPY_FR_fs;
        }
        else if (dts->ps % 1000 != 0) {
            base = NPY_FR_ps;
        }
        else if (dts->ps != 0) {
            base = NPY_FR_ns;
        }
        else if (dts->us % 1000 != 0) {
            base = NPY_FR_us;
        }
        else if (dts->us != 0) {
            base = NPY_FR_ms;
        }
        else if (dts->sec != 0) {
            base = NPY_FR_s;
        }
        /*
         * hours and minutes don't get split up by default, and printing
         * in local time forces minutes
         */
        else if (local || dts->min != 0 || dts->hour != 0) {
            base = NPY_FR_m;
        }
        /* dates don't get split up by default */
        else {
            base = NPY_FR_D;
        }
    }
    /*
     * Print business days and weeks with the same precision as days.
     *
     * TODO: Could print weeks with YYYY-Www format if the week
     *       epoch is a Monday.
     */
    else if (base == NPY_FR_B || base == NPY_FR_W) {
        base = NPY_FR_D;
    }

    /* Printed dates have no time zone */
    if (base < NPY_FR_h) {
        local = 0;
    }

    /* Use the C API to convert from UTC to local time */
    if (local && tzoffset == -1) {
        time_t rawtime = 0, localrawtime;
        struct tm tm_;

        /*
         * Convert everything in 'dts' to a time_t, to minutes precision.
         * This is POSIX time, which skips leap-seconds, but because
         * we drop the seconds value from the npy_datetimestruct, everything
         * is ok for this operation.
         */
        rawtime = (time_t)get_datetimestruct_days(dts) * 24 * 60 * 60;
        rawtime += dts->hour * 60 * 60;
        rawtime += dts->min * 60;

        /* localtime converts a 'time_t' into a local 'struct tm' */
#if defined(_WIN32)
        if (localtime_s(&tm_, &rawtime) != 0) {
            PyErr_SetString(PyExc_OSError, "Failed to use localtime_s to "
                                        "get a local time");
            return -1;
        }
#else
        /* Other platforms may require something else */
        if (localtime_r(&rawtime, &tm_) == NULL) {
            PyErr_SetString(PyExc_OSError, "Failed to use localtime_r to "
                                        "get a local time");
            return -1;
        }
#endif
        /* Make a copy of the npy_datetimestruct we can modify */
        dts_local = *dts;

        /* Copy back all the values except seconds */
        dts_local.min = tm_.tm_min;
        dts_local.hour = tm_.tm_hour;
        dts_local.day = tm_.tm_mday;
        dts_local.month = tm_.tm_mon + 1;
        dts_local.year = tm_.tm_year + 1900;

        /* Extract the timezone offset that was applied */
        rawtime /= 60;
        localrawtime = (time_t)get_datetimestruct_days(&dts_local) * 24 * 60;
        localrawtime += dts_local.hour * 60;
        localrawtime += dts_local.min;

        timezone_offset = localrawtime - rawtime;

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

    /* YEAR */
#ifdef _WIN32
    tmplen = _snprintf(substr, sublen, "%04" NPY_INT64_FMT, dts->year);
#else
    tmplen = snprintf(substr, sublen, "%04" NPY_INT64_FMT, dts->year);
#endif
    /* If it ran out of space or there isn't space for the NULL terminator */
    if (tmplen < 0 || tmplen >= sublen) {
        goto string_too_short;
    }
    substr += tmplen;
    sublen -= tmplen;

    /* Stop if the unit is years */
    if (base == NPY_FR_Y) {
        *substr = '\0';
        return 0;
    }

    /* MONTH */
    substr[0] = '-';
    if (sublen <= 1 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->month / 10) + '0');
    if (sublen <= 2 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->month % 10) + '0');
    if (sublen <= 3 ) {
        goto string_too_short;
    }
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is months */
    if (base == NPY_FR_M) {
        *substr = '\0';
        return 0;
    }

    /* DAY */
    substr[0] = '-';
    if (sublen <= 1 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->day / 10) + '0');
    if (sublen <= 2 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->day % 10) + '0');
    if (sublen <= 3 ) {
        goto string_too_short;
    }
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is days */
    if (base == NPY_FR_D) {
        *substr = '\0';
        return 0;
    }

    /* HOUR */
    substr[0] = 'T';
    if (sublen <= 1 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->hour / 10) + '0');
    if (sublen <= 2 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->hour % 10) + '0');
    if (sublen <= 3 ) {
        goto string_too_short;
    }
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is hours */
    if (base == NPY_FR_h) {
        goto add_time_zone;
    }

    /* MINUTE */
    substr[0] = ':';
    if (sublen <= 1 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->min / 10) + '0');
    if (sublen <= 2 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->min % 10) + '0');
    if (sublen <= 3 ) {
        goto string_too_short;
    }
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is minutes */
    if (base == NPY_FR_m) {
        goto add_time_zone;
    }

    /* SECOND */
    substr[0] = ':';
    if (sublen <= 1 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->sec / 10) + '0');
    if (sublen <= 2 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->sec % 10) + '0');
    if (sublen <= 3 ) {
        goto string_too_short;
    }
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is seconds */
    if (base == NPY_FR_s) {
        goto add_time_zone;
    }

    /* MILLISECOND */
    substr[0] = '.';
    if (sublen <= 1 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->us / 100000) % 10 + '0');
    if (sublen <= 2 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->us / 10000) % 10 + '0');
    if (sublen <= 3 ) {
        goto string_too_short;
    }
    substr[3] = (char)((dts->us / 1000) % 10 + '0');
    if (sublen <= 4 ) {
        goto string_too_short;
    }
    substr += 4;
    sublen -= 4;

    /* Stop if the unit is milliseconds */
    if (base == NPY_FR_ms) {
        goto add_time_zone;
    }

    /* MICROSECOND */
    substr[0] = (char)((dts->us / 100) % 10 + '0');
    if (sublen <= 1 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->us / 10) % 10 + '0');
    if (sublen <= 2 ) {
        goto string_too_short;
    }
    substr[2] = (char)(dts->us % 10 + '0');
    if (sublen <= 3 ) {
        goto string_too_short;
    }
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is microseconds */
    if (base == NPY_FR_us) {
        goto add_time_zone;
    }

    /* NANOSECOND */
    substr[0] = (char)((dts->ps / 100000) % 10 + '0');
    if (sublen <= 1 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->ps / 10000) % 10 + '0');
    if (sublen <= 2 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->ps / 1000) % 10 + '0');
    if (sublen <= 3 ) {
        goto string_too_short;
    }
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is nanoseconds */
    if (base == NPY_FR_ns) {
        goto add_time_zone;
    }

    /* PICOSECOND */
    substr[0] = (char)((dts->ps / 100) % 10 + '0');
    if (sublen <= 1 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->ps / 10) % 10 + '0');
    if (sublen <= 2 ) {
        goto string_too_short;
    }
    substr[2] = (char)(dts->ps % 10 + '0');
    if (sublen <= 3 ) {
        goto string_too_short;
    }
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is picoseconds */
    if (base == NPY_FR_ps) {
        goto add_time_zone;
    }

    /* FEMTOSECOND */
    substr[0] = (char)((dts->as / 100000) % 10 + '0');
    if (sublen <= 1 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->as / 10000) % 10 + '0');
    if (sublen <= 2 ) {
        goto string_too_short;
    }
    substr[2] = (char)((dts->as / 1000) % 10 + '0');
    if (sublen <= 3 ) {
        goto string_too_short;
    }
    substr += 3;
    sublen -= 3;

    /* Stop if the unit is femtoseconds */
    if (base == NPY_FR_fs) {
        goto add_time_zone;
    }

    /* ATTOSECOND */
    substr[0] = (char)((dts->as / 100) % 10 + '0');
    if (sublen <= 1 ) {
        goto string_too_short;
    }
    substr[1] = (char)((dts->as / 10) % 10 + '0');
    if (sublen <= 2 ) {
        goto string_too_short;
    }
    substr[2] = (char)(dts->as % 10 + '0');
    if (sublen <= 3 ) {
        goto string_too_short;
    }
    substr += 3;
    sublen -= 3;

add_time_zone:
    if (local) {
        /* Add the +/- sign */
        if (timezone_offset < 0) {
            substr[0] = '-';
            timezone_offset = -timezone_offset;
        }
        else {
            substr[0] = '+';
        }
        if (sublen <= 1) {
            goto string_too_short;
        }
        substr += 1;
        sublen -= 1;

        /* Add the timezone offset */
        substr[0] = (char)((timezone_offset / (10*60)) % 10 + '0');
        if (sublen <= 1 ) {
            goto string_too_short;
        }
        substr[1] = (char)((timezone_offset / 60) % 10 + '0');
        if (sublen <= 2 ) {
            goto string_too_short;
        }
        substr[2] = (char)(((timezone_offset % 60) / 10) % 10 + '0');
        if (sublen <= 3 ) {
            goto string_too_short;
        }
        substr[3] = (char)((timezone_offset % 60) % 10 + '0');
        if (sublen <= 4 ) {
            goto string_too_short;
        }
        substr += 4;
        sublen -= 4;
    }
    /* UTC "Zulu" time */
    else {
        substr[0] = 'Z';
        if (sublen <= 1) {
            goto string_too_short;
        }
        substr += 1;
        sublen -= 1;
    }

    /* Add a NULL terminator, and return */
    substr[0] = '\0';

    return 0;

string_too_short:
    /* Put a NULL terminator on anyway */
    if (outlen > 0) {
        outstr[outlen-1] = '\0';
    }

    PyErr_Format(PyExc_RuntimeError,
                "The string provided for NumPy ISO datetime formatting "
                "was too short, with length %d",
                outlen);
    return -1;
}

/*
 * Tests for and converts a Python datetime.datetime or datetime.date
 * object into a NumPy npy_datetimestruct.
 *
 * While the C API has PyDate_* and PyDateTime_* functions, the following
 * implementation just asks for attributes, and thus supports
 * datetime duck typing. The tzinfo time zone conversion would require
 * this style of access anyway.
 *
 * 'out_bestunit' gives a suggested unit based on whether the object
 *      was a datetime.date or datetime.datetime object.
 *
 * Returns -1 on error, 0 on success, and 1 (with no error set)
 * if obj doesn't have the neeeded date or datetime attributes.
 */
NPY_NO_EXPORT int
convert_pydatetime_to_datetimestruct(PyObject *obj, npy_datetimestruct *out,
                                     NPY_DATETIMEUNIT *out_bestunit)
{
    PyObject *tmp;
    int isleap;

    /* Initialize the output to all zeros */
    memset(out, 0, sizeof(npy_datetimestruct));
    out->month = 1;
    out->day = 1;

    /* Need at least year/month/day attributes */
    if (!PyObject_HasAttrString(obj, "year") ||
            !PyObject_HasAttrString(obj, "month") ||
            !PyObject_HasAttrString(obj, "day")) {
        return 1;
    }

    /* Get the year */
    tmp = PyObject_GetAttrString(obj, "year");
    if (tmp == NULL) {
        return -1;
    }
    out->year = PyInt_AsLong(tmp);
    if (out->year == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the month */
    tmp = PyObject_GetAttrString(obj, "month");
    if (tmp == NULL) {
        return -1;
    }
    out->month = PyInt_AsLong(tmp);
    if (out->month == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the day */
    tmp = PyObject_GetAttrString(obj, "day");
    if (tmp == NULL) {
        return -1;
    }
    out->day = PyInt_AsLong(tmp);
    if (out->day == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Validate that the month and day are valid for the year */
    if (out->month < 1 || out->month > 12) {
        goto invalid_date;
    }
    isleap = is_leapyear(out->year);
    if (out->day < 1 || out->day > days_in_month[isleap][out->month-1]) {
        goto invalid_date;
    }

    /* Check for time attributes (if not there, return success as a date) */
    if (!PyObject_HasAttrString(obj, "hour") ||
            !PyObject_HasAttrString(obj, "minute") ||
            !PyObject_HasAttrString(obj, "second") ||
            !PyObject_HasAttrString(obj, "microsecond")) {
        /* The best unit for date is 'D' */
        if (out_bestunit != NULL) {
            *out_bestunit = NPY_FR_D;
        }
        return 0;
    }

    /* Get the hour */
    tmp = PyObject_GetAttrString(obj, "hour");
    if (tmp == NULL) {
        return -1;
    }
    out->hour = PyInt_AsLong(tmp);
    if (out->hour == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the minute */
    tmp = PyObject_GetAttrString(obj, "minute");
    if (tmp == NULL) {
        return -1;
    }
    out->min = PyInt_AsLong(tmp);
    if (out->min == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the second */
    tmp = PyObject_GetAttrString(obj, "second");
    if (tmp == NULL) {
        return -1;
    }
    out->sec = PyInt_AsLong(tmp);
    if (out->sec == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    /* Get the microsecond */
    tmp = PyObject_GetAttrString(obj, "microsecond");
    if (tmp == NULL) {
        return -1;
    }
    out->us = PyInt_AsLong(tmp);
    if (out->us == -1 && PyErr_Occurred()) {
        Py_DECREF(tmp);
        return -1;
    }
    Py_DECREF(tmp);

    if (out->hour < 0 || out->hour >= 24 ||
            out->min < 0 || out->min >= 60 ||
            out->sec < 0 || out->sec >= 60 ||
            out->us < 0 || out->us >= 1000000) {
        goto invalid_time;
    }

    /* Apply the time zone offset if it exists */
    if (PyObject_HasAttrString(obj, "tzinfo")) {
        tmp = PyObject_GetAttrString(obj, "tzinfo");
        if (tmp == NULL) {
            return -1;
        }
        if (tmp == Py_None) {
            Py_DECREF(tmp);
        }
        else {
            PyObject *offset;
            int seconds_offset, minutes_offset;

            /* The utcoffset function should return a timedelta */
            offset = PyObject_CallMethod(tmp, "utcoffset", "O", obj);
            if (offset == NULL) {
                Py_DECREF(tmp);
                return -1;
            }
            Py_DECREF(tmp);

            /*
             * The timedelta should have an attribute "seconds"
             * which contains the value we want.
             */
            tmp = PyObject_GetAttrString(obj, "seconds");
            if (tmp == NULL) {
                return -1;
            }
            seconds_offset = PyInt_AsLong(tmp);
            if (seconds_offset == -1 && PyErr_Occurred()) {
                Py_DECREF(tmp);
                return -1;
            }
            Py_DECREF(tmp);

            /* Convert to a minutes offset and apply it */
            minutes_offset = seconds_offset / 60;

            add_minutes_to_datetimestruct(out, -minutes_offset);
        }
    }

    /* The resolution of Python's datetime is 'us' */
    if (out_bestunit != NULL) {
        *out_bestunit = NPY_FR_us;
    }

    return 0;

invalid_date:
    PyErr_Format(PyExc_ValueError,
            "Invalid date (%d,%d,%d) when converting to NumPy datetime",
            (int)out->year, (int)out->month, (int)out->day);
    return -1;

invalid_time:
    PyErr_Format(PyExc_ValueError,
            "Invalid time (%d,%d,%d,%d) when converting "
            "to NumPy datetime",
            (int)out->hour, (int)out->min, (int)out->sec, (int)out->us);
    return -1;
}

/*
 * Converts a PyObject * into a datetime, in any of the forms supported.
 *
 * If the units metadata isn't known ahead of time, set meta->base
 * to -1, and this function will populate meta with either default
 * values or values from the input object.
 *
 * Returns -1 on error, 0 on success.
 */
NPY_NO_EXPORT int
convert_pyobject_to_datetime(PyArray_DatetimeMetaData *meta, PyObject *obj,
                                npy_datetime *out)
{
    if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        PyObject *bytes = NULL;
        char *str = NULL;
        Py_ssize_t len = 0;
        npy_datetimestruct dts;
        NPY_DATETIMEUNIT bestunit = -1;

        /* Convert to an ASCII string for the date parser */
        if (PyUnicode_Check(obj)) {
            bytes = PyUnicode_AsASCIIString(obj);
            if (bytes == NULL) {
                return -1;
            }
        }
        else {
            bytes = obj;
            Py_INCREF(bytes);
        }
        if (PyBytes_AsStringAndSize(bytes, &str, &len) == -1) {
            Py_DECREF(bytes);
            return -1;
        }

        /* Parse the ISO date */
        if (parse_iso_8601_date(str, len, meta->base, &dts,
                                NULL, &bestunit, NULL) < 0) {
            Py_DECREF(bytes);
            return -1;
        }
        Py_DECREF(bytes);

        /* Use the detected unit if none was specified */
        if (meta->base == -1) {
            meta->base = bestunit;
            meta->num = 1;
            meta->events = 1;
        }

        if (convert_datetimestruct_to_datetime(meta, &dts, out) < 0) {
            return -1;
        }

        return 0;
    }
    /* Do no conversion on raw integers */
    else if (PyInt_Check(obj) || PyLong_Check(obj)) {
        /* Don't allow conversion from an integer without specifying a unit */
        if (meta->base == -1 || meta->base == NPY_FR_GENERIC) {
            PyErr_SetString(PyExc_ValueError, "Converting an integer to a "
                            "NumPy datetime requires a specified unit");
            return -1;
        }
        *out = PyLong_AsLongLong(obj);
        return 0;
    }
    /* Could be a tuple with event number in the second entry */
    else if (PyTuple_Check(obj) && PyTuple_Size(obj) == 2) {
        int event, event_old;
        if (convert_pyobject_to_datetime(meta, PyTuple_GET_ITEM(obj, 0),
                                                                out) < 0) {
            return -1;
        }
        event = (int)PyInt_AsLong(PyTuple_GET_ITEM(obj, 1));
        if (event == -1 && PyErr_Occurred()) {
            return -1;
        }
        if (event < 0 || event >= meta->events) {
            PyErr_SetString(PyExc_ValueError, "event value for NumPy "
                            "datetime is out of range");
            return -1;
        }
        /* Replace the event with the one from the tuple */
        event_old = *out % meta->events;
        if (event_old < 0) {
            event_old += meta->events;
        }
        *out = *out - event_old + event;

        return 0;
    }
    /* Datetime scalar */
    else if (PyArray_IsScalar(obj, Datetime)) {
        PyDatetimeScalarObject *dts = (PyDatetimeScalarObject *)obj;

        /* Copy the scalar directly if units weren't specified */
        if (meta->base == -1) {
            *meta = dts->obmeta;
            *out = dts->obval;

            return 0;
        }
        /* Otherwise do a casting transformation */
        else {
            return cast_datetime_to_datetime(&dts->obmeta, meta,
                                             dts->obval, out);
        }
    }
    /* Datetime zero-dimensional array */
    else if (PyArray_Check(obj) &&
                    PyArray_NDIM(obj) == 0 &&
                    PyArray_DESCR(obj)->type_num == NPY_DATETIME) {
        PyArray_DatetimeMetaData *obj_meta;
        npy_datetime dt = 0;

        obj_meta = get_datetime_metadata_from_dtype(PyArray_DESCR(obj));
        if (obj_meta == NULL) {
            return -1;
        }
        PyArray_DESCR(obj)->f->copyswap(&dt,
                                        PyArray_DATA(obj),
                                        !PyArray_ISNOTSWAPPED(obj),
                                        obj);

        /* Copy the value directly if units weren't specified */
        if (meta->base == -1) {
            *meta = *obj_meta;
            *out = dt;

            return 0;
        }
        /* Otherwise do a casting transformation */
        else {
            return cast_datetime_to_datetime(obj_meta, meta, dt, out);
        }
    }
    /* Convert from a Python date or datetime object */
    else {
        int code;
        npy_datetimestruct dts;
        NPY_DATETIMEUNIT bestunit = -1;

        code = convert_pydatetime_to_datetimestruct(obj, &dts, &bestunit);
        if (code == -1) {
            return -1;
        }
        else if (code == 0) {
            /* Use the detected unit if none was specified */
            if (meta->base == -1) {
                meta->base = bestunit;
                meta->num = 1;
                meta->events = 1;
            }

            if (convert_datetimestruct_to_datetime(meta, &dts, out) < 0) {
                return -1;
            }

            return 0;
        }
    }

    PyErr_SetString(PyExc_ValueError,
            "Could not convert object to NumPy datetime");
    return -1;
}

/*
 * Converts a PyObject * into a timedelta, in any of the forms supported
 *
 * If the units metadata isn't known ahead of time, set meta->base
 * to -1, and this function will populate meta with either default
 * values or values from the input object.
 *
 * Returns -1 on error, 0 on success.
 */
NPY_NO_EXPORT int
convert_pyobject_to_timedelta(PyArray_DatetimeMetaData *meta, PyObject *obj,
                                npy_timedelta *out)
{
    /* Do no conversion on raw integers */
    if (PyInt_Check(obj) || PyLong_Check(obj)) {
        /* Use the default unit if none was specified */
        if (meta->base == -1) {
            meta->base = NPY_DATETIME_DEFAULTUNIT;
            meta->num = 1;
            meta->events = 1;
        }

        *out = PyLong_AsLongLong(obj);
        return 0;
    }
    /* Timedelta scalar */
    else if (PyArray_IsScalar(obj, Timedelta)) {
        PyTimedeltaScalarObject *dts = (PyTimedeltaScalarObject *)obj;

        /* Copy the scalar directly if units weren't specified */
        if (meta->base == -1) {
            *meta = dts->obmeta;
            *out = dts->obval;

            return 0;
        }
        /* Otherwise do a casting transformation */
        else {
            return cast_timedelta_to_timedelta(&dts->obmeta, meta,
                                                dts->obval, out);
        }
    }
    /* Timedelta zero-dimensional array */
    else if (PyArray_Check(obj) &&
                    PyArray_NDIM(obj) == 0 &&
                    PyArray_DESCR(obj)->type_num == NPY_TIMEDELTA) {
        PyArray_DatetimeMetaData *obj_meta;
        npy_timedelta dt = 0;

        obj_meta = get_datetime_metadata_from_dtype(PyArray_DESCR(obj));
        if (obj_meta == NULL) {
            return -1;
        }
        PyArray_DESCR(obj)->f->copyswap(&dt,
                                        PyArray_DATA(obj),
                                        !PyArray_ISNOTSWAPPED(obj),
                                        obj);

        /* Copy the value directly if units weren't specified */
        if (meta->base == -1) {
            *meta = *obj_meta;
            *out = dt;

            return 0;
        }
        /* Otherwise do a casting transformation */
        else {
            return cast_timedelta_to_timedelta(obj_meta, meta, dt, out);
        }
    }
    /* Convert from a Python timedelta object */
    else if (PyObject_HasAttrString(obj, "days") &&
                PyObject_HasAttrString(obj, "seconds") &&
                PyObject_HasAttrString(obj, "microseconds")) {
        PyObject *tmp;
        PyArray_DatetimeMetaData us_meta;
        npy_timedelta td;
        npy_int64 days;
        int seconds = 0, useconds = 0;

        /* Get the days */
        tmp = PyObject_GetAttrString(obj, "days");
        if (tmp == NULL) {
            return -1;
        }
        days = PyLong_AsLongLong(tmp);
        if (days == -1 && PyErr_Occurred()) {
            Py_DECREF(tmp);
            return -1;
        }
        Py_DECREF(tmp);

        /* Get the seconds */
        tmp = PyObject_GetAttrString(obj, "seconds");
        if (tmp == NULL) {
            return -1;
        }
        seconds = PyInt_AsLong(tmp);
        if (seconds == -1 && PyErr_Occurred()) {
            Py_DECREF(tmp);
            return -1;
        }
        Py_DECREF(tmp);

        /* Get the microseconds */
        tmp = PyObject_GetAttrString(obj, "microseconds");
        if (tmp == NULL) {
            return -1;
        }
        useconds = PyInt_AsLong(tmp);
        if (useconds == -1 && PyErr_Occurred()) {
            Py_DECREF(tmp);
            return -1;
        }
        Py_DECREF(tmp);

        td = days*(24*60*60*1000000LL) + seconds*1000000LL + useconds;

        /* Use microseconds if none was specified */
        if (meta->base == -1) {
            meta->base = NPY_FR_us;
            meta->num = 1;
            meta->events = 1;

            *out = td;

            return 0;
        }
        else {
            /*
             * Convert to a microseconds timedelta, then cast to the
             * desired units.
             */
            us_meta.base = NPY_FR_us;
            us_meta.num = 1;
            us_meta.events = 1;
            
            return cast_timedelta_to_timedelta(&us_meta, meta, td, out);
        }
    }

    PyErr_SetString(PyExc_ValueError,
            "Could not convert object to NumPy timedelta");
    return -1;
}

/*
 * Converts a datetime into a PyObject *.
 *
 * Not-a-time is returned as the string "NaT".
 * For days or coarser, returns a datetime.date.
 * For microseconds or coarser, returns a datetime.datetime.
 * For units finer than microseconds, returns an integer.
 */
NPY_NO_EXPORT PyObject *
convert_datetime_to_pyobject(npy_datetime dt, PyArray_DatetimeMetaData *meta)
{
    PyObject *ret = NULL, *tup = NULL;
    npy_datetimestruct dts;

    /* Handle not-a-time, and generic units as NaT as well */
    if (dt == NPY_DATETIME_NAT || meta->base == NPY_FR_GENERIC) {
        return PyUString_FromString("NaT");
    }

    /* If the type's precision is greater than microseconds, return an int */
    if (meta->base > NPY_FR_us) {
        /* Skip use of a tuple for the events, just return the raw int */
        return PyLong_FromLongLong(dt);
    }

    /* Convert to a datetimestruct */
    if (convert_datetime_to_datetimestruct(meta, dt, &dts) < 0) {
        return NULL;
    }

    /*
     * If the year is outside the range of years supported by Python's
     * datetime, or the datetime64 falls on a leap second,
     * return a raw int.
     */
    if (dts.year < 1 || dts.year > 9999 || dts.sec == 60) {
        /* Also skip use of a tuple for the events */
        return PyLong_FromLongLong(dt);
    }

    /* If the type's precision is greater than days, return a datetime */
    if (meta->base > NPY_FR_D) {
        ret = PyDateTime_FromDateAndTime(dts.year, dts.month, dts.day,
                                dts.hour, dts.min, dts.sec, dts.us);
    }
    /* Otherwise return a date */
    else {
        ret = PyDate_FromDate(dts.year, dts.month, dts.day);
    }

    /* If there is one event, just return the datetime */
    if (meta->events == 1) {
        return ret;
    }
    /* Otherwise return a tuple with the event in the second position */
    else {
        tup = PyTuple_New(2);
        if (tup == NULL) {
            Py_DECREF(ret);
            return NULL;
        }
        PyTuple_SET_ITEM(tup, 0, ret);

        ret = PyInt_FromLong(dts.event);
        if (ret == NULL) {
            Py_DECREF(tup);
            return NULL;
        }
        PyTuple_SET_ITEM(tup, 1, ret);

        return tup;
    }
}

/*
 * Converts a timedelta into a PyObject *.
 *
 * Not-a-time is returned as the string "NaT".
 * For microseconds or coarser, returns a datetime.timedelta.
 * For units finer than microseconds, returns an integer.
 */
NPY_NO_EXPORT PyObject *
convert_timedelta_to_pyobject(npy_timedelta td, PyArray_DatetimeMetaData *meta)
{
    PyObject *ret = NULL, *tup = NULL;
    npy_timedelta value;
    int event = 0;
    int days = 0, seconds = 0, useconds = 0;

    /* Handle not-a-time */
    if (td == NPY_DATETIME_NAT) {
        return PyUString_FromString("NaT");
    }

    /*
     * If the type's precision is greater than microseconds, is
     * Y/M/B (nonlinear units), or is generic units, return an int
     */
    if (meta->base > NPY_FR_us ||
                    meta->base == NPY_FR_Y ||
                    meta->base == NPY_FR_M ||
                    meta->base == NPY_FR_B ||
                    meta->base == NPY_FR_GENERIC) {
        /* Skip use of a tuple for the events, just return the raw int */
        return PyLong_FromLongLong(td);
    }

    value = td;

    /* If there are events, extract the event */
    if (meta->events > 1) {
        event = (int)(value % meta->events);
        value = value / meta->events;
        if (event < 0) {
            --value;
            event += meta->events;
        }
    }

    /* Apply the unit multiplier (TODO: overflow treatment...) */
    value *= meta->num;

    /* Convert to days/seconds/useconds */
    switch (meta->base) {
        case NPY_FR_W:
            value *= 7;
            break;
        case NPY_FR_D:
            break;
        case NPY_FR_h:
            seconds = (int)((value % 24) * (60*60));
            value = value / 24;
            break;
        case NPY_FR_m:
            seconds = (int)(value % (24*60)) * 60;
            value = value / (24*60);
            break;
        case NPY_FR_s:
            seconds = (int)(value % (24*60*60));
            value = value / (24*60*60);
            break;
        case NPY_FR_ms:
            useconds = (int)(value % 1000) * 1000;
            value = value / 1000;
            seconds = (int)(value % (24*60*60));
            value = value / (24*60*60);
            break;
        case NPY_FR_us:
            useconds = (int)(value % (1000*1000));
            value = value / (1000*1000);
            seconds = (int)(value % (24*60*60));
            value = value / (24*60*60);
            break;
        default:
            break;
    }
    /*
     * 'value' represents days, and seconds/useconds are filled.
     *
     * If it would overflow the datetime.timedelta days, return a raw int
     */
    if (value < -999999999 || value > 999999999) {
        return PyLong_FromLongLong(td);
    }
    else {
        days = (int)value;
        ret = PyDelta_FromDSU(days, seconds, useconds);
        if (ret == NULL) {
            return NULL;
        }
    }

    /* If there is one event, just return the datetime */
    if (meta->events == 1) {
        return ret;
    }
    /* Otherwise return a tuple with the event in the second position */
    else {
        tup = PyTuple_New(2);
        if (tup == NULL) {
            Py_DECREF(ret);
            return NULL;
        }
        PyTuple_SET_ITEM(tup, 0, ret);

        ret = PyInt_FromLong(event);
        if (ret == NULL) {
            Py_DECREF(tup);
            return NULL;
        }
        PyTuple_SET_ITEM(tup, 1, ret);

        return tup;
    }
}

/*
 * Returns true if the datetime metadata matches
 */
NPY_NO_EXPORT npy_bool
has_equivalent_datetime_metadata(PyArray_Descr *type1, PyArray_Descr *type2)
{
    PyArray_DatetimeMetaData *meta1, *meta2;

    if ((type1->type_num != NPY_DATETIME &&
                        type1->type_num != NPY_TIMEDELTA) ||
                    (type2->type_num != NPY_DATETIME &&
                        type2->type_num != NPY_TIMEDELTA)) {
        return 0;
    }

    meta1 = get_datetime_metadata_from_dtype(type1);
    if (meta1 == NULL) {
        PyErr_Clear();
        return 0;
    }
    meta2 = get_datetime_metadata_from_dtype(type2);
    if (meta2 == NULL) {
        PyErr_Clear();
        return 0;
    }

    /* For generic units, the num and events are ignored */
    if (meta1->base == NPY_FR_GENERIC && meta2->base == NPY_FR_GENERIC) {
        return 1;
    }

    return meta1->base == meta2->base &&
            meta1->num == meta2->num &&
            meta1->events == meta2->events;
}

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
                          npy_datetime *dst_dt)
{
    npy_datetimestruct dts;

    /* If the metadata is the same, short-circuit the conversion */
    if (src_meta->base == dst_meta->base &&
            src_meta->num == dst_meta->num &&
            src_meta->events == dst_meta->events) {
        *dst_dt = src_dt;
        return 0;
    }

    /* Otherwise convert through a datetimestruct */
    if (convert_datetime_to_datetimestruct(src_meta, src_dt, &dts) < 0) {
            *dst_dt = NPY_DATETIME_NAT;
            return -1;
    }
    if (dts.event >= dst_meta->events) {
        dts.event = dts.event % dst_meta->events;
    }
    if (convert_datetimestruct_to_datetime(dst_meta, &dts, dst_dt) < 0) {
        *dst_dt = NPY_DATETIME_NAT;
        return -1;
    }

    return 0;
}

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
                          npy_timedelta *dst_dt)
{
    npy_int64 num = 0, denom = 0;
    int event = 0;

    /* If the metadata is the same, short-circuit the conversion */
    if (src_meta->base == dst_meta->base &&
            src_meta->num == dst_meta->num &&
            src_meta->events == dst_meta->events) {
        *dst_dt = src_dt;
        return 0;
    }

    /* Get the conversion factor */
    get_datetime_conversion_factor(src_meta, dst_meta, &num, &denom);

    if (num == 0) {
        return -1;
    }

    /* Remove the event number from the value */
    if (src_meta->events > 1) {
        event = (int)(src_dt % src_meta->events);
        src_dt = src_dt / src_meta->events;
        if (event < 0) {
            --src_dt;
            event += src_meta->events;
        }
    }

    /* Apply the scaling */
    if (src_dt < 0) {
        *dst_dt = (src_dt * num - (denom - 1)) / denom;
    }
    else {
        *dst_dt = src_dt * num / denom;
    }

    /* Add the event number back in */
    if (dst_meta->events > 1) {
        event = event % dst_meta->events;
        *dst_dt = (*dst_dt) * dst_meta->events + event;
    }

    return 0;
}

/*
 * Returns true if the object is something that is best considered
 * a Datetime, false otherwise.
 */
static npy_bool
is_any_numpy_datetime(PyObject *obj)
{
    return (PyArray_IsScalar(obj, Datetime) ||
            (PyArray_Check(obj) && (
                PyArray_DESCR(obj)->type_num == NPY_DATETIME)) ||
            PyDate_Check(obj) ||
            PyDateTime_Check(obj));
}

/*
 * Returns true if the object is something that is best considered
 * a Timedelta, false otherwise.
 */
static npy_bool
is_any_numpy_timedelta(PyObject *obj)
{
    return (PyArray_IsScalar(obj, Timedelta) ||
            (PyArray_Check(obj) && (
                PyArray_DESCR(obj)->type_num == NPY_TIMEDELTA)) ||
            PyDelta_Check(obj));
}

/*
 * Returns true if the object is something that is best considered
 * a Datetime or Timedelta, false otherwise.
 */
NPY_NO_EXPORT npy_bool
is_any_numpy_datetime_or_timedelta(PyObject *obj)
{
    return obj != NULL &&
           (is_any_numpy_datetime(obj) ||
            is_any_numpy_timedelta(obj));
}

NPY_NO_EXPORT PyObject *
datetime_arange(PyObject *start, PyObject *stop, PyObject *step,
                PyArray_Descr *dtype)
{
    PyArray_DatetimeMetaData meta;
    int is_timedelta = 0;
    /*
     * Both datetime and timedelta are stored as int64, so they can
     * share value variables.
     */
    npy_int64 start_value = 0, stop_value = 0, step_value;

    /*
     * First normalize the input parameters so there is no Py_None,
     * and start is moved to stop if stop is unspecified.
     */
    if (step == Py_None) {
        step = NULL;
    }
    if (stop == NULL || stop == Py_None) {
        stop = start;
        start = NULL;
        /* If start was NULL or None, raise an exception */
        if (stop == NULL || stop == Py_None) {
            PyErr_SetString(PyExc_ValueError,
                    "arange needs at least a stopping value");
            return NULL;
        }
    }
    if (start == Py_None) {
        start = NULL;
    }

    /* Step must not be a Datetime */
    if (step != NULL && is_any_numpy_datetime(step)) {
        PyErr_SetString(PyExc_ValueError,
                    "cannot use a datetime as a step in arange");
        return NULL;
    }

    /*
     * Now figure out whether we're doing a datetime or a timedelta
     * arange, and get the metadata for start and stop.
     */
    if (dtype != NULL) {
        PyArray_DatetimeMetaData *meta_tmp;

        if (dtype->type_num == NPY_TIMEDELTA) {
            is_timedelta = 1;
        }
        else if (dtype->type_num != NPY_DATETIME) {
            PyErr_SetString(PyExc_ValueError,
                        "datetime_arange was given a non-datetime dtype");
            return NULL;
        }

        meta_tmp = get_datetime_metadata_from_dtype(dtype);
        if (meta_tmp == NULL) {
            return NULL;
        }

        meta = *meta_tmp;

        if (is_timedelta) {
            /* Get 'start', defaulting to 0 if not provided */
            if (start == NULL) {
                start_value = 0;
            }
            else {
                if (convert_pyobject_to_timedelta(&meta, start,
                                                &start_value) < 0) {
                    return NULL;
                }
            }
            /* Get 'stop' */
            if (convert_pyobject_to_timedelta(&meta, stop,
                                            &stop_value) < 0) {
                return NULL;
            }
        }
        else {
            /* Get 'start', raising an exception if it is not provided */
            if (start == NULL) {
                PyErr_SetString(PyExc_ValueError,
                        "arange requires both a start and a stop for "
                        "NumPy datetime64 ranges");
                return NULL;
            }
            else {
                if (convert_pyobject_to_datetime(&meta, start,
                                                &start_value) < 0) {
                    return NULL;
                }
            }
            /*
             * Get 'stop', treating it specially if it's an integer
             * or a timedelta.
             */
            if (PyInt_Check(stop) ||
                            PyLong_Check(stop) ||
                            PyArray_IsScalar(stop, Integer) ||
                            is_any_numpy_timedelta(stop)) {
                if (convert_pyobject_to_timedelta(&meta, stop,
                                                &stop_value) < 0) {
                    return NULL;
                }

                /* Add the start value to make it a datetime */
                stop_value += start_value;
            }
            else {
                if (convert_pyobject_to_datetime(&meta, stop,
                                                &stop_value) < 0) {
                    return NULL;
                }
            }
        }
    }
    /* if dtype is NULL */
    else {
        /*
         * Start meta as generic units, so taking the gcd
         * does the right thing
         */
        meta.base = NPY_FR_GENERIC;

        /* Timedelta arange */
        if (!is_any_numpy_datetime(start) && !is_any_numpy_datetime(stop)) {
            PyArray_DatetimeMetaData meta_tmp, meta_mrg;

            is_timedelta = 1;

            /* Convert start to a timedelta */
            if (start == NULL) {
                start_value = 0;
            }
            else {
                /* Directly copy to meta, no point in gcd for 'start' */
                meta.base = -1;
                /* Convert 'start' to a timedelta */
                if (convert_pyobject_to_timedelta(&meta, start,
                                                &start_value) < 0) {
                    return NULL;
                }
            }

            /* Temporary metadata, to merge with 'meta' */
            meta_tmp.base = -1;
            /* Convert 'stop' to a timedelta */
            if (convert_pyobject_to_timedelta(&meta_tmp, stop,
                                            &stop_value) < 0) {
                return NULL;
            }

            /* Merge the metadata */
            if (compute_datetime_metadata_greatest_common_divisor(
                                    &meta, &meta_tmp, &meta_mrg, 1, 1) < 0) {
                return NULL;
            }

            /* Cast 'start' and 'stop' to the merged metadata */
            if (cast_timedelta_to_timedelta(&meta, &meta_mrg,
                                         start_value, &start_value) < 0) {
                return NULL;
            }
            if (cast_timedelta_to_timedelta(&meta_tmp, &meta_mrg,
                                         stop_value, &stop_value) < 0) {
                return NULL;
            }

            meta = meta_mrg;

            if (step == NULL) {
                step_value = 1;
            }
            else {
                /* Temporary metadata, to merge with 'meta' */
                meta_tmp.base = -1;
                /* Convert 'step' to a timedelta */
                if (convert_pyobject_to_timedelta(&meta_tmp, step,
                                                &step_value) < 0) {
                    return NULL;
                }

                /* Merge the metadata */
                if (compute_datetime_metadata_greatest_common_divisor(
                                    &meta, &meta_tmp, &meta_mrg, 1, 1) < 0) {
                    return NULL;
                }

                /* Cast 'start', 'stop', and 'step' to the merged metadata */
                if (cast_timedelta_to_timedelta(&meta, &meta_mrg,
                                         start_value, &start_value) < 0) {
                    return NULL;
                }
                if (cast_timedelta_to_timedelta(&meta, &meta_mrg,
                                         stop_value, &stop_value) < 0) {
                    return NULL;
                }
                if (cast_timedelta_to_timedelta(&meta_tmp, &meta_mrg,
                                         step_value, &step_value) < 0) {
                    return NULL;
                }

                meta = meta_mrg;
            }
        }
        /* Datetime arange */
        else {
            PyArray_DatetimeMetaData meta_tmp, meta_mrg;

            /* Get 'start', raising an exception if it is not provided */
            if (start == NULL) {
                PyErr_SetString(PyExc_ValueError,
                        "arange requires both a start and a stop for "
                        "NumPy datetime64 ranges");
                return NULL;
            }
            else {
                /* Directly copy to meta, no point in gcd for 'start' */
                meta.base = -1;
                /* Convert 'start' to a datetime */
                if (convert_pyobject_to_datetime(&meta, start,
                                                &start_value) < 0) {
                    return NULL;
                }
            }

            /* Temporary metadata, to merge with 'meta' */
            meta_tmp.base = -1;
            /* Convert 'stop' to a datetime */
            if (convert_pyobject_to_datetime(&meta_tmp, stop,
                                            &stop_value) < 0) {
                return NULL;
            }

            /* Merge the metadata */
            if (compute_datetime_metadata_greatest_common_divisor(
                                    &meta, &meta_tmp, &meta_mrg, 0, 0) < 0) {
                return NULL;
            }

            /* Cast 'start' and 'stop' to the merged metadata */
            if (cast_datetime_to_datetime(&meta, &meta_mrg,
                                         start_value, &start_value) < 0) {
                return NULL;
            }
            if (cast_datetime_to_datetime(&meta_tmp, &meta_mrg,
                                         stop_value, &stop_value) < 0) {
                return NULL;
            }

            meta = meta_mrg;

            if (step == NULL) {
                step_value = 1;
            }
            else {
                /* Temporary metadata, to merge with 'meta' */
                meta_tmp.base = -1;
                /* Convert 'step' to a timedelta */
                if (convert_pyobject_to_timedelta(&meta_tmp, step,
                                                &step_value) < 0) {
                    return NULL;
                }

                /* Merge the metadata */
                if (compute_datetime_metadata_greatest_common_divisor(
                                    &meta, &meta_tmp, &meta_mrg, 0, 1) < 0) {
                    return NULL;
                }

                /* Cast 'start', 'stop', and 'step' to the merged metadata */
                if (cast_datetime_to_datetime(&meta, &meta_mrg,
                                         start_value, &start_value) < 0) {
                    return NULL;
                }
                if (cast_datetime_to_datetime(&meta, &meta_mrg,
                                         stop_value, &stop_value) < 0) {
                    return NULL;
                }
                if (cast_timedelta_to_timedelta(&meta_tmp, &meta_mrg,
                                         step_value, &step_value) < 0) {
                    return NULL;
                }

                meta = meta_mrg;
            }
        }
    }
    
    PyErr_SetString(PyExc_RuntimeError, "datetime_arange is incomplete");
    return NULL;
}
