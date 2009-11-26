#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <datetime.h>

#include <time.h>

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include <numpy/ndarrayobject.h>

#include "npy_config.h"

#include "_datetime.h"

/* For defaults and errors */
#define NPY_FR_ERR  -1

/* Offset for number of days between Dec 31, 1969 and Jan 1, 0001
*  Assuming Gregorian calendar was always in effect (proleptic Gregorian calendar)
*/

/* Calendar Structure for Parsing Long -> Date */
typedef struct {
    int year, month, day;
} ymdstruct;

typedef struct {
    int hour, min, sec;
} hmsstruct;


/*
  ====================================================
  == Beginning of section borrowed from mx.DateTime ==
  ====================================================
*/

/*
 * Functions in the following section are borrowed from mx.DateTime version
 * 2.0.6, and hence this code is subject to the terms of the egenix public
 * license version 1.0.0
 */

#define Py_AssertWithArg(x,errortype,errorstr,a1) {if (!(x)) {PyErr_Format(errortype,errorstr,a1);goto onError;}}

/* Table with day offsets for each month (0-based, without and with leap) */
static int month_offset[2][13] = {
    { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 },
    { 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366 }
};

/* Table of number of days in a month (0-based, without and with leap) */
static int days_in_month[2][12] = {
    { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
    { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};

/* Return 1/0 iff year points to a leap year in calendar. */
static int
is_leapyear(register long year)
{
    return (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
}


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
 * Return the year offset, that is the absolute date of the day
 * 31.12.(year-1) since 31.12.1969 in the proleptic Gregorian calendar.
 */
static npy_longlong
year_offset(register npy_longlong year)
{
    /* Note that 477 == 1969/4 - 1969/100 + 1969/400 */
    year--;
    if (year >= 0 || -1/4 == -1)
        return (year-1969)*365 + year/4 - year/100 + year/400 - 477;
    else
        return (year-1969)*365 + (year-3)/4 - (year-99)/100 + (year-399)/400 - 477;
}

/*
 * Modified version of mxDateTime function
 * Returns absolute number of days since Jan 1, 1970
 * assuming a proleptic Gregorian Calendar
 * Raises a ValueError if out of range month or day
 * day -1 is Dec 31, 1969, day 0 is Jan 1, 1970, day 1 is Jan 2, 1970
 */
static npy_longlong
days_from_ymd(int year, int month, int day)
{

    /* Calculate the absolute date */
    int leap;
    npy_longlong yearoffset, absdate;

    /* Is it a leap year ? */
    leap = is_leapyear(year);

    /* Negative month values indicate months relative to the years end */
    if (month < 0) month += 13;
    Py_AssertWithArg(month >= 1 && month <= 12,
                     PyExc_ValueError,
                     "month out of range (1-12): %i",
                     month);

    /* Negative values indicate days relative to the months end */
    if (day < 0) day += days_in_month[leap][month - 1] + 1;
    Py_AssertWithArg(day >= 1 && day <= days_in_month[leap][month - 1],
                     PyExc_ValueError,
                     "day out of range: %i",
                     day);

    /*
     * Number of days between Dec 31, (year - 1) and Dec 31, 1969
     *    (can be negative).
     */
    yearoffset = year_offset(year);

    if (PyErr_Occurred()) goto onError;

    /*
     * Calculate the number of days using yearoffset
     * Jan 1, 1970 is day 0 and thus Dec. 31, 1969 is day -1
     */
    absdate = day-1 + month_offset[leap][month - 1] + yearoffset;

    return absdate;

 onError:
    return 0;

}

/* Returns absolute seconds from an hour, minute, and second
 */
#define secs_from_hms(hour, min, sec) ((hour)*3600 + (min)*60 + (sec))

/*
 * Takes a number of days since Jan 1, 1970 (positive or negative)
 * and returns the year. month, and day in the proleptic
 * Gregorian calendar
 *
 * Examples:
 *
 * -1 returns 1969, 12, 31
 * 0  returns 1970, 1, 1
 * 1  returns 1970, 1, 2
 */

static ymdstruct
days_to_ymdstruct(npy_datetime dlong)
{
    ymdstruct ymd;
    register long year;
    npy_longlong yearoffset;
    int leap, dayoffset;
    int month = 1, day = 1;
    int *monthoffset;

    dlong += 1;

    /* Approximate year */
    year = 1970 + dlong / 365.2425;

    /* Apply corrections to reach the correct year */
    while (1) {
        /* Calculate the year offset */
        yearoffset = year_offset(year);

        /*
         * Backward correction: absdate must be greater than the
         * yearoffset
         */
        if (yearoffset >= dlong) {
            year--;
            continue;
        }

        dayoffset = dlong - yearoffset;
        leap = is_leapyear(year);

        /* Forward correction: non leap years only have 365 days */
        if (dayoffset > 365 && !leap) {
            year++;
            continue;
        }
        break;
    }

    /* Now iterate to find the month */
    monthoffset = month_offset[leap];
    for (month = 1; month < 13; month++) {
        if (monthoffset[month] >= dayoffset)
            break;
    }
    day = dayoffset - month_offset[leap][month-1];

    ymd.year  = year;
    ymd.month = month;
    ymd.day   = day;

    return ymd;
}

/*
 * Converts an integer number of seconds in a day to hours minutes seconds.
 * It assumes seconds is between 0 and 86399.
 */

static hmsstruct
seconds_to_hmsstruct(npy_longlong dlong)
{
    int hour, minute, second;
    hmsstruct hms;

    hour   = dlong / 3600;
    minute = (dlong % 3600) / 60;
    second = dlong - (hour*3600 + minute*60);

    hms.hour   = hour;
    hms.min = minute;
    hms.sec = second;

    return hms;
}

/*
  ====================================================
  == End of section adapted from mx.DateTime       ==
  ====================================================
*/


/*==================================================
// Parsing DateTime struct and returns a date-time number
// =================================================

 Structure is assumed to be already normalized
*/

/*
 * Create a datetime value from a filled datetime struct and resolution unit.
 */
NPY_NO_EXPORT npy_datetime
PyArray_DatetimeStructToDatetime(NPY_DATETIMEUNIT fr, npy_datetimestruct *d)
{
    npy_datetime ret;
    npy_longlong days; /* The absolute number of days since Jan 1, 1970 */

    if (fr > NPY_FR_M) {
        days = days_from_ymd(d->year, d->month, d->day);
    }
    if (fr == NPY_FR_Y) {
        ret = d->year - 1970;
    }
    else if (fr == NPY_FR_M) {
        ret = (d->year - 1970) * 12 + d->month - 1;
    }
    else if (fr == NPY_FR_W) {
        /* This is just 7-days for now. */
        if (days >= 0) {
            ret = days / 7;
        }
        else {
            ret = (days - 6) / 7;
        }
    }
    else if (fr == NPY_FR_B) {
        npy_longlong x;
        int dotw = day_of_week(days);

        if (dotw > 4) {
            /* Invalid business day */
            ret = 0;
        }
        else {
            if (days >= 0) {
                /* offset to adjust first week */
                x = days - 4;
            }
            else {
                x = days - 2;
            }
            ret = 2 + (x / 7) * 5 + x % 7;
        }
    }
    else if (fr == NPY_FR_D) {
        ret = days;
    }
    else if (fr == NPY_FR_h) {
        ret = days * 24 + d->hour;
    }
    else if (fr == NPY_FR_m) {
        ret = days * 1440 + d->hour * 60 + d->min;
    }
    else if (fr == NPY_FR_s) {
        ret = days * 86400 + secs_from_hms(d->hour, d->min, d->sec);
    }
    else if (fr == NPY_FR_ms) {
        ret = days * 86400000 + secs_from_hms(d->hour, d->min, d->sec) * 1000
            + (d->us / 1000);
    }
    else if (fr == NPY_FR_us) {
        npy_int64 num = 86400 * 1000;
        num *= 1000;
        ret = days * num + secs_from_hms(d->hour, d->min, d->sec) * 1000000
            + d->us;
    }
    else if (fr == NPY_FR_ns) {
        npy_int64 num = 86400 * 1000;
        num *= 1000 * 1000;
        ret = days * num + secs_from_hms(d->hour, d->min, d->sec) * 1000000000
            + d->us * 1000 + (d->ps / 1000);
    }
    else if (fr == NPY_FR_ps) {
        npy_int64 num2 = 1000 * 1000;
        npy_int64 num1;

        num2 *= 1000 * 1000;
        num1 = 86400 * num2;
        ret = days * num1 + secs_from_hms(d->hour, d->min, d->sec) * num2
            + d->us * 1000000 + d->ps;
    }
    else if (fr == NPY_FR_fs) {
        /* only 2.6 hours */
        npy_int64 num2 = 1000000;
        num2 *= 1000000;
        num2 *= 1000;

        /* get number of seconds as a postive or negative number */
        if (days >= 0) {
            ret = secs_from_hms(d->hour, d->min, d->sec);
        }
        else {
            ret = ((d->hour - 24)*3600 + d->min*60 + d->sec);
        }
        ret = ret * num2 + d->us * 1000000000 + d->ps * 1000 + (d->as / 1000);
    }
    else if (fr == NPY_FR_as) {
        /* only 9.2 secs */
        npy_int64 num1, num2;

        num1 = 1000000;
        num1 *= 1000000;
        num2 = num1 * 1000000;

        if (days >= 0) {
            ret = d->sec;
        }
        else {
            ret = d->sec - 60;
        }
        ret = ret * num2 + d->us * num1 + d->ps * 1000000 + d->as;
    }
    else {
        /* Shouldn't get here */
        PyErr_SetString(PyExc_ValueError, "invalid internal frequency");
        ret = -1;
    }

    return ret;
}

/* Uses Average values when frequency is Y, M, or B */

#define _DAYS_PER_MONTH 30.436875
#define _DAYS_PER_YEAR  365.2425

/*
 * Create a timdelta value from a filled timedelta struct and resolution unit.
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
        ret = d->day * 1440 + d->sec / 60;
    }
    else if (fr == NPY_FR_s) {
        ret = d->day * 86400 + d->sec;
    }
    else if (fr == NPY_FR_ms) {
        ret = d->day * 86400000 + d->sec * 1000 + d->us / 1000;
    }
    else if (fr == NPY_FR_us) {
        npy_int64 num = 86400000;
        num *= 1000;
        ret = d->day * num + d->sec * 1000000 + d->us;
    }
    else if (fr == NPY_FR_ns) {
        npy_int64 num = 86400000;
        num *= 1000000;
        ret = d->day * num + d->sec * 1000000000 + d->us * 1000 + (d->ps / 1000);
    }
    else if (fr == NPY_FR_ps) {
        npy_int64 num2, num1;

        num2 = 1000000;
        num2 *= 1000000;
        num1 = 86400 * num2;

        ret = d->day * num1 + d->sec * num2 + d->us * 1000000 + d->ps;
    }
    else if (fr == NPY_FR_fs) {
        /* only 2.6 hours */
        npy_int64 num2 = 1000000000;
        num2 *= 1000000;
        ret = d->sec * num2 + d->us * 1000000000 + d->ps * 1000 + (d->as / 1000);
    }
    else if (fr == NPY_FR_as) {
        /* only 9.2 secs */
        npy_int64 num1, num2;

        num1 = 1000000;
        num1 *= 1000000;
        num2 = num1 * 1000000;
        ret = d->sec * num2 + d->us * num1 + d->ps * 1000000 + d->as;
    }
    else {
        /* Shouldn't get here */
        PyErr_SetString(PyExc_ValueError, "invalid internal frequency");
        ret = -1;
    }

    return ret;
}



/*
 * Fill the datetime struct from the value and resolution unit.
 */
NPY_NO_EXPORT void
PyArray_DatetimeToDatetimeStruct(npy_datetime val, NPY_DATETIMEUNIT fr,
                                 npy_datetimestruct *result)
{
    int year = 1970, month = 1, day = 1,
        hour = 0, min = 0, sec = 0,
        us = 0, ps = 0, as = 0;

    npy_int64 tmp;
    ymdstruct ymd;
    hmsstruct hms;

    /*
     * Note that what looks like val / N and val % N for positive numbers maps to
     * [val - (N-1)] / N and [N-1 + (val+1) % N] for negative numbers (with the 2nd
     * value, the remainder, being positive in both cases).
     */
    if (fr == NPY_FR_Y) {
        year = 1970 + val;
    }
    else if (fr == NPY_FR_M) {
        if (val >= 0) {
            year  = 1970 + val / 12;
            month = val % 12 + 1;
        }
        else {
            year  = 1969 + (val + 1) / 12;
            month = 12 + (val + 1)% 12;
        }
    }
    else if (fr == NPY_FR_W) {
        /* A week is the same as 7 days */
        ymd = days_to_ymdstruct(val * 7);
        year  = ymd.year;
        month = ymd.month;
        day   = ymd.day;
    }
    else if (fr == NPY_FR_B) {
        /* Number of business days since Thursday, 1-1-70 */
        npy_longlong absdays;
        /*
         * A buisness day is M T W Th F (i.e. all but Sat and Sun.)
         * Convert the business day to the number of actual days.
         *
         * Must convert [0,1,2,3,4,5,6,7,...] to
         *                  [0,1,4,5,6,7,8,11,...]
         * and  [...,-9,-8,-7,-6,-5,-4,-3,-2,-1,0] to
         *        [...,-13,-10,-9,-8,-7,-6,-3,-2,-1,0]
         */
        if (val >= 0) {
            absdays = 7 * ((val + 3) / 5) + ((val + 3) % 5) - 3;
        }
        else {
            /* Recall how C computes / and % with negative numbers */
            absdays = 7 * ((val - 1) / 5) + ((val - 1) % 5) + 1;
        }
        ymd = days_to_ymdstruct(absdays);
        year  = ymd.year;
        month = ymd.month;
        day   = ymd.day;
    }
    else if (fr == NPY_FR_D) {
        ymd = days_to_ymdstruct(val);
        year  = ymd.year;
        month = ymd.month;
        day   = ymd.day;
    }
    else if (fr == NPY_FR_h) {
        if (val >= 0) {
            ymd  = days_to_ymdstruct(val / 24);
            hour  = val % 24;
        }
        else {
            ymd  = days_to_ymdstruct((val - 23) / 24);
            hour = 23 + (val + 1) % 24;
        }
        year  = ymd.year;
        month = ymd.month;
        day   = ymd.day;
    }
    else if (fr == NPY_FR_m) {
        if (val >= 0) {
            ymd = days_to_ymdstruct(val / 1440);
            min = val % 1440;
        }
        else {
            ymd = days_to_ymdstruct((val - 1439) / 1440);
            min = 1439 + (val + 1) % 1440;
        }
        hms = seconds_to_hmsstruct(min * 60);
        year   = ymd.year;
        month  = ymd.month;
        day    = ymd.day;
        hour   = hms.hour;
        min = hms.min;
    }
    else if (fr == NPY_FR_s) {
        if (val >= 0) {
            ymd = days_to_ymdstruct(val / 86400);
            sec = val % 86400;
        }
        else {
            ymd = days_to_ymdstruct((val - 86399) / 86400);
            sec = 86399 + (val + 1) % 86400;
        }
        hms = seconds_to_hmsstruct(val);
        year   = ymd.year;
        month  = ymd.month;
        day    = ymd.day;
        hour   = hms.hour;
        min = hms.min;
        sec = hms.sec;
    }
    else if (fr == NPY_FR_ms) {
        if (val >= 0) {
            ymd = days_to_ymdstruct(val / 86400000);
            tmp  = val % 86400000;
        }
        else {
            ymd = days_to_ymdstruct((val - 86399999) / 86400000);
            tmp  = 86399999 + (val + 1) % 86399999;
        }
        hms = seconds_to_hmsstruct(tmp / 1000);
        us  = (tmp % 1000)*1000;
        year    = ymd.year;
        month   = ymd.month;
        day     = ymd.day;
        hour    = hms.hour;
        min     = hms.min;
        sec     = hms.sec;
    }
    else if (fr == NPY_FR_us) {
        npy_int64 num1, num2;
        num1 = 86400000;
        num1 *= 1000;
        num2 = num1 - 1;
        if (val >= 0) {
            ymd = days_to_ymdstruct(val / num1);
            tmp = val % num1;
        }
        else {
            ymd = days_to_ymdstruct((val - num2)/ num1);
            tmp = num2 + (val + 1) % num1;
        }
        hms = seconds_to_hmsstruct(tmp / 1000000);
        us = tmp % 1000000;
        year    = ymd.year;
        month   = ymd.month;
        day     = ymd.day;
        hour    = hms.hour;
        min     = hms.min;
        sec     = hms.sec;
    }
    else if (fr == NPY_FR_ns) {
        npy_int64 num1, num2, num3;
        num1 = 86400000;
        num1 *= 1000000000;
        num2 = num1 - 1;
        num3 = 1000000;
        num3 *= 1000000;
        if (val >= 0) {
            ymd = days_to_ymdstruct(val / num1);
            tmp = val % num1;
        }
        else {
            ymd = days_to_ymdstruct((val - num2)/ num1);
            tmp = num2 + (val + 1) % num1;
        }
        hms = seconds_to_hmsstruct(tmp / 1000000000);
        tmp = tmp % 1000000000;
        us = tmp / 1000;
        ps = (tmp % 1000) * 1000;
        year    = ymd.year;
        month   = ymd.month;
        day     = ymd.day;
        hour    = hms.hour;
        min     = hms.min;
        sec     = hms.sec;
    }
    else if (fr == NPY_FR_ps) {
        npy_int64 num1, num2, num3;
        num3 = 1000000000;
        num3 *= 1000;
        num1 = 86400 * num3;
        num2 = num1 - 1;

        if (val >= 0) {
            ymd = days_to_ymdstruct(val / num1);
            tmp = val % num1;
        }
        else {
            ymd = days_to_ymdstruct((val - num2)/ num1);
            tmp = num2 + (val + 1) % num1;
        }
        hms = seconds_to_hmsstruct(tmp / num3);
        tmp = tmp % num3;
        us = tmp / 1000000;
        ps = tmp % 1000000;
        year    = ymd.year;
        month   = ymd.month;
        day     = ymd.day;
        hour    = hms.hour;
        min     = hms.min;
        sec     = hms.sec;
    }
    else if (fr == NPY_FR_fs) {
        /* entire range is only += 2.6 hours */
        npy_int64 num1, num2;
        num1 = 1000000000;
        num1 *= 1000;
        num2 = num1 * 1000;

        if (val >= 0) {
            sec = val / num2;
            tmp = val % num2;
            hms = seconds_to_hmsstruct(sec);
            hour = hms.hour;
            min = hms.min;
            sec = hms.sec;
        }
        else {
            /* tmp (number of fs) will be positive after this segment */
            year = 1969;
            day = 31;
            month = 12;
            sec = (val - (num2-1))/num2;
            tmp = (num2-1) + (val + 1) % num2;
            if (sec == 0) {
                /* we are at the last second */
                hour = 23;
                min = 59;
                sec = 59;
            }
            else {
                hour = 24 + (sec - 3599)/3600;
                sec = 3599 + (sec+1)%3600;
                min = sec / 60;
                sec = sec % 60;
            }
        }
        us = tmp / 1000000000;
        tmp = tmp % 1000000000;
        ps = tmp / 1000;
        as = (tmp % 1000) * 1000;
    }
    else if (fr == NPY_FR_as) {
        /* entire range is only += 9.2 seconds */
        npy_int64 num1, num2, num3;
        num1 = 1000000;
        num2 = num1 * 1000000;
        num3 = num2 * 1000000;
        if (val >= 0) {
            hour = 0;
            min = 0;
            sec = val / num3;
            tmp = val % num3;
        }
        else {
            year = 1969;
            day = 31;
            month = 12;
            hour = 23;
            min = 59;
            sec = 60 + (val - (num3-1)) / num3;
            tmp = (num3-1) + (val+1) % num3;
        }
        us = tmp / num2;
        tmp = tmp % num2;
        ps = tmp / num1;
        as = tmp % num1;
    }
    else {
        PyErr_SetString(PyExc_RuntimeError, "invalid internal time resolution");
    }

    result->year  = year;
    result->month = month;
    result->day   = day;
    result->hour  = hour;
    result->min   = min;
    result->sec   = sec;
    result->us    = us;
    result->ps    = ps;
    result->as    = as;

    return;
}

/*
 * FIXME: Overflow is not handled at all
 *   To convert from Years, Months, and Business Days, multiplication by the average is done
 */

/*
 * Fill the timedelta struct from the timedelta value and resolution unit.
 */
NPY_NO_EXPORT void
PyArray_TimedeltaToTimedeltaStruct(npy_timedelta val, NPY_DATETIMEUNIT fr,
                                 npy_timedeltastruct *result)
{
    npy_longlong day=0;
    int sec=0, us=0, ps=0, as=0;
    npy_bool negative=0;

    /*
     * Note that what looks like val / N and val % N for positive numbers maps to
     * [val - (N-1)] / N and [N-1 + (val+1) % N] for negative numbers (with the 2nd
     * value, the remainder, being positive in both cases).
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
        day = val / 86400;
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
        ps  = (val % 1000) * 1000;
    }
    else if (fr == NPY_FR_ps) {
        npy_int64 num1, num2;
        num2 = 1000000000;
        num2 *= 1000;
        num1 = 86400 * num2;

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
        num2 = num1 * 1000000;

        day = 0;
        sec = val / num2;
        val = val % num2;
        us = val / num1;
        val = val % num1;
        ps = val / 1000;
        as = (val % 1000) * 1000;
    }
    else if (fr == NPY_FR_as) {
        /* entire range is only += 2.6 seconds */
        npy_int64 num1, num2, num3;
        num1 = 1000000;
        num2 = num1 * 1000000;
        num3 = num2 * 1000000;
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
