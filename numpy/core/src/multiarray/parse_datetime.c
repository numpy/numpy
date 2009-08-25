
#include <datetime.h>
#include <time.h>

/* For defaults and errors */
#define NPY_FR_ERR  -1

/* Offset for number of days between Jan 1, 1970 and Jan 1, 0001 */
*  Assuming Gregorian calendar was always in effect 
*/

#define DAYS_EPOCH 719163

/* Calendar Structure for Parsing Long -> Date */
typedef struct {
    int year, month, day;
} ymdstruct;

typedef struct {
    int hour, minute, second;
} hmsstruct;

typedef struct {
    int year, month, day, hour,
	minute, second, msecond,
	usecond, nsecond, psecond, fsecond,
	asecond;
} datestruct;


/* =============
 *  callbacks
 * ============ 
 */

static PyObject *callback = NULL;

static PyObject *
set_callback(PyObject *dummy, PyObject *args)
{
    PyObject *result = NULL;
    PyObject *temp;

    if (PyArg_ParseTuple(args, "O:set_callback", &temp))
	{
	    if (!PyCallable_Check(temp))
		{
		    PyErr_SetString(PyExc_TypeError, "parameter must be callable");
		    return NULL;
		}
	    // Reference to new callback
	    Py_XINCREF(temp);
	    // Dispose of previous callback
	    Py_XDECREF(callback);
	    // Remember new callback
	    callback = temp;
	    // Boilerplate to return "None"
	    Py_INCREF(Py_None);
	    result = Py_None;
	}
	
    return result;
}


PyObject *DateCalc_RangeError = NULL;
PyObject *DateCalc_Error      = NULL;

// Frequency Checker
int _check_freq(int freq)
{
    return freq;
}
/*
  ====================================================
  == Beginning of section borrowed from mx.DateTime ==
  ====================================================
*/

/*
  Functions in the following section are borrowed from mx.DateTime version
  2.0.6, and hence this code is subject to the terms of the egenix public
  license version 1.0.0
*/

#define Py_AssertWithArg(x,errortype,errorstr,a1) {if (!(x)) {PyErr_Format(errortype,errorstr,a1);goto onError;}}
#define Py_Error(errortype,errorstr) {PyErr_SetString(errortype,errorstr);goto onError;}

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
static
int is_leapyear(register long year)
{
    return (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
}


/* Return the day of the week for the given absolute date. */
static
int day_of_week(register long absdate)
{
    int day_of_week;

    // Add in three for the Thursday on Jan 1, 1970 (epoch offset)
    if (absdate >= 0) 
	day_of_week = (absdate + 4) % 7;
    else 
	day_of_week = 6 - ((-absdate + 2) % 7);
    
    return day_of_week;
}

/* Return the year offset, that is the absolute date of the day
   31.12.(year-1) in the given calendar.
*/
static
long year_offset(register long year)
{
    year--;
    if (year >= 0 || -1/4 == -1)
	return year*365 + year/4 - year/100 + year/400;
    else
	return year*365 + (year-3)/4 - (year-99)/100 + (year-399)/400;
}    

/*
 * UNUSED FUNCTION
 static
 int week_from_ady(long absdate, int day, int year)
 {
 int week, dotw, day_of_year;
 dotw = day_of_week(absdate);
 day_of_year = (int)(absdate - year_offset(year) + DAYS_EPOCH);

 // Estimate
 week = (day_of_year - 1) - dotw + 3;
 if (week >= 0) 
 week = week / 7 + 1;

 // Verify 
 if (week < 0) 
 {
 // The day lies in last week of the previous year 
 if ((week > -2) || ((week == -2) && (is_leapyear(year-1))))
 week = 53;
 else
 week = 52;
 } 
 else if (week == 53) 
 {
 // Check if the week belongs to year or year + 1 
 if ((31 - day + dotw) < 3) 
 week = 1;
 }

 return week;
 }
*/


// Modified version of mxDateTime function
// Returns absolute number of days since Jan 1, 1970
static
long long absdays_from_ymd(int year, int month, int day)
{

    /* Calculate the absolute date */
    int leap;
    long yearoffset, absdate;

    /* Range check */
    Py_AssertWithArg(year > -(INT_MAX / 366) && year < (INT_MAX / 366),
		     DateCalc_RangeError,
		     "year out of range: %i",
		     year);

    /* Is it a leap year ? */
    leap = is_leapyear(year);

    /* Negative month values indicate months relative to the years end */
    if (month < 0) month += 13;
    Py_AssertWithArg(month >= 1 && month <= 12,
		     DateCalc_RangeError,
		     "month out of range (1-12): %i",
		     month);

    /* Negative values indicate days relative to the months end */
    if (day < 0) day += days_in_month[leap][month - 1] + 1;
    Py_AssertWithArg(day >= 1 && day <= days_in_month[leap][month - 1],
		     DateCalc_RangeError,
		     "day out of range: %i",
		     day);

    // Number of days between (year - 1) and 1970
    // !! This is a bad implementation: if year_offset overflows a long, we lose a potential
    //     of DAYS_EPOCH days range
    yearoffset = year_offset(year) - DAYS_EPOCH;

    if (PyErr_Occurred()) goto onError;

    // Calculate the number of days using yearoffset
    absdate = day + month_offset[leap][month - 1] + yearoffset;

    return absdate;

 onError:
    // do bad stuff
    return 0;

}

// Returns absolute seconds from an hour, minute, and second
static
long long abssecs_from_hms(int hour, int minute, int second)
{
    // Needs to perform checks for valid times
    return hour * 3600 + minute * 60 + second;
}
static
ymdstruct long_to_ymdstruct(long long dlong)
{
    ymdstruct ymd;
    register long year;
    long long yearoffset;
    int leap, dayoffset;
    int month = 1, day = 1;
    int *monthoffset;
    dlong += DAYS_EPOCH;

    /* Approximate year */
    year = 1970 + dlong / 365.25;
    
    if (dlong > 0) year++;

    /* Apply corrections to reach the correct year */
    while (1) {
	/* Calculate the year offset */
	yearoffset = year_offset(year);

	/* Backward correction: absdate must be greater than the
	   yearoffset */
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
    {
	for (month = 1; month < 13; month++) {
	    if (monthoffset[month] >= dayoffset)
		break;
	}
	day = dayoffset - month_offset[leap][month-1];
    }
	
    ymd.year  = year;
    ymd.month = month;
    ymd.day   = day;

    return ymd;
}

/* Sets the time part of the DateTime object. */
static
hmsstruct long_to_hmsstruct(long long dlong)
{
    int hour, minute, second;
    hmsstruct hms;

    // Make dlong within a one day period
    dlong = dlong % 86400;

    if (dlong < 0)
	dlong = 86400 + dlong;
    hour   = dlong / 3600;
    minute = (dlong % 3600) / 60;
    second = dlong - (hour*3600 + minute*60);

    hms.hour   = hour;
    hms.minute = minute;
    hms.second = second;

    return hms;
}


/*
  ====================================================
  == End of section borrowed from mx.DateTime       ==
  ====================================================
*/

//==================================================
// Parsing datetime/datestring to long
// =================================================

// Takes a datetime object and a string as frequency
// Returns the number of (frequency) since Jan 1, 1970
static 
long long datetime_to_long(PyObject* datetime, int frequency)
{
    int year = 0, month = 0, day = 0, hour = 0, 
	minute = 0, second = 0, microsecond = 0;
	
    // Get the time units from PyDateTime
    year        = PyDateTime_GET_YEAR(datetime);
    month       = PyDateTime_GET_MONTH(datetime);
    day         = PyDateTime_GET_DAY(datetime);

    minute      = PyDateTime_DATE_GET_MINUTE(datetime);
    second      = PyDateTime_DATE_GET_SECOND(datetime);
    microsecond = PyDateTime_DATE_GET_MICROSECOND(datetime);
	
    // The return value
    long long result = 0;

    // The absolute number of days since 1970
    long long absdays = absdays_from_ymd(year, month, day);

    // These calculations depend on the frequency

    if (frequency == NPY_FR_Y) {
	result = year - 1970;
    } else if (frequency == NPY_FR_M) {
	result = (year - 1970) * 12 + month - 1;
    } else if (frequency == NPY_FR_W) {
	// 4 day offset for post 1970 to get correct week
	int dotw = day_of_week(absdays);
	if (absdays >= 0)
	    result = (absdays + 4) / 7;
	else
	    result = (absdays - dotw)/ 7;
    } else if (frequency == NPY_FR_B) {
	int dotw = day_of_week(absdays);
	// Post epoch
	if (year >= 1970) {
	    // To get to Sunday, Jan 4, 1970
	    // number of weeks * 5 + dotw [0-6] - Saturdays + 1 for offset
	    if (absdays > 2)
		result = ((absdays - dotw) / 7) * 5 + dotw - (dotw / 6) + 1;
	    else 
		result = dotw - 4 - (dotw / 6);
	    // Pre epoch
	} else {
	    // To get beyond Sunday, Dec 28, 1969
	    if (absdays < -4) {
		// Offset by 1 for Sundays
		if (dotw)
		    result = ((absdays + 7 - dotw) / 7) * 5 - (6 - dotw) - 3;
		else
		    result = ((absdays + 7 - dotw) / 7) * 5 - (6 - dotw) - 2;
	    } else {
		// Offset by 1 for Sundays
		if (dotw)
		    result = -4 + dotw;
		else
		    result = -3; // Sunday, Dec 28, 1969
	    }
	}
    } else if (frequency == NPY_FR_D) {
	result = absdays;
    } else if (frequency == NPY_FR_h) {
	result = absdays * 24 + hour;
    } else if (frequency == NPY_FR_m) {
	result = absdays * 1440 + hour * 60 + minute;
    } else if (frequency == NPY_FR_s) {
	result = absdays * 86400LL + abssecs_from_hms(hour, minute, second);
    } else if (frequency == NPY_FR_ms) {
	result = absdays * 86400000LL + abssecs_from_hms(hour, minute, second) * 1000LL
	    + (microsecond / 1000LL);
    } else if (frequency == NPY_FR_us) {
	result = absdays * 86400000000LL + abssecs_from_hms(hour, minute, second) * 1000000LL
	    + microsecond;
    }
    // Starting from here, we need extra units (ns, ps, fs, as)
    //  for correct precision: datetime doesn't include beyond microsecond
    else if (frequency == NPY_FR_ns) {
	PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
	result = 0;
    } else if (frequency == NPY_FR_ps) {
	PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
	result = 0;
    } else if (frequency == NPY_FR_fs) {
	PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
	result = 0;
    } else if (frequency == NPY_FR_as) {
	PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
	result = 0;
    } else {
	// Throw some Not Valid Frequency error here
	result = -1;
    }	

    return result;
}

// Takes a string object as the date, and a string as frequency, 
//  parses that into a datetime and passes the datetime object 
//  to datetime_to_long
// Returns the number of (frequency) since Jan 1, 1970
static
long long datestring_to_long(PyObject *string, int frequency)
{
    // Send to datetime_to_long
    PyObject *datetime = NULL;
    long long result = 0;

    // Make the string into a tuple for the callback function
    PyObject *string_arg = PyTuple_New(1);
    PyTuple_SET_ITEM(string_arg, 0, string);
    Py_INCREF(string);

    // Parse the string into a datetime object
    datetime = PyEval_CallObject(callback, string_arg);

    Py_DECREF(string_arg);

    // If the parsing worked, send the datetime and frequency 
    //  to datetime_to_long
    if (datetime)
	{
	    result = datetime_to_long(datetime, frequency);
	}
    else
	{
	    PyErr_SetString(PyExc_TypeError, "error processing datetime");
	    //return NULL;
	    //Return bad stuff
	    result = -1;
	}

    return result;
}

// This is the callable wrapper for datestring/datetime_to_long
// Decides if the arguments are a string or a datetime object
//  and passes them to the correct datestring/datetime function
// Returns a PyLong generated from the appropriate function
static PyObject *
date_to_long(PyObject *self, PyObject *args)
{
    PyObject *date_arg = NULL;    // string or datetime
    PyObject *freq_arg = NULL;	  // frequency as string
    PyObject *result   = NULL;	  // long result

    int freq = NPY_FR_ERR;			  // freq_arg is a PyObject to be parsed to freq

    // macro PyDateTime_IMPORT must be invoked for PyDateTime_Check
    PyDateTime_IMPORT;

    // Make sure the callback function is set
    //  ! This doesn't check to make sure it's the right callback function
    //  ! This should all be done in some init script
    if (!PyCallable_Check(callback))
	{
	    PyErr_SetString(PyExc_TypeError, "callback not set.");
	    return NULL;
	}

    // Parse out date_arg & freq_arg
    if (!PyArg_ParseTuple(args, "OO", &date_arg, &freq_arg))
	{
	    return NULL;
	}

    // Make sure frequency is not NULL
    if (!freq_arg)
	{	
	    PyErr_SetString(PyExc_TypeError, "frequency not set.");
	    return NULL;
	}

    // Parse out frequency into an int so we can use it easily
    if ((freq = _unit_from_str(PyString_AsString(freq_arg))) == NPY_FR_ERR)
	{
	    // If the frequency is invalid, set an error and return null
	    PyErr_SetString(PyExc_TypeError, "invalid frequency.");
	    return NULL;
	}

    // Make sure date_arg is not NULL
    if (!date_arg)
	{
	    PyErr_SetString(PyExc_TypeError, "no date provided.");
	    return NULL;
	}

    // Decide if the date_arg is a string or a datetime
    if (PyString_Check(date_arg))
	{
	    // XXX PyINCREF here?
	    // date_arg is a string, so return datestring_to_long
	    result = PyLong_FromLongLong(datestring_to_long(date_arg, freq));
	}
    else if (PyDateTime_Check(date_arg))
	{
	    // XXX PyINCREF here?
	    // date_arg is a datetime, so return datetime_to_long
	    result = PyLong_FromLongLong(datetime_to_long(date_arg, freq));
	}
    else
	{
	    // date_arg is neither a string, nor a datetime
	    PyErr_SetString(PyExc_TypeError, "invalid date type");
	    return NULL;
	}

    if (PyErr_Occurred())
	return NULL;

    return result;
}
//==================================================
// Parsing long to datetime/datestring
// =================================================

// Takes a long long value and a frequency
// Returns a datestruct formatted with the correct calendar values
static 
datestruct long_to_datestruct(long long dlong, int frequency)
{
    int year = 1970, month = 1, day = 1, 
	hour = 0, minute = 0, second = 0,
	msecond = 0, usecond = 0, nsecond = 0,
	psecond = 0, fsecond = 0, asecond = 0;

    datestruct result;

    if (frequency == NPY_FR_Y) {
	year = 1970 + dlong;
    } else if (frequency == NPY_FR_M) {
	if (dlong >= 0) {
	    year  = 1970 + dlong / 12;
	    month = dlong % 12 + 1;
	} else {
	    year  = 1969 + (dlong + 1) / 12;
	    month = 12 + (dlong + 1)% 12;
	}
    } else if (frequency == NPY_FR_W) {
	ymdstruct ymd;
	ymd = long_to_ymdstruct((dlong * 7) - 4);
	year  = ymd.year;
	month = ymd.month;
	day   = ymd.day;
    } else if (frequency == NPY_FR_B) {
	ymdstruct ymd;
	long long absdays;
	if (dlong >= 0) {
	    // Special Case
	    if (dlong < 3)
		absdays = dlong + (dlong / 2) * 2;
	    else
		absdays = 7 * ((dlong + 3) / 5) + ((dlong + 3) % 5) - 3;
	} else {
	    // Special Case
	    if (dlong > -7)
		absdays = dlong + (dlong / 4) * 2;
	    else
		absdays = 7 * ((dlong - 1) / 5) + ((dlong - 1) % 5) + 1;
	}
	ymd = long_to_ymdstruct(absdays);
	year  = ymd.year;
	month = ymd.month;
	day   = ymd.day;
    } else if (frequency == NPY_FR_D) {
	ymdstruct ymd = long_to_ymdstruct(dlong);
	year  = ymd.year;
	month = ymd.month;
	day   = ymd.day;	
    } else if (frequency == NPY_FR_h) {
	ymdstruct ymd;	
	if (dlong >= 0) {
	    ymd  = long_to_ymdstruct(dlong / 24);
	    hour  = dlong % 24;
	} else {
	    ymd  = long_to_ymdstruct((dlong - 23) / 24);
	    hour = 24 + (dlong + 1) % 24 - 1;
	}
	year  = ymd.year;
	month = ymd.month;
	day   = ymd.day;
    } else if (frequency == NPY_FR_m) {
	ymdstruct ymd;
	hmsstruct hms;
	if (dlong >= 0) {
	    ymd = long_to_ymdstruct(dlong / 1440);
	} else {
	    ymd = long_to_ymdstruct((dlong - 1439) / 1440);
	}
	hms = long_to_hmsstruct(dlong * 60);
	year   = ymd.year;
	month  = ymd.month;
	day    = ymd.day;
	hour   = hms.hour;
	minute = hms.minute;
    } else if (frequency == NPY_FR_s) {
	ymdstruct ymd;
	hmsstruct hms;
	if (dlong >= 0) {
	    ymd = long_to_ymdstruct(dlong / 86400);
	} else {
	    ymd = long_to_ymdstruct((dlong - 86399) / 86400);
	}
	hms = long_to_hmsstruct(dlong);
	year   = ymd.year;
	month  = ymd.month;
	day    = ymd.day;
	hour   = hms.hour;
	minute = hms.minute;
	second = hms.second;
    } else if (frequency == NPY_FR_ms) {
	ymdstruct ymd;
	hmsstruct hms;
	if (dlong >= 0) {
	    ymd = long_to_ymdstruct(dlong / 86400000LL);
	    hms = long_to_hmsstruct(dlong / 1000);
	    msecond = dlong % 1000;
	} else {
	    ymd = long_to_ymdstruct((dlong - 86399999LL) / 86400000LL);
	    hms = long_to_hmsstruct((dlong - 999LL) / 1000);
	    msecond = (1000 + dlong % 1000) % 1000;
	}
	year    = ymd.year;
	month   = ymd.month;
	day     = ymd.day;
	hour    = hms.hour;
	minute  = hms.minute;
	second  = hms.second;
    } else if (frequency == NPY_FR_us) {
	ymdstruct ymd;
	hmsstruct hms;
	if (dlong >= 0) {
	    ymd = long_to_ymdstruct(dlong / 86400000000LL);
	    hms = long_to_hmsstruct(dlong / 1000000LL);
	    msecond = (dlong / 1000) % 1000;
	    usecond = dlong % 1000LL;
	} else {
	    ymd = long_to_ymdstruct((dlong - 86399999999LL) / 86400000000LL);
	    hms = long_to_hmsstruct((dlong - 999999LL) / 1000000LL);
	    usecond = ((1000000LL + (dlong % 1000000)) % 1000000);
	    msecond = usecond / 1000;
	    usecond = usecond % 1000;
	}
	year    = ymd.year;
	month   = ymd.month;
	day     = ymd.day;
	hour    = hms.hour;
	minute  = hms.minute;
	second  = hms.second;
    }
    // Starting from here, we need extra units (ns, ps, fs, as)
    //  for correct precision: datetime doesn't include beyond microsecond
    else if (frequency == NPY_FR_ns) {
	PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
    } else if (frequency == NPY_FR_ps) {
	PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
    } else if (frequency == NPY_FR_fs) {
	PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
    } else if (frequency == NPY_FR_as) {
	PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
    } else {
	// Throw some Not Valid Frequency error here
    }
	
    result.year    = year;
    result.month   = month;
    result.day     = day;
    result.hour    = hour;
    result.minute  = minute;
    result.second  = second;
    result.msecond = msecond;
    result.usecond = usecond;
    result.nsecond = nsecond;
    result.psecond = psecond;
    result.fsecond = fsecond;
    result.asecond = asecond;

    return result;
}

// Takes a long and a frequency
// Returns a Python DateTime Object
static PyObject *
long_to_datetime(PyObject *self, PyObject *args)
{
    PyObject *long_arg = NULL;    // string or datetime
    PyObject *freq_arg = NULL;	  // frequency as string
    PyObject *result   = NULL;	  // long result

    long long dlong = 0;          // Stores the long_arg
    int freq = NPY_FR_ERR;			  // freq_arg is a PyObject to be parsed to freq
    datestruct dstruct;		      // To store date values

    // macro PyDateTime_IMPORT must be invoked for PyDateTime_Check
    PyDateTime_IMPORT;

    // Make sure the callback function is set
    //  ! This doesn't check to make sure it's the right callback function
    //  ! This should all be done in some init script
    if (!PyCallable_Check(callback))
	{
	    PyErr_SetString(PyExc_TypeError, "callback not set.");
	    return NULL;
	}

    // Parse out long_arg & freq_arg
    if (!PyArg_ParseTuple(args, "OO", &long_arg, &freq_arg))
	{
	    return NULL;
	}

    // Make sure frequency is not NULL
    if (!freq_arg)
	{	
	    PyErr_SetString(PyExc_TypeError, "frequency not set.");
	    return NULL;
	}

    // Parse out frequency into an int so we can use it easily
    if ((freq = _unit_from_str(PyString_AsString(freq_arg))) == NPY_FR_ERR)
	{
	    // If the frequency is invalid, set an error and return null
	    PyErr_SetString(PyExc_TypeError, "invalid frequency.");
	    return NULL;
	}
    // Make sure long_arg is not NULL
    if (!long_arg)
	{
	    PyErr_SetString(PyExc_TypeError, "no date provided.");
	    return NULL;
	}
    // Be sure long_arg is a long
    if (PyLong_Check(long_arg))
	{

	    // XXX PyINCREF here?
	    // Convert long_arg to a long long
	    dlong = PyLong_AsLongLong(long_arg);
	    // Format the dstruct to create the datetime object
	    dstruct = long_to_datestruct(dlong, freq);
	    // Create the PyDateTime Object as result
	    result = PyDateTime_FromDateAndTime(dstruct.year, dstruct.month,
						dstruct.day, dstruct.hour, dstruct.minute, dstruct.second,
						dstruct.msecond * 1000 + dstruct.usecond);
	}
    else
	{
	    // long_arg is not a long; error
	    PyErr_SetString(PyExc_TypeError, "invalid date type");
	    return NULL;
	}

    if (PyErr_Occurred())
	return NULL;

    return result;
}

// Takes a long and a frequency
// Returns a string formatted to represent the date from the long
static PyObject *
long_to_datestring(PyObject *self, PyObject *args)
{
    PyObject *long_arg = NULL;    // string or datetime
    PyObject *freq_arg = NULL;	  // frequency as string
    PyObject *result   = NULL;	  // string result

    long long dlong = 0;          // Stores the long_arg
    int freq = NPY_FR_ERR;			  // freq_arg is a PyObject to be parsed to freq
    datestruct dstruct;		      // To store date values

    // macro PyDateTime_IMPORT must be invoked for PyDateTime_Check
    PyDateTime_IMPORT;

    // Make sure the callback function is set
    //  ! This doesn't check to make sure it's the right callback function
    //  ! This should all be done in some init script
    if (!PyCallable_Check(callback))
	{
	    PyErr_SetString(PyExc_TypeError, "callback not set.");
	    return NULL;
	}

    // Parse out long_arg & freq_arg
    if (!PyArg_ParseTuple(args, "OO", &long_arg, &freq_arg))
	{
	    return NULL;
	}

    // Make sure frequency is not NULL
    if (!freq_arg)
	{	
	    PyErr_SetString(PyExc_TypeError, "frequency not set.");
	    return NULL;
	}

    // Parse out frequency into an int so we can use it easily
    if ((freq = _unit_from_str(PyString_AsString(freq_arg))) == NPY_FR_ERR)
	{
	    // If the frequency is invalid, set an error and return null
	    PyErr_SetString(PyExc_TypeError, "invalid frequency.");
	    return NULL;
	}
    // Make sure long_arg is not NULL
    if (!long_arg)
	{
	    PyErr_SetString(PyExc_TypeError, "no date provided.");
	    return NULL;
	}
    // Be sure long_arg is a long
    if (PyLong_Check(long_arg))
	{

	    // XXX PyINCREF here?
	    // Convert long_arg to a long long
	    dlong = PyLong_AsLongLong(long_arg);
	    // Format the dstruct to create the datetime object
	    dstruct = long_to_datestruct(dlong, freq);
	    // Make sure date is less than 4 digits
	    if (dstruct.year > 9999)
		{
		    PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
		    return NULL;
		}
	    // Create the Python String formatted according frequency
	    if ((freq == NPY_FR_Y) || (freq == (NPY_FR_M) ||
				       (freq == NPY_FR_W) || (freq == NPY_FR_B) || freq == (NPY_FR_D))) {
		// Good. PyString_FromFormat won't let me do simple printf stuff
		// like "%04d-%02d-%02d" for simple date formatting.
		// Now I have to write this stuff from scratch...
		
		char year[4];
		char month[2];
		char day[2];
				
		sprintf(year,  "%04d", dstruct.year);
		sprintf(month, "%02d", dstruct.month);
		sprintf(day,   "%02d", dstruct.day);

		// Now form the result with our char*
		result = PyString_FromFormat("%s-%s-%s", year, month, day);
	    } else if ((freq == NPY_FR_h) || (freq == NPY_FR_m) || freq == (NPY_FR_s)) {
		char year[4];
		char month[2];
		char day[2];
		char hour[2];
		char minute[2];
		char second[2];
				
		sprintf(year,   "%04d", dstruct.year);
		sprintf(month,  "%02d", dstruct.month);
		sprintf(day,    "%02d", dstruct.day);
		sprintf(hour,   "%02d", dstruct.hour);
		sprintf(minute, "%02d", dstruct.minute);
		sprintf(second, "%02d", dstruct.second);

		result = PyString_FromFormat("%s-%s-%s %s:%s:%s", year,
					     month, day, hour, minute, second);	
	    } else if ((freq == NPY_FR_ms) || (freq == NPY_FR_us)) {
		char year[4];
		char month[2];
		char day[2];
		char hour[2];
		char minute[2];
		char second[2];
		char msecond[3];
		char usecond[3];
				
		sprintf(year,   "%04d", dstruct.year);
		sprintf(month,  "%02d", dstruct.month);
		sprintf(day,    "%02d", dstruct.day);
		sprintf(hour,   "%02d", dstruct.hour);
		sprintf(minute, "%02d", dstruct.minute);
		sprintf(second, "%02d", dstruct.second);
		sprintf(msecond, "%03d", dstruct.msecond);
		sprintf(usecond, "%03d", dstruct.usecond);
		if ((dstruct.usecond) || (dstruct.msecond))
		    {
			result = PyString_FromFormat("%s-%s-%s %s:%s:%s.%s%s", 
						     year, month, day, 
						     hour, minute, second, 
						     msecond, usecond);
			//result = PyString_FromFormat("ms: %s -- us: %s", msecond, usecond);
		    }
		else 
		    {
			result = PyString_FromFormat("%s-%s-%s %s:%s:%s", 
						     year, month, day, 
						     hour, minute, second);
		    }
	    }
	}
    else
	{
	    // long_arg is not a long; error
	    PyErr_SetString(PyExc_TypeError, "invalid date type");
	    return NULL;
	}

    if (PyErr_Occurred())
	return NULL;

    return result;
}

//==================================================
// Frequency Conversions
//==================================================

// Taken from TimeSeries //
// helpers for frequency conversion routines
/*
  static long DtoB_weekday(long fromDate) { return (((fromDate) / 7) * 5) + (fromDate)%7; }

  static long DtoB_WeekendToMonday(long absdate, int day_of_week) {

  if (day_of_week > 4) {
  //change to Monday after weekend
  absdate += (7 - day_of_week);
  }
  return DtoB_weekday(absdate);
  }

  static long DtoB_WeekendToFriday(long absdate, int day_of_week) {

  if (day_of_week > 4) {
  //change to friday before weekend
  absdate -= (day_of_week - 4);
  }
  return DtoB_weekday(absdate);
  }
*/


// Taken from TimeSeries //
// conversion routines for frequencies


// *************** From Day *************** //
static long long as_freq_D2Y(long long dlong)
{
    ymdstruct ymd = long_to_ymdstruct(dlong);
    return ymd.year - 1970;
}
static long long as_freq_D2M(long long dlong)
{
    ymdstruct ymd = long_to_ymdstruct(dlong);
    return ymd.month + (ymd.year - 1970) * 12 - 1;
}
static long long as_freq_D2W(long long dlong)
{
    // convert to the previous Sunday
    int dotw = day_of_week(dlong);
    if (dlong < 0)
	return (dlong - dotw) / 7;
    else
	return (dlong + 4) / 7;
}
static long long as_freq_D2B(long long dlong)
{
    int dotw = day_of_week(dlong);
    // Pre epoch
    if (dlong < 0)
	{
	    // To get beyond Sunday, Dec 28, 1969
	    if (dlong < -4) {
		// Offset by 1 for Sundays
		if (dotw)
		    return ((dlong + 7 - dotw) / 7) * 5 - (6 - dotw) - 3;
		else
		    return ((dlong + 7 - dotw) / 7) * 5 - (6 - dotw) - 2;
	    } else {
		// Offset by 1 for Sundays
		if (dotw)
		    return -4 + dotw;
		else
		    return -3; // Sunday, Dec 28, 1969
	    }
	    // Post epoch
	} else {
	// To get to Sunday, Jan 4, 1970
	// number of weeks * 5 + dotw [0-6] - Saturdays + 1 for offset
	if (dlong > 2)
	    return ((dlong - dotw) / 7) * 5 + dotw - (dotw / 6) + 1;
	else 
	    return dotw - 4 - (dotw / 6);
    }
}
static long long as_freq_D2h(long long dlong)
{
    return dlong * 24LL;
}
static long long as_freq_D2m(long long dlong)
{
    return dlong * 1440LL;
}
static long long as_freq_D2s(long long dlong)
{
    return dlong * 86400LL;
}
static long long as_freq_D2ms(long long dlong)
{
    return dlong * 86400000LL;
}
static long long as_freq_D2us(long long dlong)
{
    return dlong * 86400000000LL;
}
static long long as_freq_D2ns(long long dlong)
{
    return dlong * 86400000000000LL;
}
static long long as_freq_D2ps(long long dlong)
{
    return dlong * 86400000000000000LL;
}
static long long as_freq_D2fs(long long dlong)
{
    return 0;
    // should throw an error...
    //return dlong * 86400000000000000000LL;
}
static long long as_freq_D2as(long long dlong)
{
    return 0;
    // should throw an error...
    //return dlong * 86400000000000000000000LL;
}

// *************** From Year *************** //
static long long as_freq_Y2D(long long dlong)
{
    long long absdays = absdays_from_ymd(1970 + dlong, 1, 1);
    return absdays;
}
static long long as_freq_Y2M(long long dlong)
{
    return dlong * 12;
}
static long long as_freq_Y2W(long long dlong)
{
    return as_freq_D2W(as_freq_Y2D(dlong));
}
static long long as_freq_Y2B(long long dlong)
{
    return as_freq_D2B(as_freq_Y2D(dlong));
}
static long long as_freq_Y2h(long long dlong)
{
    return as_freq_Y2D(dlong) * 24;
}
static long long as_freq_Y2m(long long dlong)
{
    return as_freq_Y2D(dlong) * 1440;
}
static long long as_freq_Y2s(long long dlong)
{
    return as_freq_Y2D(dlong) * 86400;
}
static long long as_freq_Y2ms(long long dlong)
{
    return as_freq_Y2D(dlong) * 86400000LL;
}
static long long as_freq_Y2us(long long dlong)
{
    return as_freq_Y2D(dlong) * 86400000000LL;
}
static long long as_freq_Y2ns(long long dlong)
{
    return as_freq_Y2D(dlong) * 86400000000000LL;
}
static long long as_freq_Y2ps(long long dlong)
{
    return as_freq_Y2D(dlong) * 86400000000000000LL;
}
static long long as_freq_Y2fs(long long dlong)
{
    return 0;
    // should return an error
    // return as_freq_Y2D(dlong) * 86400000000000000000LL;
}
static long long as_freq_Y2as(long long dlong)
{
    return 0;
    // should return an error
    // return as_freq_Y2D(dlong) * 86400000000000000000000LL;
}

// *************** From Month *************** //
// Taken from TimeSeries
static long long as_freq_M2D(long long dlong)
{
    long long absdays;
    long y;
    long m;

    if (dlong < 0) {
	y = (dlong + 1) / 12 - 1;
	m = 12 + (dlong + 1) % 12;
	if (!m) {   m = 12;   }
	absdays = absdays_from_ymd(1970 + y, m, 1);
	return absdays;
    } else {
	y = (dlong) / 12;
	m =  dlong % 12 + 1;
	absdays = absdays_from_ymd(1970 + y, m, 1);
	return absdays;
    }
}
static long long as_freq_M2Y(long long dlong)
{
    if (dlong < 0)
	return (dlong + 1) / 12 - 1;
    return dlong / 12;
}
static long long as_freq_M2W(long long dlong)
{
    return as_freq_D2W(as_freq_M2D(dlong));
}
static long long as_freq_M2B(long long dlong)
{
    return as_freq_D2B(as_freq_M2D(dlong));
}
static long long as_freq_M2h(long long dlong)
{
    return as_freq_D2h(as_freq_M2D(dlong));
}
static long long as_freq_M2m(long long dlong)
{
    return as_freq_D2m(as_freq_M2D(dlong));
}
static long long as_freq_M2s(long long dlong)
{
    return as_freq_D2s(as_freq_M2D(dlong));
}
static long long as_freq_M2ms(long long dlong)
{
    return as_freq_D2ms(as_freq_M2D(dlong));
}
static long long as_freq_M2us(long long dlong)
{
    return as_freq_D2us(as_freq_M2D(dlong));
}
static long long as_freq_M2ns(long long dlong)
{
    return as_freq_D2ns(as_freq_M2D(dlong));
}
static long long as_freq_M2ps(long long dlong)
{
    return as_freq_D2ps(as_freq_M2D(dlong));
}
static long long as_freq_M2fs(long long dlong)
{
    return as_freq_D2fs(as_freq_M2D(dlong));
}
static long long as_freq_M2as(long long dlong)
{
    return as_freq_D2as(as_freq_M2D(dlong));
}

// *************** From Week *************** //
static long long as_freq_W2D(long long dlong)
{
    return (dlong * 7) - 4;
}
static long long as_freq_W2Y(long long dlong)
{
    return as_freq_D2Y(as_freq_W2D(dlong));
}
static long long as_freq_W2M(long long dlong)
{
    return as_freq_D2M(as_freq_W2D(dlong));
}
static long long as_freq_W2B(long long dlong)
{
    return as_freq_D2B(as_freq_W2D(dlong));
}
static long long as_freq_W2h(long long dlong)
{
    return as_freq_D2h(as_freq_W2D(dlong));
}
static long long as_freq_W2m(long long dlong)
{
    return as_freq_D2m(as_freq_W2D(dlong));
}
static long long as_freq_W2s(long long dlong)
{
    return as_freq_D2s(as_freq_W2D(dlong));
}
static long long as_freq_W2ms(long long dlong)
{
    return as_freq_D2ms(as_freq_W2D(dlong));
}
static long long as_freq_W2us(long long dlong)
{
    return as_freq_D2us(as_freq_W2D(dlong));
}
static long long as_freq_W2ns(long long dlong)
{
    return as_freq_D2ns(as_freq_W2D(dlong));
}
static long long as_freq_W2ps(long long dlong)
{
    return as_freq_D2ps(as_freq_W2D(dlong));
}
static long long as_freq_W2fs(long long dlong)
{
    return as_freq_D2fs(as_freq_W2D(dlong));
}
static long long as_freq_W2as(long long dlong)
{
    return as_freq_D2as(as_freq_W2D(dlong));
}

// *************** From Business Day *************** //
static long long as_freq_B2D(long long dlong)
{
    if (dlong < 0) {
	// Special Case
	if (dlong > -7)
	    return dlong + (dlong / 4) * 2;
	else
	    return 7 * ((dlong - 1) / 5) + ((dlong - 1) % 5) + 1;
    } else {
	// Special Case
	if (dlong < 3)
	    return dlong + (dlong / 2) * 2;
	else
	    return 7 * ((dlong + 3) / 5) + ((dlong + 3) % 5) - 3;
    }
}
static long long as_freq_B2Y(long long dlong)
{
    return as_freq_D2Y(as_freq_B2D(dlong));
}
static long long as_freq_B2M(long long dlong)
{
    return as_freq_D2M(as_freq_B2D(dlong));
}
static long long as_freq_B2W(long long dlong)
{
    return as_freq_D2W(as_freq_B2D(dlong));
}
static long long as_freq_B2h(long long dlong)
{
    return as_freq_D2h(as_freq_B2D(dlong));
}
static long long as_freq_B2m(long long dlong)
{
    return as_freq_D2m(as_freq_B2D(dlong));
}
static long long as_freq_B2s(long long dlong)
{
    return as_freq_D2s(as_freq_B2D(dlong));
}
static long long as_freq_B2ms(long long dlong)
{
    return as_freq_D2ms(as_freq_B2D(dlong));
}
static long long as_freq_B2us(long long dlong)
{
    return as_freq_D2us(as_freq_B2D(dlong));
}
static long long as_freq_B2ns(long long dlong)
{
    return as_freq_D2ns(as_freq_B2D(dlong));
}
static long long as_freq_B2ps(long long dlong)
{
    return as_freq_D2ps(as_freq_B2D(dlong));
}
static long long as_freq_B2fs(long long dlong)
{
    return as_freq_D2fs(as_freq_B2D(dlong));
}
static long long as_freq_B2as(long long dlong)
{
    return as_freq_D2as(as_freq_B2D(dlong));
}

// *************** From Hour *************** //
static long long as_freq_h2D(long long dlong)
{
    if (dlong < 0)
	return dlong / 24 - 1;
    return dlong / 24;
}
static long long as_freq_h2Y(long long dlong)
{
    return as_freq_D2Y(as_freq_h2D(dlong));
}
static long long as_freq_h2M(long long dlong)
{
    return as_freq_D2M(as_freq_h2D(dlong));
}
static long long as_freq_h2W(long long dlong)
{
    return as_freq_D2W(as_freq_h2D(dlong));
}
static long long as_freq_h2B(long long dlong)
{
    return as_freq_D2B(as_freq_h2D(dlong));
}
// these are easier to think about with a simple calculation
static long long as_freq_h2m(long long dlong)
{
    return dlong * 60;
}
static long long as_freq_h2s(long long dlong)
{
    return dlong * 3600;
}
static long long as_freq_h2ms(long long dlong)
{
    return dlong * 3600000;
}
static long long as_freq_h2us(long long dlong)
{
    return dlong * 3600000000LL;
}
static long long as_freq_h2ns(long long dlong)
{
    return dlong * 3600000000000LL;
}
static long long as_freq_h2ps(long long dlong)
{
    return dlong * 3600000000000000LL;
}
static long long as_freq_h2fs(long long dlong)
{
    return dlong * 3600000000000000000LL;
}
static long long as_freq_h2as(long long dlong)
{
    return 0;
    // should return an error...	
    //return dlong * 3600000000000000000000LL;
}

// *************** From Minute *************** //
static long long as_freq_m2D(long long dlong)
{
    if (dlong < 0)
	return dlong / 1440 - 1;
    return dlong / 1440;
}
static long long as_freq_m2Y(long long dlong)
{
    return as_freq_D2Y(as_freq_m2D(dlong));
}
static long long as_freq_m2M(long long dlong)
{
    return as_freq_D2M(as_freq_m2D(dlong));
}
static long long as_freq_m2W(long long dlong)
{
    return as_freq_D2W(as_freq_m2D(dlong));
}
static long long as_freq_m2B(long long dlong)
{
    return as_freq_D2B(as_freq_m2D(dlong));
}
// these are easier to think about with a simple calculation
static long long as_freq_m2h(long long dlong)
{
    if (dlong < 0)
	return dlong / 60 - 1;
    return dlong / 60;
}
static long long as_freq_m2s(long long dlong)
{
    return dlong * 60;
}
static long long as_freq_m2ms(long long dlong)
{
    return dlong * 60000;
}
static long long as_freq_m2us(long long dlong)
{
    return dlong * 60000000LL;
}
static long long as_freq_m2ns(long long dlong)
{
    return dlong * 60000000000LL;
}
static long long as_freq_m2ps(long long dlong)
{
    return dlong * 60000000000000LL;
}
static long long as_freq_m2fs(long long dlong)
{
    return dlong * 60000000000000000LL;
}
static long long as_freq_m2as(long long dlong)
{
    return 0;
    // should return an error...
    //return dlong * 60000000000000000000LL;
}

// *************** From Second *************** //
static long long as_freq_s2D(long long dlong)
{
    if (dlong < 0)
	return dlong / 86400 - 1;
    return dlong / 86400;
}
static long long as_freq_s2Y(long long dlong)
{
    return as_freq_D2Y(as_freq_s2D(dlong));
}
static long long as_freq_s2M(long long dlong)
{
    return as_freq_D2M(as_freq_s2D(dlong));
}
static long long as_freq_s2W(long long dlong)
{
    return as_freq_D2W(as_freq_s2D(dlong));
}
static long long as_freq_s2B(long long dlong)
{
    return as_freq_D2B(as_freq_s2D(dlong));
}
// these are easier to think about with a simple calculation
static long long as_freq_s2h(long long dlong)
{
    if (dlong < 0)
	return dlong / 3600 - 1;
    return dlong / 3600;
}
static long long as_freq_s2m(long long dlong)
{
    if (dlong < 0)
	return dlong / 60 - 1;
    return dlong / 60;
}
static long long as_freq_s2ms(long long dlong)
{
    return dlong * 1000;
}
static long long as_freq_s2us(long long dlong)
{
    return dlong * 1000000;
}
static long long as_freq_s2ns(long long dlong)
{
    return dlong * 1000000000LL;
}
static long long as_freq_s2ps(long long dlong)
{
    return dlong * 1000000000000LL;
}
static long long as_freq_s2fs(long long dlong)
{
    return dlong * 1000000000000000LL;
}
static long long as_freq_s2as(long long dlong)
{
    return dlong * 1000000000000000000LL;
}

// *************** From Millisecond *************** //
static long long as_freq_ms2D(long long dlong)
{
    if (dlong < 0)
	return dlong / 86400000 - 1;
    return dlong / 86400000;
}
static long long as_freq_ms2Y(long long dlong)
{
    return as_freq_D2Y(as_freq_ms2D(dlong));
}
static long long as_freq_ms2M(long long dlong)
{
    return as_freq_D2M(as_freq_ms2D(dlong));
}
static long long as_freq_ms2W(long long dlong)
{
    return as_freq_D2W(as_freq_ms2D(dlong));
}
static long long as_freq_ms2B(long long dlong)
{
    return as_freq_D2B(as_freq_ms2D(dlong));
}
// these are easier to think about with a simple calculation
static long long as_freq_ms2h(long long dlong)
{
    if (dlong < 0)
	return dlong / 3600000 - 1;
    return dlong / 3600000;
}
static long long as_freq_ms2m(long long dlong)
{
    if (dlong < 0)
	return dlong / 60000 - 1;
    return dlong / 60000;
}
static long long as_freq_ms2s(long long dlong)
{
    if (dlong < 0)
	return dlong / 1000 - 1;
    return dlong / 1000;
}
static long long as_freq_ms2us(long long dlong)
{
    return dlong * 1000;
}
static long long as_freq_ms2ns(long long dlong)
{
    return dlong * 1000000;
}
static long long as_freq_ms2ps(long long dlong)
{
    return dlong * 1000000000LL;
}
static long long as_freq_ms2fs(long long dlong)
{
    return dlong * 1000000000000LL;
}
static long long as_freq_ms2as(long long dlong)
{
    return dlong * 1000000000000000LL;
}

// *************** From Microsecond *************** //
static long long as_freq_us2D(long long dlong)
{
    if (dlong < 0)
	return dlong / 86400000000LL - 1;
    return dlong / 86400000000LL;
}
static long long as_freq_us2Y(long long dlong)
{
    return as_freq_D2Y(as_freq_us2D(dlong));
}
static long long as_freq_us2M(long long dlong)
{
    return as_freq_D2M(as_freq_us2D(dlong));
}
static long long as_freq_us2W(long long dlong)
{
    return as_freq_D2W(as_freq_us2D(dlong));
}
static long long as_freq_us2B(long long dlong)
{
    return as_freq_D2B(as_freq_us2D(dlong));
}
// these are easier to think about with a simple calculation
static long long as_freq_us2h(long long dlong)
{
    if (dlong < 0)
	return dlong / 3600000000LL - 1;
    return dlong / 3600000000LL;
}
static long long as_freq_us2m(long long dlong)
{
    if (dlong < 0)
	return dlong / 60000000LL - 1;
    return dlong / 60000000LL;
}
static long long as_freq_us2s(long long dlong)
{
    if (dlong < 0)
	return dlong / 1000000LL - 1;
    return dlong / 1000000LL;
}
static long long as_freq_us2ms(long long dlong)
{
    // We're losing precision on XX:XX:XX.xx1 times for some reason
    //  can't find a fix, so here's a cheap hack...
    if ((dlong < 0) && ((dlong % 10000) != -9000))
	return dlong / 1000LL - 1;
    return dlong / 1000LL;
}
static long long as_freq_us2ns(long long dlong)
{
    return dlong * 1000000LL;
}
static long long as_freq_us2ps(long long dlong)
{
    return dlong * 1000000000LL;
}
static long long as_freq_us2fs(long long dlong)
{
    return dlong * 1000000000000LL;
}
static long long as_freq_us2as(long long dlong)
{
    return dlong * 1000000000000000LL;
}


// ******* THESE ARE UNSUPPORTED CURRENTLY ******* // 
// *********************************************** //
// *************** From Nanosecond *************** //
// *********************************************** //
static long long as_freq_ns2D(long long dlong)
{
    return as_freq_s2D(dlong) * 1000000000LL;
}
static long long as_freq_ns2Y(long long dlong)
{
    return as_freq_D2Y(as_freq_ns2D(dlong));
}
static long long as_freq_ns2M(long long dlong)
{
    return as_freq_D2M(as_freq_ns2D(dlong));
}
static long long as_freq_ns2W(long long dlong)
{
    return as_freq_D2W(as_freq_ns2D(dlong));
}
static long long as_freq_ns2B(long long dlong)
{
    return as_freq_D2B(as_freq_ns2D(dlong));
}
// these are easier to think about with a simple calculation
static long long as_freq_ns2h(long long dlong)
{
    return as_freq_D2h(as_freq_ns2D(dlong));
}
static long long as_freq_ns2m(long long dlong)
{
    return as_freq_D2m(as_freq_ns2D(dlong));
}
static long long as_freq_ns2s(long long dlong)
{
    return as_freq_D2s(as_freq_ns2D(dlong));
}
static long long as_freq_ns2ms(long long dlong)
{
    return as_freq_D2ms(as_freq_ns2D(dlong));
}
static long long as_freq_ns2us(long long dlong)
{
    return as_freq_D2us(as_freq_ns2D(dlong));
}
static long long as_freq_ns2ps(long long dlong)
{
    return as_freq_D2ps(as_freq_ns2D(dlong));
}
static long long as_freq_ns2fs(long long dlong)
{
    return as_freq_D2fs(as_freq_ns2D(dlong));
}
static long long as_freq_ns2as(long long dlong)
{
    return as_freq_D2as(as_freq_ns2D(dlong));
}

// *************** From Picosecond *************** //
static long long as_freq_ps2Y(long long dlong)
{
    return dlong;
}
static long long as_freq_ps2M(long long dlong)
{
    return dlong;
}
static long long as_freq_ps2W(long long dlong)
{
    return dlong;
}
static long long as_freq_ps2B(long long dlong)
{
    return dlong;
}
// these are easier to think about with a simple calculation
static long long as_freq_ps2h(long long dlong)
{
    return dlong;
}
static long long as_freq_ps2m(long long dlong)
{
    return dlong;
}
static long long as_freq_ps2s(long long dlong)
{
    return dlong;
}
static long long as_freq_ps2ms(long long dlong)
{
    return dlong;
}
static long long as_freq_ps2us(long long dlong)
{
    return dlong;
}
static long long as_freq_ps2ns(long long dlong)
{
    return dlong;
}
static long long as_freq_ps2D(long long dlong)
{
    return dlong;
}
static long long as_freq_ps2fs(long long dlong)
{
    return dlong;
}
static long long as_freq_ps2as(long long dlong)
{
    return dlong;
}

// *************** From Femtosecond *************** //
static long long as_freq_fs2Y(long long dlong)
{
    return dlong;
}
static long long as_freq_fs2M(long long dlong)
{
    return dlong;
}
static long long as_freq_fs2W(long long dlong)
{
    return dlong;
}
static long long as_freq_fs2B(long long dlong)
{
    return dlong;
}
// these are easier to think about with a simple calculation
static long long as_freq_fs2h(long long dlong)
{
    return dlong;
}
static long long as_freq_fs2m(long long dlong)
{
    return dlong;
}
static long long as_freq_fs2s(long long dlong)
{
    return dlong;
}
static long long as_freq_fs2ms(long long dlong)
{
    return dlong;
}
static long long as_freq_fs2us(long long dlong)
{
    return dlong;
}
static long long as_freq_fs2ns(long long dlong)
{
    return dlong;
}
static long long as_freq_fs2ps(long long dlong)
{
    return dlong;
}
static long long as_freq_fs2D(long long dlong)
{
    return dlong;
}
static long long as_freq_fs2as(long long dlong)
{
    return dlong;
}

// *************** From Attosecond *************** //
static long long as_freq_as2Y(long long dlong)
{
    return dlong;
}
static long long as_freq_as2M(long long dlong)
{
    return dlong;
}
static long long as_freq_as2W(long long dlong)
{
    return dlong;
}
static long long as_freq_as2B(long long dlong)
{
    return dlong;
}
// these are easier to think about with a simple calculation
static long long as_freq_as2h(long long dlong)
{
    return dlong;
}
static long long as_freq_as2m(long long dlong)
{
    return dlong;
}
static long long as_freq_as2s(long long dlong)
{
    return dlong;
}
static long long as_freq_as2ms(long long dlong)
{
    return dlong;
}
static long long as_freq_as2us(long long dlong)
{
    return dlong;
}
static long long as_freq_as2ns(long long dlong)
{
    return dlong;
}
static long long as_freq_as2ps(long long dlong)
{
    return dlong;
}
static long long as_freq_as2fs(long long dlong)
{
    return dlong;
}
static long long as_freq_as2D(long long dlong)
{
    return dlong;
}

static long long NO_FUNC(long long empty)
{ return empty; }

// Convert (dlong, ifreq) to a new date based on ofreq
// Returns the long value to represent the date with the ofreq
static long long (*get_conversion_ftn(int ifreq, int ofreq)) (long long)
{
    if (ifreq == ofreq)
	return &NO_FUNC;// Error out

    // Switch to decide which routine to run
    switch (ifreq) 
	{
	case NPY_FR_Y:
	    switch (ofreq) 
		{
		case NPY_FR_M: return &as_freq_Y2M; break;
		case NPY_FR_W: return &as_freq_Y2W; break;
		case NPY_FR_B: return &as_freq_Y2B; break;
		case NPY_FR_D: return &as_freq_Y2D; break;
		case NPY_FR_h: return &as_freq_Y2h; break;
		case NPY_FR_m: return &as_freq_Y2m; break;
		case NPY_FR_s: return &as_freq_Y2s; break;
		case NPY_FR_ms:return &as_freq_Y2ms; break;
		case NPY_FR_us:return &as_freq_Y2us; break;
		case NPY_FR_ns:return &as_freq_Y2ns; break;
		case NPY_FR_ps:return &as_freq_Y2ps; break;
		case NPY_FR_fs:return &as_freq_Y2fs; break;
		case NPY_FR_as:return &as_freq_Y2as; break;
		}
	    break;
	case NPY_FR_M:
	    switch (ofreq) {
	    case NPY_FR_Y: return &as_freq_M2Y; break;
	    case NPY_FR_W: return &as_freq_M2W; break;
	    case NPY_FR_B: return &as_freq_M2B; break;
	    case NPY_FR_D: return &as_freq_M2D; break;
	    case NPY_FR_h: return &as_freq_M2h; break; 
	    case NPY_FR_m: return &as_freq_M2m; break; 
	    case NPY_FR_s: return &as_freq_M2s; break;
	    case NPY_FR_ms: return &as_freq_M2ms; break;
	    case NPY_FR_us: return &as_freq_M2us; break;
	    case NPY_FR_ns: return &as_freq_M2ns; break;
	    case NPY_FR_ps: return &as_freq_M2ps; break;
	    case NPY_FR_fs: return &as_freq_M2fs; break;
	    case NPY_FR_as: return &as_freq_M2as; break;
	    }
	    break;
	case NPY_FR_W:
	    switch (ofreq) {
	    case NPY_FR_Y: return &as_freq_W2Y; break;
	    case NPY_FR_M: return &as_freq_W2M; break;
	    case NPY_FR_B: return &as_freq_W2B; break;
	    case NPY_FR_D: return &as_freq_W2D; break;
	    case NPY_FR_h: return &as_freq_W2h; break;
	    case NPY_FR_m: return &as_freq_W2m; break;
	    case NPY_FR_s: return &as_freq_W2s; break;
	    case NPY_FR_ms: return &as_freq_W2ms; break;
	    case NPY_FR_us: return &as_freq_W2us; break;
	    case NPY_FR_ns: return &as_freq_W2ns; break;
	    case NPY_FR_ps: return &as_freq_W2ps; break;
	    case NPY_FR_fs: return &as_freq_W2fs; break;
	    case NPY_FR_as: return &as_freq_W2as; break;
	    }
	    break;
	case NPY_FR_B:
	    switch (ofreq) {
	    case NPY_FR_Y: return &as_freq_B2Y; break;
	    case NPY_FR_M: return &as_freq_B2M; break;  
	    case NPY_FR_W: return &as_freq_B2W; break;
	    case NPY_FR_D: return &as_freq_B2D; break;
	    case NPY_FR_h: return &as_freq_B2h; break;
	    case NPY_FR_m: return &as_freq_B2m; break;
	    case NPY_FR_s: return &as_freq_B2s; break;
	    case NPY_FR_ms: return &as_freq_B2ms; break;
	    case NPY_FR_us: return &as_freq_B2us; break;
	    case NPY_FR_ns: return &as_freq_B2ns; break;
	    case NPY_FR_ps: return &as_freq_B2ps; break;
	    case NPY_FR_fs: return &as_freq_B2fs; break;
	    case NPY_FR_as: return &as_freq_B2as; break;
	    }
	    break;
	case NPY_FR_D:
	    switch (ofreq) {
	    case NPY_FR_Y: return &as_freq_D2Y; break;
	    case NPY_FR_M: return &as_freq_D2M; break;
	    case NPY_FR_W: return &as_freq_D2W; break;
	    case NPY_FR_B: return &as_freq_D2B; break;
	    case NPY_FR_h: return &as_freq_D2h; break;
	    case NPY_FR_m: return &as_freq_D2m; break;
	    case NPY_FR_s: return &as_freq_D2s; break;
	    case NPY_FR_ms: return &as_freq_D2ms; break;
	    case NPY_FR_us: return &as_freq_D2us; break;
	    case NPY_FR_ns: return &as_freq_D2ns; break;
	    case NPY_FR_ps: return &as_freq_D2ps; break;
	    case NPY_FR_fs: return &as_freq_D2fs; break;
	    case NPY_FR_as: return &as_freq_D2as; break;
	    }
	    break;
	case NPY_FR_h:
	    switch (ofreq) {
	    case NPY_FR_Y: return &as_freq_h2Y; break;
	    case NPY_FR_M: return &as_freq_h2M; break;
	    case NPY_FR_W: return &as_freq_h2W; break;
	    case NPY_FR_B: return &as_freq_h2B; break;
	    case NPY_FR_D: return &as_freq_h2D; break;
	    case NPY_FR_m: return &as_freq_h2m; break;
	    case NPY_FR_s: return &as_freq_h2s; break;
	    case NPY_FR_ms: return &as_freq_h2ms; break;
	    case NPY_FR_us: return &as_freq_h2us; break;
	    case NPY_FR_ns: return &as_freq_h2ns; break;
	    case NPY_FR_ps: return &as_freq_h2ps; break;
	    case NPY_FR_fs: return &as_freq_h2fs; break;
	    case NPY_FR_as: return &as_freq_h2as; break;
	    }
	    break;
	case NPY_FR_m:
	    switch (ofreq) {
	    case NPY_FR_Y: return &as_freq_m2Y; break;
	    case NPY_FR_M: return &as_freq_m2M;
	    case NPY_FR_W: return &as_freq_m2W; break;
	    case NPY_FR_B: return &as_freq_m2B; break;
	    case NPY_FR_D: return &as_freq_m2D; break;
	    case NPY_FR_h: return &as_freq_m2h; break;
	    case NPY_FR_s: return &as_freq_m2s; break;
	    case NPY_FR_us: return &as_freq_m2us; break;
	    case NPY_FR_ms: return &as_freq_m2ms; break;
	    case NPY_FR_ns: return &as_freq_m2ns; break;
	    case NPY_FR_ps: return &as_freq_m2ps; break;
	    case NPY_FR_fs: return &as_freq_m2fs; break;
	    case NPY_FR_as: return &as_freq_m2as; break;
	    }
	    break;
	case NPY_FR_s:
	    switch (ofreq) {
	    case NPY_FR_Y: return &as_freq_s2Y; break;
	    case NPY_FR_M: return &as_freq_s2M; break;
	    case NPY_FR_W: return &as_freq_s2W; break;
	    case NPY_FR_B: return &as_freq_s2B; break;
	    case NPY_FR_D: return &as_freq_s2D; break;
	    case NPY_FR_h: return &as_freq_s2h; break;
	    case NPY_FR_m: return &as_freq_s2m; break;
	    case NPY_FR_ms: return &as_freq_s2ms; break;
	    case NPY_FR_us: return &as_freq_s2us; break;
	    case NPY_FR_ns: return &as_freq_s2ns; break;
	    case NPY_FR_ps: return &as_freq_s2ps; break;
	    case NPY_FR_fs: return &as_freq_s2fs; break;
	    case NPY_FR_as: return &as_freq_s2as; break;
	    }
	    break;
	case NPY_FR_ms:
	    switch (ofreq) {
	    case NPY_FR_Y: return &as_freq_ms2Y; break;
	    case NPY_FR_M: return &as_freq_ms2M; break;
	    case NPY_FR_W: return &as_freq_ms2W; break;
	    case NPY_FR_B: return &as_freq_ms2B; break;
	    case NPY_FR_D: return &as_freq_ms2D; break;
	    case NPY_FR_h: return &as_freq_ms2h; break;
	    case NPY_FR_m: return &as_freq_ms2m; break;
	    case NPY_FR_s: return &as_freq_ms2s; break;
	    case NPY_FR_us: return &as_freq_ms2us; break;
	    case NPY_FR_ns: return &as_freq_ms2ns; break;
	    case NPY_FR_ps: return &as_freq_ms2ps; break;
	    case NPY_FR_fs: return &as_freq_ms2fs; break;
	    case NPY_FR_as: return &as_freq_ms2as; break;
	    }
	    break;
	case NPY_FR_us:
	    switch (ofreq) {
	    case NPY_FR_Y: return &as_freq_us2Y; break;
	    case NPY_FR_M: return &as_freq_us2M; break;
	    case NPY_FR_W: return &as_freq_us2W; break;
	    case NPY_FR_B: return &as_freq_us2B; break;
	    case NPY_FR_D: return &as_freq_us2D; break;
	    case NPY_FR_h: return &as_freq_us2h; break;
	    case NPY_FR_m: return &as_freq_us2m; break;
	    case NPY_FR_s: return &as_freq_us2s; break;
	    case NPY_FR_ms: return &as_freq_us2ms; break;
	    case NPY_FR_ns: return &as_freq_us2ns; break;
	    case NPY_FR_ps: return &as_freq_us2ps; break;
	    case NPY_FR_fs: return &as_freq_us2fs; break;
	    case NPY_FR_as: return &as_freq_us2as; break;
	    }
	    break;
	case NPY_FR_ns:
	    switch (ofreq) {
	    case NPY_FR_Y: return &as_freq_ns2Y; break;
	    case NPY_FR_M: return &as_freq_ns2M; break;
	    case NPY_FR_W: return &as_freq_ns2W; break;
	    case NPY_FR_B: return &as_freq_ns2B; break;
	    case NPY_FR_D: return &as_freq_ns2D; break;
	    case NPY_FR_h: return &as_freq_ns2h; break;
	    case NPY_FR_m: return &as_freq_ns2m; break;
	    case NPY_FR_s: return &as_freq_ns2s; break;
	    case NPY_FR_ms: return &as_freq_ns2ms; break;
	    case NPY_FR_us: return &as_freq_ns2us; break;
	    case NPY_FR_ps: return &as_freq_ns2ps; break;
	    case NPY_FR_fs: return &as_freq_ns2fs; break;
	    case NPY_FR_as: return &as_freq_ns2as; break;
	    }
	    break;
	case NPY_FR_ps:
	    switch (ofreq) {
	    case NPY_FR_Y: return &as_freq_ps2Y; break;
	    case NPY_FR_M: return &as_freq_ps2M; break;
	    case NPY_FR_W: return &as_freq_ps2W; break;
	    case NPY_FR_B: return &as_freq_ps2B; break;
	    case NPY_FR_D: return &as_freq_ps2D; break;
	    case NPY_FR_h: return &as_freq_ps2h; break;
	    case NPY_FR_m: return &as_freq_ps2m; break;
	    case NPY_FR_s: return &as_freq_ps2s; break;
	    case NPY_FR_ms: return &as_freq_ps2ms; break;
	    case NPY_FR_us: return &as_freq_ps2us; break;
	    case NPY_FR_ns: return &as_freq_ps2ns; break;
	    case NPY_FR_fs: return &as_freq_ps2fs; break;
	    case NPY_FR_as: return &as_freq_ps2as; break;
	    }
	    break;
	case NPY_FR_fs:
	    switch (ofreq) {
	    case NPY_FR_Y: return &as_freq_fs2Y; break;
	    case NPY_FR_M: return &as_freq_fs2M; break;
	    case NPY_FR_W: return &as_freq_fs2W; break;
	    case NPY_FR_B: return &as_freq_fs2B; break;
	    case NPY_FR_D: return &as_freq_fs2D; break;
	    case NPY_FR_h: return &as_freq_fs2h; break;
	    case NPY_FR_m: return &as_freq_fs2m; break;
	    case NPY_FR_s: return &as_freq_fs2s; break;
	    case NPY_FR_ms: return &as_freq_fs2ms; break;
	    case NPY_FR_us: return &as_freq_fs2us; break;
	    case NPY_FR_ns: return &as_freq_fs2ns; break;
	    case NPY_FR_ps: return &as_freq_fs2ps; break;
	    case NPY_FR_as: return &as_freq_fs2as; break;
	    }
	    break;
	case NPY_FR_as:
	    switch (ofreq) {
	    case NPY_FR_Y: return &as_freq_as2Y; break;
	    case NPY_FR_M: return &as_freq_as2M; break;
	    case NPY_FR_W: return &as_freq_as2W; break;
	    case NPY_FR_B: return &as_freq_as2B; break;
	    case NPY_FR_D: return &as_freq_as2D; break;
	    case NPY_FR_h: return &as_freq_as2h; break;
	    case NPY_FR_m: return &as_freq_as2m; break;
	    case NPY_FR_s: return &as_freq_as2s; break;
	    case NPY_FR_ms: return &as_freq_as2ms; break;
	    case NPY_FR_us: return &as_freq_as2us; break;
	    case NPY_FR_ns: return &as_freq_as2ns; break;
	    case NPY_FR_ps: return &as_freq_as2ps; break;
	    case NPY_FR_fs: return &as_freq_as2fs; break;
	    }
	    break;
	default:
	    return &NO_FUNC;
	    break;
	    // error out
	}
	
    // error out
    return &NO_FUNC;
}

// Uses get_conversion_ftn to find which function to return
static long long as_freq_to_long(long long dlong, int ifreq, int ofreq)
{
    // Needs more error checking, but it works for now
    if (ifreq == ofreq)
	return -1;// Error out

    // grab conversion function based on ifreq and ofreq
    long long (*conversion_ftn)(long long) = get_conversion_ftn(ifreq, ofreq);
    // return conversion function ran with dlong
    return (*conversion_ftn)(dlong);
}

// Takes a long and an in frequency ( to emulate a date )
//  and an out frequency to learn the conversion to run
// OR takes a PyList and an in frequency and outputs a PyList as the second freq
// Returns a long
static PyObject *
convert_freq(PyObject *self, PyObject *args)
{
    PyObject *main_arg = NULL;    // string or datetime
    PyObject *ifreq_arg = NULL;	  // in frequency as string
    PyObject *ofreq_arg = NULL;	  // out frequency as string
    PyObject *result   = NULL;	  // long result

    long long dlong = 0;          // Stores the main_arg
    int ifreq = NPY_FR_ERR;			  // freq_arg is a PyObject to be parsed to freq
    int ofreq = NPY_FR_ERR;			  // freq_arg is a PyObject to be parsed to freq

    // Parse out main_arg & freq_arg
    if (!PyArg_ParseTuple(args, "OOO", &main_arg, &ifreq_arg, &ofreq_arg))
	return NULL;
    // Parse the in frequency into an int so we can use it easily
    if ((ifreq = _unit_from_str(PyString_AsString(ifreq_arg))) == NPY_FR_ERR)
	{
	    // If the frequency is invalid, set an error and return null
	    PyErr_SetString(PyExc_TypeError, "invalid frequency.");
	    return NULL;
	}
    // Parse the out frequency into an int so we can use it easily
    if ((ofreq = _unit_from_str(PyString_AsString(ofreq_arg))) == NPY_FR_ERR)
	{
	    // If the frequency is invalid, set an error and return null
	    PyErr_SetString(PyExc_TypeError, "invalid frequency.");
	    return NULL;
	}

    // Make sure main_arg is not NULL
    if (!main_arg)
	{
	    PyErr_SetString(PyExc_TypeError, "no date provided.");
	    return NULL;
	}
    // For Scalars
    if (PyLong_Check(main_arg))
	{
	    // XXX PyINCREF here?
	    // Convert main_arg to a long long
	    dlong = PyLong_AsLongLong(main_arg);
	
	    // All the basic tests are out of the way, now we need to figure out 
	    //  which frequency conversion to run based on the ofreq
	    result = PyLong_FromLongLong(as_freq_to_long(dlong, ifreq, ofreq));
	}
    // For lists
    else if (PyList_Check(main_arg))
	{
	    // Result needs to be a list size of main arg
	    result = PyList_New(0);
	    // get the pointer ftn here
	    // We shouldn't just use as_freq_to_long because that checks
	    //  each ifreq and ofreq. We'll always be using the same ifreq
	    //  and ofreq, so we just need that one function...
	    long long (*conversion_ftn)(long long) = get_conversion_ftn(ifreq, ofreq);
	    // Iterate through main_arg
	    Py_ssize_t idx;
	    for (idx = 0; idx < PyList_Size(main_arg); idx++)
		{
		    // extract correct value of main arg
		    long long dlong = PyLong_AsLongLong(PyList_GetItem(main_arg, idx));
		    long long resultant_dlong = (*conversion_ftn)(dlong);
		    // put calculated dlong into result
		    PyList_Append(result,
				  PyLong_FromLongLong(resultant_dlong));
		}
	}
    // For NumPy narrays
    else if (PyArray_Check(main_arg))
	{
	    // Create new 	
	}	
    else
	{
	    PyErr_SetString(PyExc_TypeError, "invalid long entry.");
	    return NULL;
	}
    return result;
}

//==================================================
// TimeDelta
//==================================================
// Very similar to datetime_to_long
// Takes a datetime timedelta object and a string as frequency
// Returns the number of (frequency) since Jan 1, 1970
static 
long long timedelta_to_long(PyObject* timedelta, int frequency)
{
    int year = 0, month = 0, day = 0, hour = 0, 
	minute = 0, second = 0, microsecond = 0;
	
    // Get the time units from PyDateTime
    year        = PyDateTime_GET_YEAR(timedelta);
    month       = PyDateTime_GET_MONTH(timedelta);
    day         = PyDateTime_GET_DAY(timedelta);
    hour        = PyDateTime_DATE_GET_HOUR(timedelta);
    minute      = PyDateTime_DATE_GET_MINUTE(timedelta);
    second      = PyDateTime_DATE_GET_SECOND(timedelta);
    microsecond = PyDateTime_DATE_GET_MICROSECOND(timedelta);
	
    // The return value
    long long result = 0;

    // The absolute number of days since 1970
    long long absdays = absdays_from_ymd(year, month, day);

    // These calculations depend on the frequency

    if (frequency == NPY_FR_Y) {
	result = year;
    } else if (frequency == NPY_FR_M) {
	result = (year) * 12 + month - 1;
    } else if (frequency == NPY_FR_W) {
	// 4 day offset for post 1970 to get correct week
	int dotw = day_of_week(absdays);
	if (absdays >= 0)
	    result = (absdays + 4) / 7;
	else
	    result = (absdays - dotw)/ 7;
    } else if (frequency == NPY_FR_B) {
	int dotw = day_of_week(absdays);
	// To get to Sunday, Jan 4, 1970
	// number of weeks * 5 + dotw [0-6] - Saturdays + 1 for offset
	if (absdays > 2)
	    result = ((absdays - dotw) / 7) * 5 + dotw - (dotw / 6) + 1;
	else 
	    result = dotw - 4 - (dotw / 6);
    } else if (frequency == NPY_FR_D) {
	result = absdays;
    } else if (frequency == NPY_FR_h) {
	result = absdays * 24 + hour;
    } else if (frequency == NPY_FR_m) {
	result = absdays * 1440 + hour * 60 + minute;
    } else if (frequency == NPY_FR_s) {
	result = absdays * 86400LL + abssecs_from_hms(hour, minute, second);
    } else if (frequency == NPY_FR_ms) {
	result = absdays * 86400000LL + abssecs_from_hms(hour, minute, second) * 1000LL
	    + (microsecond / 1000LL);
    } else if (frequency == NPY_FR_us) {
	result = absdays * 86400000000LL + abssecs_from_hms(hour, minute, second) * 1000000LL
	    + microsecond;
    }
    // Starting from here, we need extra units (ns, ps, fs, as)
    //  for correct precision: timedelta doesn't include beyond microsecond
    else if (frequency == NPY_FR_ns) {
	PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
	result = 0;
    } else if (frequency == NPY_FR_ps) {
	PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
	result = 0;
    } else if (frequency == NPY_FR_fs) {
	PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
	result = 0;
    } else if (frequency == NPY_FR_as) {
	PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
	result = 0;
    } else {
	// Throw some Not Valid Frequency error here
	result = -1;
    }	

    return result;
}

static char* timedelta_to_cstring(long long tlong, int freq)
{
    char result[64];
    switch (freq)
	{
	case NPY_FR_Y:
	    {
		sprintf(result, "%lld Years", tlong); 
		break;
	    }
	case NPY_FR_M:
	    {	
		if (tlong > 11)
		    {
			sprintf(result, "%lld Years, %d Days", 
				tlong / 12, 
				month_offset[0][tlong % 12]); 
		    }
		else
		    {
			sprintf(result, "%d Days", 
				month_offset[0][tlong % 12]); 
		    }
		break;
	    }
	case NPY_FR_W: 
	    {
		if (tlong > 51)
		    {
			sprintf(result, "%lli Years, %lli Days",
				tlong / 52,
				(tlong % 52) * 7);
		    }
		else
		    {
			sprintf(result, "%lli Days", tlong * 7);
		    }
		break;
	    }
	case NPY_FR_B:
	    {

	    }
	case NPY_FR_D:
	    {
		if (tlong > 364)
		    sprintf(result, "%lli Years, %lli Days",
			    tlong / 365,
			    tlong % 365);
		else
		    sprintf(result, "%lli Days", tlong);
		break;
	    }
	case NPY_FR_h:
	    {
		// 8760 hours in 365 days
		if (tlong > 8759)
		    {
			long long years = as_freq_h2Y(tlong);
			sprintf(result, "%lli Years, %lli Days %lli:00:00",
				years, (tlong / 24) % 365, tlong % 24);
			break;
		    }
		else if (tlong > 23)
		    {
			sprintf(result, "%lli Days, %lli:00:00",
				tlong / 24, tlong % 24);
		    }
		else
		    {
			sprintf(result, "%lli:00:00", tlong);
		    }
		break;
	    }
	case NPY_FR_m:
	    {
		// 525600 minutes in 365 days
		if (tlong > 525600)
		    {
			long long years = as_freq_m2Y(tlong);
			long long days  = as_freq_m2D(tlong % 525600);
			sprintf(result, "%lli Years, %lli Days, %lli:%lli:00",
				years, days, (tlong / 60) % 24, tlong % 60);
			break;
		    }
		else if (tlong > 1440)
		    {
			long long days = as_freq_m2D(tlong % 525600);
			sprintf(result, "%lli Days, %lli:%lli:00",
				days, (tlong / 60) % 24, tlong % 60);
			break;
		    }
		else
		    {
			sprintf(result, "%02lli:%02lli:00",
				(tlong / 60) % 24, tlong % 60);
		    }
		break;
	    }
	case NPY_FR_s:
	    {
		// How many Years
		// How many Days
		// How many hours
		// How many minutes
	    }
	case NPY_FR_ms:
	    {

	    }
	case NPY_FR_us:
	    {

	    }
	case NPY_FR_ns:
	    {

	    }
	case NPY_FR_ps:
	    {

	    }
	case NPY_FR_fs:
	    {

	    }
	case NPY_FR_as:
	    {
		break;
	    }
	default: return "error"; break;
	}
    return "error";
}

// TimeDelta long -> timedelta is very simple.
// DateTime timedelta -> long is very simple.
//  Takes a Python DateTime TimeDelta object and a freq
//   returns a long
static PyObject *
dt_timedelta_to_long(PyObject *self, PyObject *args)
{
    PyObject *dt_timedelta_arg = NULL;
    PyObject *freq_arg = NULL;

    PyDateTime_IMPORT;

    int freq = NPY_FR_ERR;

    // Parse out long_arg & freq_arg
    if (!PyArg_ParseTuple(args, "OO", &dt_timedelta_arg, &freq_arg))
	return NULL;

    if (!PyDelta_Check(dt_timedelta_arg))
	{
	    PyErr_SetString(PyExc_TypeError, "invalid datetime timedelta entry.");
	    return NULL;
	}
	
    if ((freq = _unit_from_str(PyString_AsString(freq_arg))) == NPY_FR_ERR)
	{
	    // If the frequency is invalid, set an error and return null
	    PyErr_SetString(PyExc_TypeError, "invalid frequency.");
	    return NULL;
	}
    return PyLong_FromLongLong(timedelta_to_long(dt_timedelta_arg, freq));
}
// TimeDelta TimeDelta -> long is also simple.
// TimeDelta TimeDelta -> String is less simple.
//  Takes a long long value for timedelta and 
//   a frequency as it's frequency
//  Returns a string formatted:
//  X Years, Y Days xx:xx:xx.xxxxxx
static PyObject *
timedelta_to_string(PyObject *self, PyObject *args)
{
    PyObject *long_arg = NULL;
    PyObject *freq_arg = NULL;
	
    long long tlong = 0;
    int freq = NPY_FR_ERR;

    // Parse out long_arg & freq_arg
    if (!PyArg_ParseTuple(args, "OO", &long_arg, &freq_arg))
	return NULL;

    if (PyLong_Check(long_arg))
	tlong = PyLong_AsLongLong(long_arg);
    else
	PyErr_SetString(PyExc_TypeError, "invalid long entry.");
	
    if ((freq = _unit_from_str(PyString_AsString(freq_arg))) == NPY_FR_ERR)
	{
	    // If the frequency is invalid, set an error and return null
	    PyErr_SetString(PyExc_TypeError, "invalid frequency.");
	    return NULL;
	}
    return PyString_FromString(timedelta_to_cstring(tlong, freq));
}
// TimeDelta + TimeDelta is interesting
// if it's one TimeDelta and one TimeDelta
//  return one TimeDelta
// if it's one list and one TimeDelta
//  return one (list + that TimeDelta)
// if it's one list and one list
//  return one (list + that list)
static PyObject *
timedelta_plus_timedelta(PyObject *self, PyObject *args)
{
    PyObject *long_arg1 = NULL;
    PyObject *freq_arg1 = NULL;
    PyObject *long_arg2 = NULL;
    PyObject *freq_arg2 = NULL;
	
    long long tlong1 = 0;
    long long tlong2 = 0;
    int freq1 = NPY_FR_ERR;
    int freq2 = NPY_FR_ERR;

    // Parse out long_arg & freq_arg
    if (!PyArg_ParseTuple(args, "OOOO", &long_arg1, &freq_arg1,
			  &long_arg2, &freq_arg2))
	return NULL;

    // Parse out both freqs	
    if (((freq1 = _unit_from_str(PyString_AsString(freq_arg1))) == NPY_FR_ERR) ||
	((freq2 = _unit_from_str(PyString_AsString(freq_arg2))) == NPY_FR_ERR))
	{
	    // If the frequency is invalid, set an error and return null
	    PyErr_SetString(PyExc_TypeError, "invalid frequency.");
	    return NULL;
	}
	
    // TimeDelta + TimeDelta
    if (PyLong_Check(long_arg1) && PyLong_Check(long_arg2))
	{
	    tlong1 = PyLong_AsLongLong(long_arg1);
	    tlong2 = PyLong_AsLongLong(long_arg2);

	    if (freq1 == freq2)
		return PyLong_FromLongLong(tlong1 + tlong2);
	    // Freq1 is more precise than freq2
	    //  change freq2 to freq1
	    else if (freq1 < freq2)
		{
		    long long (* conversion_ftn) (long long) = 
			get_conversion_ftn(freq2, freq1);
		    return PyLong_FromLongLong(conversion_ftn(tlong2) + tlong1);
		}
	    // Freq1 is less precise than freq2
	    //  change freq1 to freq2
	    else
		{
		    long long (* conversion_ftn) (long long) = 
			get_conversion_ftn(freq1, freq2);
		    return PyLong_FromLongLong(conversion_ftn(tlong1) + tlong2);
		}
	}
    // List + TimeDelta
    else if (PyList_Check(long_arg1) && PyLong_Check(long_arg2))
	{
	    long long tlong_scalar = PyLong_AsLongLong(long_arg2);
	    PyObject* result = PyList_New(0);
	    long long (*conversion_ftn) (long long) = NULL;
	    int smaller_arg = 0; // which argument is more precise?

	    // Freq1 is more precise than freq2
	    //  change freq2 to freq1
	    if (freq1 < freq2)
		{
		    conversion_ftn = get_conversion_ftn(freq2, freq1);
		    // list is more precise than scalar
		    smaller_arg = 0;
		}
	    // Freq1 is less precise than freq2
	    //  change freq1 to freq2
	    else
		{
		    conversion_ftn = get_conversion_ftn(freq1, freq2);
		    // scalar is more precise than list
		    smaller_arg = 1;
		}

	    // Iterate through long_arg1
	    Py_ssize_t idx;
	    for (idx = 0; idx < PyList_Size(long_arg1); idx++)
		{
		    // extract correct value of main arg
		    long long tlong_member = 
			PyLong_AsLongLong(PyList_GetItem(long_arg1, idx));
		    long long tlong_result;
		    if (conversion_ftn)
			{
			    if (smaller_arg)
				tlong_result = (*conversion_ftn)(tlong_member) 
				    + tlong_scalar;
			    else
				tlong_result = (*conversion_ftn)(tlong_scalar) 
				    + tlong_member;
			}
		    else
			tlong_result = tlong_member + tlong2;
		    // put calculated dlong into result
		    PyList_Append(result,
				  PyLong_FromLongLong(tlong_result));
		}
	    return result;
	}
    // List + List
    else if (PyList_Check(long_arg1) && PyList_Check(long_arg2))
	{
	    if (PyList_Size(long_arg1) != PyList_Size(long_arg2))
		{
		    PyErr_SetString(PyExc_TypeError, "list sizes must be equal.");
		    return NULL;
		}

	    PyObject* result = PyList_New(0);
	    long long (*conversion_ftn)(long long) = NULL;
	    int smaller_arg = 0; // which argument is smaller?

	    // Freq1 is more precise than freq2
	    //  change freq2 to freq1
	    if (freq1 < freq2)
		{
		    conversion_ftn= get_conversion_ftn(freq2, freq1);
		    smaller_arg = 0;
		}
	    // Freq1 is less precise than freq2
	    //  change freq1 to freq2
	    else
		{
		    conversion_ftn = get_conversion_ftn(freq1, freq2);
		    smaller_arg = 1;
		}

	    // Iterate through list_1
	    Py_ssize_t idx;
	    for (idx = 0; idx < PyList_Size(long_arg1); idx++)
		{
		    // extract correct value of main arg
		    long long tlong_member1 = PyLong_AsLongLong(
								PyList_GetItem(long_arg1, idx));
		    long long tlong_member2 = PyLong_AsLongLong(
								PyList_GetItem(long_arg2, idx));
		    long long tlong_result;
		    if (conversion_ftn)
			{
			    // if the second is smaller
			    if (smaller_arg)
				{
				    tlong_result = (*conversion_ftn)(tlong_member2) 
					+ tlong_member1;
				}
			    else
				{
				    tlong_result = (*conversion_ftn)(tlong_member1) 
					+ tlong_member2;
				}
			}
		    else
			tlong_result = tlong1 + tlong2;
		    // put calculated dlong into result
		    PyList_Append(result,
				  PyLong_FromLongLong(tlong_result));
		}
	    return result;
	}
    else
	{
	    PyErr_SetString(PyExc_TypeError, "invalid entries.");
	    return NULL;
	}
}

// This is the same as td_plus_td, but minus instead...
// This should be stacked on to plus, but I haven't decided how yet
static PyObject *
timedelta_minus_timedelta(PyObject *self, PyObject *args)
{
    PyObject *long_arg1 = NULL;
    PyObject *freq_arg1 = NULL;
    PyObject *long_arg2 = NULL;
    PyObject *freq_arg2 = NULL;
	
    long long tlong1 = 0;
    long long tlong2 = 0;
    int freq1 = NPY_FR_ERR;
    int freq2 = NPY_FR_ERR;

    // Parse out long_arg & freq_arg
    if (!PyArg_ParseTuple(args, "OOOO", &long_arg1, &freq_arg1,
			  &long_arg2, &freq_arg2))
	return NULL;

    // Parse out both freqs	
    if (((freq1 = _unit_from_str(PyString_AsString(freq_arg1))) == NPY_FR_ERR) ||
	((freq2 = _unit_from_str(PyString_AsString(freq_arg2))) == NPY_FR_ERR))
	{
	    // If the frequency is invalid, set an error and return null
	    PyErr_SetString(PyExc_TypeError, "invalid frequency.");
	    return NULL;
	}
	
    // TimeDelta + TimeDelta
    if (PyLong_Check(long_arg1) && PyLong_Check(long_arg2))
	{
	    tlong1 = PyLong_AsLongLong(long_arg1);
	    tlong2 = PyLong_AsLongLong(long_arg2);

	    if (freq1 == freq2)
		return PyLong_FromLongLong(tlong1 - tlong2);
	    // Freq1 is more precise than freq2
	    //  change freq2 to freq1
	    else if (freq1 < freq2)
		{
		    long long (* conversion_ftn) (long long) = 
			get_conversion_ftn(freq2, freq1);
		    return PyLong_FromLongLong(-conversion_ftn(tlong2) + tlong1);
		}
	    // Freq1 is less precise than freq2
	    //  change freq1 to freq2
	    else
		{
		    long long (* conversion_ftn) (long long) = 
			get_conversion_ftn(freq1, freq2);
		    return PyLong_FromLongLong(conversion_ftn(tlong1) - tlong2);
		}
	}
    // List + TimeDelta
    else if (PyList_Check(long_arg1) && PyLong_Check(long_arg2))
	{
	    long long tlong_scalar = PyLong_AsLongLong(long_arg2);
	    PyObject* result = PyList_New(0);
	    long long (*conversion_ftn) (long long) = NULL;
	    int smaller_arg = 0; // which argument is more precise?

	    // Freq1 is more precise than freq2
	    //  change freq2 to freq1
	    if (freq1 < freq2)
		{
		    conversion_ftn = get_conversion_ftn(freq2, freq1);
		    // list is more precise than scalar
		    smaller_arg = 0;
		}
	    // Freq1 is less precise than freq2
	    //  change freq1 to freq2
	    else
		{
		    conversion_ftn = get_conversion_ftn(freq1, freq2);
		    // scalar is more precise than list
		    smaller_arg = 1;
		}

	    // Iterate through long_arg1
	    Py_ssize_t idx;
	    for (idx = 0; idx < PyList_Size(long_arg1); idx++)
		{
		    // extract correct value of main arg
		    long long tlong_member = 
			PyLong_AsLongLong(PyList_GetItem(long_arg1, idx));
		    long long tlong_result;
		    if (conversion_ftn)
			{
			    if (smaller_arg)
				tlong_result = (*conversion_ftn)(tlong_member) 
				    - tlong_scalar;
			    else
				tlong_result = -(*conversion_ftn)(tlong_scalar) 
				    + tlong_member;
			}
		    else
			tlong_result = tlong_member + tlong2;
		    // put calculated dlong into result
		    PyList_Append(result,
				  PyLong_FromLongLong(tlong_result));
		}
	    return result;
	}
    // List + List
    else if (PyList_Check(long_arg1) && PyList_Check(long_arg2))
	{
	    if (PyList_Size(long_arg1) != PyList_Size(long_arg2))
		{
		    PyErr_SetString(PyExc_TypeError, "list sizes must be equal.");
		    return NULL;
		}

	    PyObject* result = PyList_New(0);
	    long long (*conversion_ftn)(long long) = NULL;
	    int smaller_arg = 0; // which argument is smaller?

	    // Freq1 is more precise than freq2
	    //  change freq2 to freq1
	    if (freq1 < freq2)
		{
		    conversion_ftn= get_conversion_ftn(freq2, freq1);
		    smaller_arg = 0;
		}
	    // Freq1 is less precise than freq2
	    //  change freq1 to freq2
	    else
		{
		    conversion_ftn = get_conversion_ftn(freq1, freq2);
		    smaller_arg = 1;
		}

	    // Iterate through list_1
	    Py_ssize_t idx;
	    for (idx = 0; idx < PyList_Size(long_arg1); idx++)
		{
		    // extract correct value of main arg
		    long long tlong_member1 = PyLong_AsLongLong(
								PyList_GetItem(long_arg1, idx));
		    long long tlong_member2 = PyLong_AsLongLong(
								PyList_GetItem(long_arg2, idx));
		    long long tlong_result;
		    if (conversion_ftn)
			{
			    // if the second is smaller
			    if (smaller_arg)
				{
				    tlong_result = -(*conversion_ftn)(tlong_member2) 
					+ tlong_member1;
				}
			    else
				{
				    tlong_result = (*conversion_ftn)(tlong_member1) 
					- tlong_member2;
				}
			}
		    else
			tlong_result = tlong1 - tlong2;
		    // put calculated dlong into result
		    PyList_Append(result,
				  PyLong_FromLongLong(tlong_result));
		}
	    return result;
	}
    else
	{
	    PyErr_SetString(PyExc_TypeError, "invalid entries.");
	    return NULL;
	}
}
// TimeDelta + DateTime is very interesting
// TimeDelta - DateTime is very interesting
//==================================================
// Frequency conversion UFunc
//==================================================


// This is the ufunc for handling "specific" frequency conversions
//  Later, format it as astype_Y2M_ufunc, astype_Y2W_ufunc, etc
//  and set each of these to a seperate cf_function[] depending on the
//  cf_signature
static void
convert_freq_ufunc(char **args, npy_intp *dimensions, \
		   npy_intp *steps, void *extra)
{
    npy_intp idx;
    npy_intp insteps = steps[0], outsteps = steps[1];
    npy_intp n = dimensions[0];
    char *input = args[0], *output = args[1];
    //	char *freq_char = (char *) extra;
    // parse freq
    int freq;
    //if ((freq = _unit_from_str(*freq_char)) == NPY_FR_ERR)
	
    //	freq = _unit_from_str(*freq_char);
    // error out
	
    long long (*conversion_ftn)(long long) = NULL;

    // Pull out freqs from extra OR
    //  grab freqs depending on dtype
    // int freq1 = this dtype
    // int freq2 = input 2's dtype
    // Get conversion function
    //conversion_ftn= get_conversion_ftn(13, freq);
	
    for (idx = 0; idx < n; idx++)
	{
	    // Perform operation
	    *((long long *)output) = conversion_ftn(*((long long *)input));

	    // Iterate over data
	    input  += insteps;
	    output += outsteps;
	}
}

// Which specific looping function to run
static PyUFuncGenericFunction cf_functions[] = \
    {convert_freq_ufunc};

// What GENERIC ufunc to run
static void* cf_data[1];

// Correct argument signatures
static char cf_signatures[]=\
    {NPY_INT64, NPY_INT64};

/*
 * Eventually, these should be formatted as:
 *  NPY_DATETIME64Y, NPY_DATETIMEM
 *  NPY_DATETIME64Y, NPY_DATETIMEW
 *  NPY_DATETIME64Y, NPY_DATETIMEB
 *  NPY_DATETIME64Y, NPY_DATETIMEW
 *  etc
 *
 *  with each type deserving a different function
 */

//==================================================
// Module
//==================================================

// Tell Python what methods we can run from this module
static PyMethodDef methods[] = {
    {"set_callback", (PyCFunction)set_callback, 
     METH_VARARGS, ""},
    {"date_to_long", date_to_long,
     METH_VARARGS, ""},
    {"long_to_datetime", long_to_datetime,
     METH_VARARGS, ""},
    {"long_to_datestring", long_to_datestring,
     METH_VARARGS, ""},
    {"convert_freq", (PyCFunction)convert_freq,
     METH_VARARGS, ""},
    {"dt_timedelta_to_long", (PyCFunction)dt_timedelta_to_long,
     METH_VARARGS, ""},
    {"timedelta_to_string", (PyCFunction)timedelta_to_string,
     METH_VARARGS, ""},
    {"timedelta_plus_timedelta", (PyCFunction)timedelta_plus_timedelta,
     METH_VARARGS, ""},
    {"timedelta_minus_timedelta", (PyCFunction)timedelta_minus_timedelta,
     METH_VARARGS, ""},
    {NULL, NULL}
};

PyMODINIT_FUNC
initparsedates(void)
{
    PyObject *cf_test, *dict, *module;
    module = Py_InitModule("parsedates", methods);
    import_array();
    import_ufunc();

    // Can't set this until import_ufunc()
    cf_data[0] = PyUFunc_d_d;
    cf_test = PyUFunc_FromFuncAndData(cf_functions,
				      cf_data, cf_signatures, 1, 1, 1,
				      PyUFunc_One, "cf_test",
				      "some description", 0);

    dict = PyModule_GetDict(module);
    PyDict_SetItemString(dict, "cf_test", cf_test);
    Py_DECREF(cf_test);

    if (module == NULL)
	return;
}
