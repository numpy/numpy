from __future__ import division, absolute_import, print_function

from scipy import weave

class YMD(object):
    year = 0
    month = 0
    days = 0


month_offset = [
    [ 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 ],
    [ 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366 ]
]

days_in_month = [
    [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ],
    [ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ]
]

def is_leapyear(year):
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))


# Return the year offset, that is the absolute date of the day
#   31.12.(year-1) since 31.12.1969 in the proleptic Gregorian calendar.

def year_offset(year):
    code = """
    year-=1970;
    if ((year+1969) >= 0 || -1/4 == -1)
        return_val = year*365 + year/4 - year/100 + year/400;
    else
        return_val = year*365 + (year-3)/4 - (year-99)/100 + (year-399)/400;
        """
    return weave.inline(code,['year'])


def days_from_ymd(year, month, day):

    leap = is_leapyear(year)

    # Negative month values indicate months relative to the years end */
    if (month < 0): month += 13
    if not (month >= 1 and month<=12):
        raise ValueError("month out of range (1-21): %d" % month)

    # Negative values indicate days relative to the months end */
    if (day < 0): day += days_in_month[leap][month - 1] + 1
    if not (day >= 1 and day <= days_in_month[leap][month-1]):
        raise ValueError("day out of range: %d" % day)

    # Number of days between Dec 31, (year - 1) and Dec 31, 1969
    #    (can be negative).
    #
    yearoffset = year_offset(year);

    # Calculate the number of days using yearoffset */
    # Jan 1, 1970 is day 0 and thus Dec. 31, 1969 is day -1 */
    absdate = day-1 + month_offset[leap][month - 1] + yearoffset;

    return absdate;


def ymd_from_days(days):
    ymd = YMD()

    year = 1970 + days / 365.2425
