#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <locale.h>
#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#ifdef HAVE_STRTOLD_L
#include <stdlib.h>
#ifdef HAVE_XLOCALE_H
    /*
     * the defines from xlocale.h are included in locale.h on some sytems;
     * see gh-8367
     */
    #include <xlocale.h>
#endif
#endif



/*
 * From the C99 standard, section 7.19.6: The exponent always contains at least
 * two digits, and only as many more digits as necessary to represent the
 * exponent.
 */

#define MIN_EXPONENT_DIGITS 2

/*
 * Ensure that any exponent, if present, is at least MIN_EXPONENT_DIGITS
 * in length.
 */
static void
ensure_minimum_exponent_length(char* buffer, size_t buf_size)
{
    char *p = strpbrk(buffer, "eE");
    if (p && (*(p + 1) == '-' || *(p + 1) == '+')) {
        char *start = p + 2;
        int exponent_digit_cnt = 0;
        int leading_zero_cnt = 0;
        int in_leading_zeros = 1;
        int significant_digit_cnt;

        /* Skip over the exponent and the sign. */
        p += 2;

        /* Find the end of the exponent, keeping track of leading zeros. */
        while (*p && isdigit(Py_CHARMASK(*p))) {
            if (in_leading_zeros && *p == '0') {
                ++leading_zero_cnt;
            }
            if (*p != '0') {
                in_leading_zeros = 0;
            }
            ++p;
            ++exponent_digit_cnt;
        }

        significant_digit_cnt = exponent_digit_cnt - leading_zero_cnt;
        if (exponent_digit_cnt == MIN_EXPONENT_DIGITS) {
            /*
             * If there are 2 exactly digits, we're done,
             * regardless of what they contain
             */
        }
        else if (exponent_digit_cnt > MIN_EXPONENT_DIGITS) {
            int extra_zeros_cnt;

            /*
             * There are more than 2 digits in the exponent.  See
             * if we can delete some of the leading zeros
             */
            if (significant_digit_cnt < MIN_EXPONENT_DIGITS) {
                significant_digit_cnt = MIN_EXPONENT_DIGITS;
            }
            extra_zeros_cnt = exponent_digit_cnt - significant_digit_cnt;

            /*
             * Delete extra_zeros_cnt worth of characters from the
             * front of the exponent
             */
            assert(extra_zeros_cnt >= 0);

            /*
             * Add one to significant_digit_cnt to copy the
             * trailing 0 byte, thus setting the length
             */
            memmove(start, start + extra_zeros_cnt, significant_digit_cnt + 1);
        }
        else {
            /*
             * If there are fewer than 2 digits, add zeros
             * until there are 2, if there's enough room
             */
            int zeros = MIN_EXPONENT_DIGITS - exponent_digit_cnt;
            if (start + zeros + exponent_digit_cnt + 1 < buffer + buf_size) {
                memmove(start + zeros, start, exponent_digit_cnt + 1);
                memset(start, '0', zeros);
            }
        }
    }
}

/*
 * Ensure that buffer has a decimal point in it.  The decimal point
 * will not be in the current locale, it will always be '.'
 */
static void
ensure_decimal_point(char* buffer, size_t buf_size)
{
    int insert_count = 0;
    char* chars_to_insert;

    /* search for the first non-digit character */
    char *p = buffer;
    if (*p == '-' || *p == '+')
        /*
         * Skip leading sign, if present.  I think this could only
         * ever be '-', but it can't hurt to check for both.
         */
        ++p;
    while (*p && isdigit(Py_CHARMASK(*p))) {
        ++p;
    }
    if (*p == '.') {
        if (isdigit(Py_CHARMASK(*(p+1)))) {
            /*
             * Nothing to do, we already have a decimal
             * point and a digit after it.
             */
        }
        else {
            /*
             * We have a decimal point, but no following
             * digit.  Insert a zero after the decimal.
             */
            ++p;
            chars_to_insert = "0";
            insert_count = 1;
        }
    }
    else {
        chars_to_insert = ".0";
        insert_count = 2;
    }
    if (insert_count) {
        size_t buf_len = strlen(buffer);
        if (buf_len + insert_count + 1 >= buf_size) {
            /*
             * If there is not enough room in the buffer
             * for the additional text, just skip it.  It's
             * not worth generating an error over.
             */
        }
        else {
            memmove(p + insert_count, p, buffer + strlen(buffer) - p + 1);
            memcpy(p, chars_to_insert, insert_count);
        }
    }
}

/* see FORMATBUFLEN in unicodeobject.c */
#define FLOAT_FORMATBUFLEN 120

/*
 * Given a string that may have a decimal point in the current
 * locale, change it back to a dot.  Since the string cannot get
 * longer, no need for a maximum buffer size parameter.
 */
static void
change_decimal_from_locale_to_dot(char* buffer)
{
    struct lconv *locale_data = localeconv();
    const char *decimal_point = locale_data->decimal_point;

    if (decimal_point[0] != '.' || decimal_point[1] != 0) {
        size_t decimal_point_len = strlen(decimal_point);

        if (*buffer == '+' || *buffer == '-') {
            buffer++;
        }
        while (isdigit(Py_CHARMASK(*buffer))) {
            buffer++;
        }
        if (strncmp(buffer, decimal_point, decimal_point_len) == 0) {
            *buffer = '.';
            buffer++;
            if (decimal_point_len > 1) {
                /* buffer needs to get smaller */
                size_t rest_len = strlen(buffer + (decimal_point_len - 1));
                memmove(buffer, buffer + (decimal_point_len - 1), rest_len);
                buffer[rest_len] = 0;
            }
        }
    }
}

/*
 * Check that the format string is a valid one for NumPyOS_ascii_format*
 */
static int
check_ascii_format(const char *format)
{
    char format_char;
    size_t format_len = strlen(format);

    /* The last character in the format string must be the format char */
    format_char = format[format_len - 1];

    if (format[0] != '%') {
        return -1;
    }

    /*
     * I'm not sure why this test is here.  It's ensuring that the format
     * string after the first character doesn't have a single quote, a
     * lowercase l, or a percent. This is the reverse of the commented-out
     * test about 10 lines ago.
     */
    if (strpbrk(format + 1, "'l%")) {
        return -1;
    }

    /*
     * Also curious about this function is that it accepts format strings
     * like "%xg", which are invalid for floats.  In general, the
     * interface to this function is not very good, but changing it is
     * difficult because it's a public API.
     */
    if (!(format_char == 'e' || format_char == 'E'
          || format_char == 'f' || format_char == 'F'
          || format_char == 'g' || format_char == 'G')) {
        return -1;
    }

    return 0;
}

/*
 * Fix the generated string: make sure the decimal is ., that exponent has a
 * minimal number of digits, and that it has a decimal + one digit after that
 * decimal if decimal argument != 0 (Same effect that 'Z' format in
 * PyOS_ascii_formatd
 */
static char*
fix_ascii_format(char* buf, size_t buflen, int decimal)
{
    /*
     * Get the current locale, and find the decimal point string.
     * Convert that string back to a dot.
     */
    change_decimal_from_locale_to_dot(buf);

    /*
     * If an exponent exists, ensure that the exponent is at least
     * MIN_EXPONENT_DIGITS digits, providing the buffer is large enough
     * for the extra zeros.  Also, if there are more than
     * MIN_EXPONENT_DIGITS, remove as many zeros as possible until we get
     * back to MIN_EXPONENT_DIGITS
     */
    ensure_minimum_exponent_length(buf, buflen);

    if (decimal != 0) {
        ensure_decimal_point(buf, buflen);
    }

    return buf;
}

/*
 * NumPyOS_ascii_format*:
 *      - buffer: A buffer to place the resulting string in
 *      - buf_size: The length of the buffer.
 *      - format: The printf()-style format to use for the code to use for
 *      converting.
 *      - value: The value to convert
 *      - decimal: if != 0, always has a decimal, and at leasat one digit after
 *      the decimal. This has the same effect as passing 'Z' in the origianl
 *      PyOS_ascii_formatd
 *
 * This is similar to PyOS_ascii_formatd in python > 2.6, except that it does
 * not handle 'n', and handles nan / inf.
 *
 * Converts a #gdouble to a string, using the '.' as decimal point. To format
 * the number you pass in a printf()-style format string. Allowed conversion
 * specifiers are 'e', 'E', 'f', 'F', 'g', 'G'.
 *
 * Return value: The pointer to the buffer with the converted string.
 */
#define ASCII_FORMAT(type, suffix, print_type)                          \
    NPY_NO_EXPORT char*                                                 \
    NumPyOS_ascii_format ## suffix(char *buffer, size_t buf_size,       \
                                   const char *format,                  \
                                   type val, int decimal)               \
    {                                                                   \
        if (npy_isfinite(val)) {                                        \
            if (check_ascii_format(format)) {                           \
                return NULL;                                            \
            }                                                           \
            PyOS_snprintf(buffer, buf_size, format, (print_type)val);   \
            return fix_ascii_format(buffer, buf_size, decimal);         \
        }                                                               \
        else if (npy_isnan(val)){                                       \
            if (buf_size < 4) {                                         \
                return NULL;                                            \
            }                                                           \
            strcpy(buffer, "nan");                                      \
        }                                                               \
        else {                                                          \
            if (npy_signbit(val)) {                                     \
                if (buf_size < 5) {                                     \
                    return NULL;                                        \
                }                                                       \
                strcpy(buffer, "-inf");                                 \
            }                                                           \
            else {                                                      \
                if (buf_size < 4) {                                     \
                    return NULL;                                        \
                }                                                       \
                strcpy(buffer, "inf");                                  \
            }                                                           \
        }                                                               \
        return buffer;                                                  \
    }

ASCII_FORMAT(float, f, float)
ASCII_FORMAT(double, d, double)
#ifndef FORCE_NO_LONG_DOUBLE_FORMATTING
ASCII_FORMAT(long double, l, long double)
#else
ASCII_FORMAT(long double, l, double)
#endif

/*
 * NumPyOS_ascii_isspace:
 *
 * Same as isspace under C locale
 */
NPY_NO_EXPORT int
NumPyOS_ascii_isspace(int c)
{
    return c == ' ' || c == '\f' || c == '\n' || c == '\r' || c == '\t'
                    || c == '\v';
}


/*
 * NumPyOS_ascii_isalpha:
 *
 * Same as isalpha under C locale
 */
static int
NumPyOS_ascii_isalpha(char c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}


/*
 * NumPyOS_ascii_isdigit:
 *
 * Same as isdigit under C locale
 */
static int
NumPyOS_ascii_isdigit(char c)
{
    return (c >= '0' && c <= '9');
}


/*
 * NumPyOS_ascii_isalnum:
 *
 * Same as isalnum under C locale
 */
static int
NumPyOS_ascii_isalnum(char c)
{
    return NumPyOS_ascii_isdigit(c) || NumPyOS_ascii_isalpha(c);
}


/*
 * NumPyOS_ascii_tolower:
 *
 * Same as tolower under C locale
 */
static int
NumPyOS_ascii_tolower(int c)
{
    if (c >= 'A' && c <= 'Z') {
        return c + ('a'-'A');
    }
    return c;
}


/*
 * NumPyOS_ascii_strncasecmp:
 *
 * Same as strncasecmp under C locale
 */
static int
NumPyOS_ascii_strncasecmp(const char* s1, const char* s2, size_t len)
{
    while (len > 0 && *s1 != '\0' && *s2 != '\0') {
        int diff = NumPyOS_ascii_tolower(*s1) - NumPyOS_ascii_tolower(*s2);
        if (diff != 0) {
            return diff;
        }
        ++s1;
        ++s2;
        --len;
    }
    if (len > 0) {
        return *s1 - *s2;
    }
    return 0;
}

/*
 * NumPyOS_ascii_strtod_plain:
 *
 * PyOS_ascii_strtod work-alike, with no enhanced features,
 * for forward compatibility with Python >= 2.7
 */
static double
NumPyOS_ascii_strtod_plain(const char *s, char** endptr)
{
    double result;
    NPY_ALLOW_C_API_DEF;
    NPY_ALLOW_C_API;
    result = PyOS_string_to_double(s, endptr, NULL);
    if (PyErr_Occurred()) {
        if (endptr) {
            *endptr = (char*)s;
        }
        PyErr_Clear();
    }
    NPY_DISABLE_C_API;
    return result;
}

/*
 * NumPyOS_ascii_strtod:
 *
 * Work around bugs in PyOS_ascii_strtod
 */
NPY_NO_EXPORT double
NumPyOS_ascii_strtod(const char *s, char** endptr)
{
    const char *p;
    double result;

    while (NumPyOS_ascii_isspace(*s)) {
        ++s;
    }

    /*
     * ##1
     *
     * Recognize POSIX inf/nan representations on all platforms.
     */
    p = s;
    result = 1.0;
    if (*p == '-') {
        result = -1.0;
        ++p;
    }
    else if (*p == '+') {
        ++p;
    }
    if (NumPyOS_ascii_strncasecmp(p, "nan", 3) == 0) {
        p += 3;
        if (*p == '(') {
            ++p;
            while (NumPyOS_ascii_isalnum(*p) || *p == '_') {
                ++p;
            }
            if (*p == ')') {
                ++p;
            }
        }
        if (endptr != NULL) {
            *endptr = (char*)p;
        }
        return NPY_NAN;
    }
    else if (NumPyOS_ascii_strncasecmp(p, "inf", 3) == 0) {
        p += 3;
        if (NumPyOS_ascii_strncasecmp(p, "inity", 5) == 0) {
            p += 5;
        }
        if (endptr != NULL) {
            *endptr = (char*)p;
        }
        return result*NPY_INFINITY;
    }
    /* End of ##1 */

    return NumPyOS_ascii_strtod_plain(s, endptr);
}

NPY_NO_EXPORT long double
NumPyOS_ascii_strtold(const char *s, char** endptr)
{
    const char *p;
    long double result;
#ifdef HAVE_STRTOLD_L
    locale_t clocale;
#endif

    while (NumPyOS_ascii_isspace(*s)) {
        ++s;
    }

    /*
     * ##1
     *
     * Recognize POSIX inf/nan representations on all platforms.
     */
    p = s;
    result = 1.0;
    if (*p == '-') {
        result = -1.0;
        ++p;
    }
    else if (*p == '+') {
        ++p;
    }
    if (NumPyOS_ascii_strncasecmp(p, "nan", 3) == 0) {
        p += 3;
        if (*p == '(') {
            ++p;
            while (NumPyOS_ascii_isalnum(*p) || *p == '_') {
                ++p;
            }
            if (*p == ')') {
                ++p;
            }
        }
        if (endptr != NULL) {
            *endptr = (char*)p;
        }
        return NPY_NAN;
    }
    else if (NumPyOS_ascii_strncasecmp(p, "inf", 3) == 0) {
        p += 3;
        if (NumPyOS_ascii_strncasecmp(p, "inity", 5) == 0) {
            p += 5;
        }
        if (endptr != NULL) {
            *endptr = (char*)p;
        }
        return result*NPY_INFINITY;
    }
    /* End of ##1 */

#ifdef HAVE_STRTOLD_L
    clocale = newlocale(LC_ALL_MASK, "C", NULL);
    if (clocale) {
        errno = 0;
        result = strtold_l(s, endptr, clocale);
        freelocale(clocale);
    }
    else {
        if (endptr != NULL) {
            *endptr = (char*)s;
        }
        result = 0;
    }
    return result;
#else
    return NumPyOS_ascii_strtod(s, endptr);
#endif
}

/*
 * read_numberlike_string:
 *      * fp: FILE pointer
 *      * value: Place to store the value read
 *
 * Read what looks like valid numeric input and store it in a buffer
 * for later parsing as a number.
 *
 * Similarly to fscanf, this function always consumes leading whitespace,
 * and any text that could be the leading part in valid input.
 *
 * Return value: similar to fscanf.
 *      * 0 if no number read,
 *      * 1 if a number read,
 *      * EOF if end-of-file met before reading anything.
 */
static int
read_numberlike_string(FILE *fp, char *buffer, size_t buflen)
{

    char *endp;
    char *p;
    int c;
    int ok;

    /*
     * Fill buffer with the leftmost matching part in regexp
     *
     *     \s*[+-]? ( [0-9]*\.[0-9]+([eE][+-]?[0-9]+)
     *              | nan  (  \([:alphanum:_]*\) )?
     *              | inf(inity)?
     *              )
     *
     * case-insensitively.
     *
     * The "do { ... } while (0)" wrapping in macros ensures that they behave
     * properly eg. in "if ... else" structures.
     */

#define END_MATCH()                                                         \
        goto buffer_filled

#define NEXT_CHAR()                                                         \
        do {                                                                \
            if (c == EOF || endp >= buffer + buflen - 1)            \
                END_MATCH();                                                \
            *endp++ = (char)c;                                              \
            c = getc(fp);                                                   \
        } while (0)

#define MATCH_ALPHA_STRING_NOCASE(string)                                   \
        do {                                                                \
            for (p=(string); *p!='\0' && (c==*p || c+('a'-'A')==*p); ++p)   \
                NEXT_CHAR();                                                \
            if (*p != '\0') END_MATCH();                                    \
        } while (0)

#define MATCH_ONE_OR_NONE(condition)                                        \
        do { if (condition) NEXT_CHAR(); } while (0)

#define MATCH_ONE_OR_MORE(condition)                                        \
        do {                                                                \
            ok = 0;                                                         \
            while (condition) { NEXT_CHAR(); ok = 1; }                      \
            if (!ok) END_MATCH();                                           \
        } while (0)

#define MATCH_ZERO_OR_MORE(condition)                                       \
        while (condition) { NEXT_CHAR(); }

    /* 1. emulate fscanf EOF handling */
    c = getc(fp);
    if (c == EOF) {
        return EOF;
    }
    /* 2. consume leading whitespace unconditionally */
    while (NumPyOS_ascii_isspace(c)) {
        c = getc(fp);
    }

    /* 3. start reading matching input to buffer */
    endp = buffer;

    /* 4.1 sign (optional) */
    MATCH_ONE_OR_NONE(c == '+' || c == '-');

    /* 4.2 nan, inf, infinity; [case-insensitive] */
    if (c == 'n' || c == 'N') {
        NEXT_CHAR();
        MATCH_ALPHA_STRING_NOCASE("an");

        /* accept nan([:alphanum:_]*), similarly to strtod */
        if (c == '(') {
            NEXT_CHAR();
            MATCH_ZERO_OR_MORE(NumPyOS_ascii_isalnum(c) || c == '_');
            if (c == ')') {
                NEXT_CHAR();
            }
        }
        END_MATCH();
    }
    else if (c == 'i' || c == 'I') {
        NEXT_CHAR();
        MATCH_ALPHA_STRING_NOCASE("nfinity");
        END_MATCH();
    }

    /* 4.3 mantissa */
    MATCH_ZERO_OR_MORE(NumPyOS_ascii_isdigit(c));

    if (c == '.') {
        NEXT_CHAR();
        MATCH_ONE_OR_MORE(NumPyOS_ascii_isdigit(c));
    }

    /* 4.4 exponent */
    if (c == 'e' || c == 'E') {
        NEXT_CHAR();
        MATCH_ONE_OR_NONE(c == '+' || c == '-');
        MATCH_ONE_OR_MORE(NumPyOS_ascii_isdigit(c));
    }

    END_MATCH();

buffer_filled:

    ungetc(c, fp);
    *endp = '\0';

    /* return 1 if something read, else 0 */
    return (buffer == endp) ? 0 : 1;
}

#undef END_MATCH
#undef NEXT_CHAR
#undef MATCH_ALPHA_STRING_NOCASE
#undef MATCH_ONE_OR_NONE
#undef MATCH_ONE_OR_MORE
#undef MATCH_ZERO_OR_MORE

/*
 * NumPyOS_ascii_ftolf:
 *      * fp: FILE pointer
 *      * value: Place to store the value read
 *
 * Similar to PyOS_ascii_strtod, except that it reads input from a file.
 *
 * Similarly to fscanf, this function always consumes leading whitespace,
 * and any text that could be the leading part in valid input.
 *
 * Return value: similar to fscanf.
 *      * 0 if no number read,
 *      * 1 if a number read,
 *      * EOF if end-of-file met before reading anything.
 */
NPY_NO_EXPORT int
NumPyOS_ascii_ftolf(FILE *fp, double *value)
{
    char buffer[FLOAT_FORMATBUFLEN + 1];
    char *p;
    int r;

    r = read_numberlike_string(fp, buffer, FLOAT_FORMATBUFLEN+1);

    if (r != EOF && r != 0) {
        *value = NumPyOS_ascii_strtod(buffer, &p);
        r = (p == buffer) ? 0 : 1;
    }
    return r;
}

NPY_NO_EXPORT int
NumPyOS_ascii_ftoLf(FILE *fp, long double *value)
{
    char buffer[FLOAT_FORMATBUFLEN + 1];
    char *p;
    int r;

    r = read_numberlike_string(fp, buffer, FLOAT_FORMATBUFLEN+1);

    if (r != EOF && r != 0) {
        *value = NumPyOS_ascii_strtold(buffer, &p);
        r = (p == buffer) ? 0 : 1;
    }
    return r;
}
