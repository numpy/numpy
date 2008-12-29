#include <locale.h>

/* From the C99 standard, section 7.19.6:
The exponent always contains at least two digits, and only as many more digits
as necessary to represent the exponent.
*/
#define MIN_EXPONENT_DIGITS 2

/* Ensure that any exponent, if present, is at least MIN_EXPONENT_DIGITS
   in length. */
static void ensure_minumim_exponent_length(char* buffer, size_t buf_size)
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

		/* Find the end of the exponent, keeping track of leading
		   zeros. */
		while (*p && isdigit(Py_CHARMASK(*p))) {
			if (in_leading_zeros && *p == '0')
				++leading_zero_cnt;
			if (*p != '0')
				in_leading_zeros = 0;
			++p;
			++exponent_digit_cnt;
		}

		significant_digit_cnt = exponent_digit_cnt - leading_zero_cnt;
		if (exponent_digit_cnt == MIN_EXPONENT_DIGITS) {
			/* If there are 2 exactly digits, we're done,
			   regardless of what they contain */
		}
		else if (exponent_digit_cnt > MIN_EXPONENT_DIGITS) {
			int extra_zeros_cnt;

			/* There are more than 2 digits in the exponent.  See
			   if we can delete some of the leading zeros */
			if (significant_digit_cnt < MIN_EXPONENT_DIGITS)
				significant_digit_cnt = MIN_EXPONENT_DIGITS;
			extra_zeros_cnt = exponent_digit_cnt -
				significant_digit_cnt;

			/* Delete extra_zeros_cnt worth of characters from the
			   front of the exponent */
			assert(extra_zeros_cnt >= 0);

			/* Add one to significant_digit_cnt to copy the
			   trailing 0 byte, thus setting the length */
			memmove(start,
				start + extra_zeros_cnt,
				significant_digit_cnt + 1);
		}
		else {
			/* If there are fewer than 2 digits, add zeros
			   until there are 2, if there's enough room */
			int zeros = MIN_EXPONENT_DIGITS - exponent_digit_cnt;
			if (start + zeros + exponent_digit_cnt + 1
			      < buffer + buf_size) {
				memmove(start + zeros, start,
					exponent_digit_cnt + 1);
				memset(start, '0', zeros);
			}
		}
	}
}

/* Ensure that buffer has a decimal point in it.  The decimal point
   will not be in the current locale, it will always be '.' */
void ensure_decimal_point(char* buffer, size_t buf_size)
{
	int insert_count = 0;
	char* chars_to_insert;

	/* search for the first non-digit character */
	char *p = buffer;
	if (*p == '-' || *p == '+')
		/* Skip leading sign, if present.  I think this could only
		   ever be '-', but it can't hurt to check for both. */
		++p;
	while (*p && isdigit(Py_CHARMASK(*p)))
		++p;

	if (*p == '.') {
		if (isdigit(Py_CHARMASK(*(p+1)))) {
			/* Nothing to do, we already have a decimal
			   point and a digit after it */
		}
		else {
			/* We have a decimal point, but no following
			   digit.  Insert a zero after the decimal. */
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
			/* If there is not enough room in the buffer
			   for the additional text, just skip it.  It's
			   not worth generating an error over. */
		}
		else {
			memmove(p + insert_count, p,
				buffer + strlen(buffer) - p + 1);
			memcpy(p, chars_to_insert, insert_count);
		}
	}
}

/* see FORMATBUFLEN in unicodeobject.c */
#define FLOAT_FORMATBUFLEN 120

/* Given a string that may have a decimal point in the current
   locale, change it back to a dot.  Since the string cannot get
   longer, no need for a maximum buffer size parameter. */
static void change_decimal_from_locale_to_dot(char* buffer)
{
	struct lconv *locale_data = localeconv();
	const char *decimal_point = locale_data->decimal_point;

	if (decimal_point[0] != '.' || decimal_point[1] != 0) {
		size_t decimal_point_len = strlen(decimal_point);

		if (*buffer == '+' || *buffer == '-')
			buffer++;
		while (isdigit(Py_CHARMASK(*buffer)))
			buffer++;
		if (strncmp(buffer, decimal_point, decimal_point_len) == 0) {
			*buffer = '.';
			buffer++;
			if (decimal_point_len > 1) {
				/* buffer needs to get smaller */
				size_t rest_len = strlen(buffer +
						     (decimal_point_len - 1));
				memmove(buffer,
					buffer + (decimal_point_len - 1),
					rest_len);
				buffer[rest_len] = 0;
			}
		}
	}
}
/**
 * NumPyOS_ascii_formatd:
 * @buffer: A buffer to place the resulting string in
 * @buf_size: The length of the buffer.
 * @format: The printf()-style format to use for the
 *          code to use for converting. 
 * @d: The #gdouble to convert
 *
 * This is similar to PyOS_ascii_formatd in python > 2.6, except that it does
 * not handle 'n'.
 *
 * Converts a #gdouble to a string, using the '.' as
 * decimal point. To format the number you pass in
 * a printf()-style format string. Allowed conversion
 * specifiers are 'e', 'E', 'f', 'F', 'g', 'G'.
 * 
 * 'Z' is the same as 'g', except it always has a decimal and
 *     at least one digit after the decimal.
 *
 * Return value: The pointer to the buffer with the converted string.
 **/
char* NumPyOS_ascii_formatd(char *buffer, size_t buf_size, const char *format, 
		   	    double val)
{
	char format_char;
	size_t format_len = strlen(format);

	/* Issue 2264: code 'Z' requires copying the format.  'Z' is 'g', but
	   also with at least one character past the decimal. */
	char tmp_format[FLOAT_FORMATBUFLEN];

	/* The last character in the format string must be the format char */
	format_char = format[format_len - 1];

	if (format[0] != '%') {
		return NULL;
	}

	/* I'm not sure why this test is here.  It's ensuring that the format
	   string after the first character doesn't have a single quote, a
	   lowercase l, or a percent. This is the reverse of the commented-out
	   test about 10 lines ago. */
	if (strpbrk(format + 1, "'l%")) {
		return NULL;
	}

	/* Also curious about this function is that it accepts format strings
	   like "%xg", which are invalid for floats.  In general, the
	   interface to this function is not very good, but changing it is
	   difficult because it's a public API. */

	if (!(format_char == 'e' || format_char == 'E' || 
	      format_char == 'f' || format_char == 'F' || 
	      format_char == 'g' || format_char == 'G' ||
	      format_char == 'Z')) {
		return NULL;
	}

	/* Map 'Z' format_char to 'g', by copying the format string and
	   replacing the final char with a 'g' */
	if (format_char == 'Z') {
		if (format_len + 1 >= sizeof(tmp_format)) {
			/* The format won't fit in our copy.  Error out.  In
			   practice, this will never happen and will be
			   detected by returning NULL */
			return NULL;
		}
		strcpy(tmp_format, format);
		tmp_format[format_len - 1] = 'g';
		format = tmp_format;
	}


	/* Have PyOS_snprintf do the hard work */
	PyOS_snprintf(buffer, buf_size, format, val);

	/* Do various fixups on the return string */

	/* Get the current locale, and find the decimal point string.
	   Convert that string back to a dot. */
	change_decimal_from_locale_to_dot(buffer);

	/* If an exponent exists, ensure that the exponent is at least
	   MIN_EXPONENT_DIGITS digits, providing the buffer is large enough
	   for the extra zeros.  Also, if there are more than
	   MIN_EXPONENT_DIGITS, remove as many zeros as possible until we get
	   back to MIN_EXPONENT_DIGITS */
	ensure_minumim_exponent_length(buffer, buf_size);

	/* If format_char is 'Z', make sure we have at least one character
	   after the decimal point (and make sure we have a decimal point). */
	if (format_char == 'Z') {
		ensure_decimal_point(buffer, buf_size);
	}
	ensure_decimal_point(buffer, buf_size);

	return buffer;
}

char* NumPyOS_ascii_formatf(char *buffer, size_t buf_size, const char *format, 
		   	    float val)
{
	char format_char;
	size_t format_len = strlen(format);

	/* Issue 2264: code 'Z' requires copying the format.  'Z' is 'g', but
	   also with at least one character past the decimal. */
	char tmp_format[FLOAT_FORMATBUFLEN];

	/* The last character in the format string must be the format char */
	format_char = format[format_len - 1];

	if (format[0] != '%') {
		return NULL;
	}

	/* I'm not sure why this test is here.  It's ensuring that the format
	   string after the first character doesn't have a single quote, a
	   lowercase l, or a percent. This is the reverse of the commented-out
	   test about 10 lines ago. */
	if (strpbrk(format + 1, "'l%")) {
		return NULL;
	}

	/* Also curious about this function is that it accepts format strings
	   like "%xg", which are invalid for floats.  In general, the
	   interface to this function is not very good, but changing it is
	   difficult because it's a public API. */

	if (!(format_char == 'e' || format_char == 'E' || 
	      format_char == 'f' || format_char == 'F' || 
	      format_char == 'g' || format_char == 'G' ||
	      format_char == 'Z')) {
		return NULL;
	}

	/* Map 'Z' format_char to 'g', by copying the format string and
	   replacing the final char with a 'g' */
	if (format_char == 'Z') {
		if (format_len + 1 >= sizeof(tmp_format)) {
			/* The format won't fit in our copy.  Error out.  In
			   practice, this will never happen and will be
			   detected by returning NULL */
			return NULL;
		}
		strcpy(tmp_format, format);
		tmp_format[format_len - 1] = 'g';
		format = tmp_format;
	}


	/* Have PyOS_snprintf do the hard work */
	PyOS_snprintf(buffer, buf_size, format, val);

	/* Do various fixups on the return string */

	/* Get the current locale, and find the decimal point string.
	   Convert that string back to a dot. */
	change_decimal_from_locale_to_dot(buffer);

	/* If an exponent exists, ensure that the exponent is at least
	   MIN_EXPONENT_DIGITS digits, providing the buffer is large enough
	   for the extra zeros.  Also, if there are more than
	   MIN_EXPONENT_DIGITS, remove as many zeros as possible until we get
	   back to MIN_EXPONENT_DIGITS */
	ensure_minumim_exponent_length(buffer, buf_size);

	/* If format_char is 'Z', make sure we have at least one character
	   after the decimal point (and make sure we have a decimal point). */
	if (format_char == 'Z') {
		ensure_decimal_point(buffer, buf_size);
	}
	ensure_decimal_point(buffer, buf_size);

	return buffer;
}

char* NumPyOS_ascii_formatl(char *buffer, size_t buf_size, const char *format, 
		   	    long double val)
{
	char format_char;
	size_t format_len = strlen(format);

	/* Issue 2264: code 'Z' requires copying the format.  'Z' is 'g', but
	   also with at least one character past the decimal. */
	char tmp_format[FLOAT_FORMATBUFLEN];

	/* The last character in the format string must be the format char */
	format_char = format[format_len - 1];

	if (format[0] != '%') {
		return NULL;
	}

	/* I'm not sure why this test is here.  It's ensuring that the format
	   string after the first character doesn't have a single quote, a
	   lowercase l, or a percent. This is the reverse of the commented-out
	   test about 10 lines ago. */
	if (strpbrk(format + 1, "'l%")) {
		return NULL;
	}

	/* Also curious about this function is that it accepts format strings
	   like "%xg", which are invalid for floats.  In general, the
	   interface to this function is not very good, but changing it is
	   difficult because it's a public API. */

	if (!(format_char == 'e' || format_char == 'E' || 
	      format_char == 'f' || format_char == 'F' || 
	      format_char == 'g' || format_char == 'G' ||
	      format_char == 'Z')) {
		return NULL;
	}

	/* Map 'Z' format_char to 'g', by copying the format string and
	   replacing the final char with a 'g' */
	if (format_char == 'Z') {
		if (format_len + 1 >= sizeof(tmp_format)) {
			/* The format won't fit in our copy.  Error out.  In
			   practice, this will never happen and will be
			   detected by returning NULL */
			return NULL;
		}
		strcpy(tmp_format, format);
		tmp_format[format_len - 1] = 'g';
		format = tmp_format;
	}


	/* Have PyOS_snprintf do the hard work */
	PyOS_snprintf(buffer, buf_size, format, val);

	/* Do various fixups on the return string */

	/* Get the current locale, and find the decimal point string.
	   Convert that string back to a dot. */
	change_decimal_from_locale_to_dot(buffer);

	/* If an exponent exists, ensure that the exponent is at least
	   MIN_EXPONENT_DIGITS digits, providing the buffer is large enough
	   for the extra zeros.  Also, if there are more than
	   MIN_EXPONENT_DIGITS, remove as many zeros as possible until we get
	   back to MIN_EXPONENT_DIGITS */
	ensure_minumim_exponent_length(buffer, buf_size);

	/* If format_char is 'Z', make sure we have at least one character
	   after the decimal point (and make sure we have a decimal point). */
	if (format_char == 'Z') {
		ensure_decimal_point(buffer, buf_size);
	}
	ensure_decimal_point(buffer, buf_size);

	return buffer;
}
