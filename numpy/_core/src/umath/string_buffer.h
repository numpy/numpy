#ifndef _NPY_CORE_SRC_UMATH_STRING_BUFFER_H_
#define _NPY_CORE_SRC_UMATH_STRING_BUFFER_H_

#include <Python.h>
#include <cstddef>
#include <wchar.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"
#include "stringdtype/utf8_utils.h"
#include "string_fastsearch.h"
#include "gil_utils.h"


/**
 * @internal
 * @enum ENCODING
 * @brief Enumeration for different text encodings.
 */
enum class ENCODING {
    ASCII,  ///< ASCII encoding
    UTF32,  ///< UTF-32 encoding
    UTF8    ///< UTF-8 encoding
};

/**
 * @file_internal
 * @enum IMPLEMENTED_UNARY_FUNCTIONS
 * @brief Enumeration for implemented unary functions.
 */
enum class IMPLEMENTED_UNARY_FUNCTIONS {
    ISALPHA,    ///< Checks if a character is alphabetic
    ISDECIMAL,  ///< Checks if a character is decimal
    ISDIGIT,    ///< Checks if a character is a digit
    ISSPACE,    ///< Checks if a character is whitespace
    ISALNUM,    ///< Checks if a character is alphanumeric
    ISLOWER,    ///< Checks if a character is lowercase
    ISUPPER,    ///< Checks if a character is uppercase
    ISTITLE,    ///< Checks if a string is titlecase
    ISNUMERIC,  ///< Checks if a character is numeric
    STR_LEN     ///< Returns the length of a string
};


/**
 * @file_internal
 * @brief Retrieves a Unicode character from a buffer based on
 * the specified encoding.
 *
 * This function reads a character from the given byte buffer and returns its
 * corresponding Unicode codepoint. The number of bytes consumed is also updated.
 *
 * @tparam enc The encoding type for the codepoint.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param buf Pointer to the byte buffer containing the encoded character.
 * @param bytes Pointer to an integer that will be updated with the number of bytes read.
 *
 * @return The Unicode codepoint as a npy_ucs4 value.
 */
template <ENCODING enc>
inline npy_ucs4
getchar(const unsigned char *buf, int *bytes);

template <>
inline npy_ucs4
getchar<ENCODING::ASCII>(const unsigned char *buf, int *bytes)
{
    *bytes = 1;
    return (npy_ucs4) *buf;
}

template <>
inline npy_ucs4
getchar<ENCODING::UTF32>(const unsigned char *buf, int *bytes)
{
    *bytes = 4;
    return *(npy_ucs4 *)buf;
}

template <>
inline npy_ucs4
getchar<ENCODING::UTF8>(const unsigned char *buf, int *bytes)
{
    Py_UCS4 codepoint;
    *bytes = static_cast<int>(utf8_char_to_ucs4_code(buf, &codepoint));
    return (npy_ucs4)codepoint;
}


/**
 * @file_internal
 * @brief Checks if a Unicode codepoint is an alphabetic character.
 *
 * @tparam enc The encoding type for the codepoint.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param code The Unicode codepoint to check.
 *
 * @return True if the codepoint is an alphabetic character; otherwise, false.
 */
template<ENCODING enc>
inline bool
codepoint_isalpha(npy_ucs4 code);

template<>
inline bool
codepoint_isalpha<ENCODING::ASCII>(npy_ucs4 code)
{
    return NumPyOS_ascii_isalpha((char)code);
}

template<>
inline bool
codepoint_isalpha<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISALPHA(code);
}

template<>
inline bool
codepoint_isalpha<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISALPHA(code);
}


/**
 * @file_internal
 * @brief Checks if a Unicode codepoint is a digit.
 *
 * @tparam enc The encoding type for the codepoint.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param code The Unicode codepoint to check.
 *
 * @return True if the codepoint is a digit; otherwise, false.
 */
template<ENCODING enc>
inline bool
codepoint_isdigit(npy_ucs4 code);

template<>
inline bool
codepoint_isdigit<ENCODING::ASCII>(npy_ucs4 code)
{
    return NumPyOS_ascii_isdigit((char)code);
}

template<>
inline bool
codepoint_isdigit<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISDIGIT(code);
}

template<>
inline bool
codepoint_isdigit<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISDIGIT(code);
}


/**
 * @file_internal
 * @brief Checks if the given Unicode codepoint is a whitespace character.
 *
 * @tparam enc The encoding type for the codepoint.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param code The Unicode codepoint to check.
 *
 * @return True if the codepoint is a whitespace character, false otherwise.
 */
template<ENCODING enc>
inline bool
codepoint_isspace(npy_ucs4 code);

template<>
inline bool
codepoint_isspace<ENCODING::ASCII>(npy_ucs4 code)
{
    return NumPyOS_ascii_isspace((char)code);
}

template<>
inline bool
codepoint_isspace<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISSPACE(code);
}

template<>
inline bool
codepoint_isspace<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISSPACE(code);
}


/**
 * @file_internal
 * @brief Checks if the given Unicode codepoint is alphanumeric.
 *
 * This function is a template that specializes its behavior based on the specified encoding.
 *
 * @tparam enc The encoding type for the codepoint.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param code The Unicode codepoint to check.
 *
 * @return True if the codepoint is alphanumeric, false otherwise.
 */
template<ENCODING enc>
inline bool
codepoint_isalnum(npy_ucs4 code);

template<>
inline bool
codepoint_isalnum<ENCODING::ASCII>(npy_ucs4 code)
{
    return NumPyOS_ascii_isalnum((char)code);
}

template<>
inline bool
codepoint_isalnum<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISALNUM(code);
}

template<>
inline bool
codepoint_isalnum<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISALNUM(code);
}


/**
 * @file_internal
 * @brief Checks if the given Unicode codepoint is a lowercase letter.
 *
 * @tparam enc The encoding type for the codepoint.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param code The Unicode codepoint to check.
 *
 * @return True if the codepoint is a lowercase letter, false otherwise.
 */
template<ENCODING enc>
inline bool
codepoint_islower(npy_ucs4 code);

template<>
inline bool
codepoint_islower<ENCODING::ASCII>(npy_ucs4 code)
{
    return NumPyOS_ascii_islower((char)code);
}

template<>
inline bool
codepoint_islower<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISLOWER(code);
}

template<>
inline bool
codepoint_islower<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISLOWER(code);
}


/**
 * @file_internal
 * @brief Checks if the given Unicode codepoint is an uppercase letter.
 *
 * @tparam enc The encoding type for the codepoint.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param code The Unicode codepoint to check.
 *
 * @return True if the codepoint is an uppercase letter, false otherwise.
 */
template<ENCODING enc>
inline bool
codepoint_isupper(npy_ucs4 code);

template<>
inline bool
codepoint_isupper<ENCODING::ASCII>(npy_ucs4 code)
{
    return NumPyOS_ascii_isupper((char)code);
}

template<>
inline bool
codepoint_isupper<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISUPPER(code);
}

template<>
inline bool
codepoint_isupper<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISUPPER(code);
}


/**
 * @file_internal
 * @brief Checks if the given Unicode codepoint is in title case.
 *
 * @tparam enc The character encoding to use for the check.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param code The Unicode codepoint to check.
 *
 * @return
 * - For `ENCODING::ASCII`: Always returns false, as ASCII
 *   characters do not have a concept of title case.
 * - For `ENCODING::UTF32` and `ENCODING::UTF8`: Returns
 *   the result of `Py_UNICODE_ISTITLE(code)`, which determines
 *   if the codepoint is in title case based on Unicode rules.
 */
template<ENCODING enc>
inline bool
codepoint_istitle(npy_ucs4 code);

template<>
inline bool
codepoint_istitle<ENCODING::ASCII>(npy_ucs4 code)
{
    return false;
}

template<>
inline bool
codepoint_istitle<ENCODING::UTF32>(npy_ucs4 code)
{
    return Py_UNICODE_ISTITLE(code);
}

template<>
inline bool
codepoint_istitle<ENCODING::UTF8>(npy_ucs4 code)
{
    return Py_UNICODE_ISTITLE(code);
}


/**
 * @file_internal
 * @brief Checks if the given Unicode codepoint is numeric.
 *
 * @param code The Unicode codepoint to check.
 *
 * @return true if the codepoint is a numeric character, false otherwise.
 */
inline bool
codepoint_isnumeric(npy_ucs4 code)
{
    return Py_UNICODE_ISNUMERIC(code);
}

/**
 * @file_internal
 * @brief Checks if the given Unicode codepoint is a decimal character.
 *
 * @param code The Unicode codepoint to check.
 *
 * @return true if the codepoint is a decimal digit, false otherwise.
 */
inline bool
codepoint_isdecimal(npy_ucs4 code)
{
    return Py_UNICODE_ISDECIMAL(code);
}


template <IMPLEMENTED_UNARY_FUNCTIONS f, ENCODING enc, typename T>
struct call_buffer_member_function;


/**
 * @internal
 * @brief A template struct representing a buffer for handling
 *        different character encodings.
 *
 * @tparam enc The encoding type (ASCII, UTF32, or UTF8).
 */
template <ENCODING enc>
struct Buffer {
    char *buf;    ///< Pointer to the start of the buffer.
    char *after;  ///< Pointer to the end of the buffer.

    /**
     * @brief Default constructor initializing the buffer pointers to NULL.
     */
    inline Buffer()
    {
        buf = after = NULL;
    }

    /**
     * @brief Constructs a Buffer with a specified pointer and element size.
     *
     * This constructor initializes the buffer, setting the starting
     * pointer and calculating the endpoint based on the given
     * element size.
     *
     * @param buf_ Pointer to the start of the buffer.
     * @param elsize_ Size in bytes of the buffer elements.
     */
    inline Buffer(char *buf_, npy_int64 elsize_)
    {
        buf = buf_;
        after = buf_ + elsize_;
    }

    /**
     * @brief Count the number of codepoints in the buffer.
     *
     * @return The number of codepoints.
     */
    inline size_t
    num_codepoints()
    {
        Buffer tmp(after, 0);
        size_t num_codepoints;
        switch (enc) {
            case ENCODING::ASCII:
            case ENCODING::UTF32:
            {
                tmp--;
                while (tmp >= *this && *tmp == '\0') {
                    tmp--;
                }
                num_codepoints = (size_t) (tmp - *this + 1);
                break;
            }
            case ENCODING::UTF8:
            {
                num_codepoints_for_utf8_bytes((unsigned char *)buf, &num_codepoints, (size_t)(after - buf));
            }
        }
        return num_codepoints;
    }

    /**
     * @brief Increment the buffer pointer by a specified number of positions.
     *
     * @param rhs The number of positions to move.
     *
     * @return A reference to the updated Buffer object.
     */
    inline Buffer<enc>&
    operator+=(npy_int64 rhs)
    {
        switch (enc) {
        case ENCODING::ASCII:
            buf += rhs;
            break;
        case ENCODING::UTF32:
            buf += rhs * sizeof(npy_ucs4);
            break;
        case ENCODING::UTF8:
            for (int i=0; i<rhs; i++) {
                buf += num_bytes_for_utf8_character((unsigned char *)buf);
            }
            break;
        }
        return *this;
    }

    /**
     * @brief Decrement the buffer pointer by a specified number of positions.
     *
     * @param rhs The number of positions to move back.
     *
     * @return A reference to the updated Buffer object.
     */
    inline Buffer<enc>&
    operator-=(npy_int64 rhs)
    {
        switch (enc) {
        case ENCODING::ASCII:
            buf -= rhs;
            break;
        case ENCODING::UTF32:
            buf -= rhs * sizeof(npy_ucs4);
            break;
        case ENCODING::UTF8:
            buf = (char *) find_previous_utf8_character((unsigned char *)buf, (size_t) rhs);
        }
        return *this;
    }

    /**
     * @brief Pre-increment the buffer pointer by one position.
     *
     * @return A reference to the updated Buffer object.
     */
    inline Buffer<enc>&
    operator++()
    {
        *this += 1;
        return *this;
    }

    /**
     * @brief Post-increment the buffer pointer by one position.
     *
     * @return A copy of the Buffer object before incrementing.
     */
    inline Buffer<enc>
    operator++(int)
    {
        Buffer<enc> old = *this;
        operator++();
        return old;
    }

    /**
     * @brief Pre-decrement the buffer pointer by one position.
     *
     * @return A reference to the updated Buffer object.
     */
    inline Buffer<enc>&
    operator--()
    {
        *this -= 1;
        return *this;
    }

    /**
     * @brief Post-decrement the buffer pointer by one position.
     *
     * @return A copy of the Buffer object before decrementing.
     */
    inline Buffer<enc>
    operator--(int)
    {
        Buffer<enc> old = *this;
        operator--();
        return old;
    }

    /**
     * @brief Dereference the buffer to access the current codepoint.
     *
     * @return The current codepoint.
     */
    inline npy_ucs4
    operator*()
    {
        int bytes;
        return getchar<enc>((unsigned char *) buf, &bytes);
    }

    /**
     * @brief Compare memory of two buffers.
     *
     * @param other The buffer to compare with.
     * @param len The number of bytes to compare:
     *            - For ASCII and UTF8, this is the number of bytes.
     *            - For UTF32, this is the number of characters.
     *
     * @return A negative value if the first buffer is less than the second,
     *         zero if they are equal, or a positive value if the first
     *         buffer is greater.
     */
    inline int
    buffer_memcmp(Buffer<enc> other, size_t len)
    {
        if (len == 0) {
            return 0;
        }
        switch (enc) {
            case ENCODING::ASCII:
            case ENCODING::UTF8:
                // note that len is in bytes for ASCII and UTF8 but
                // characters for UTF32
                return memcmp(buf, other.buf, len);
            case ENCODING::UTF32:
                return memcmp(buf, other.buf, len * sizeof(npy_ucs4));
        }
    }

    /**
     * @brief Copy memory from self to another buffer.
     *
     * @param other The destination buffer.
     * @param len The number of characters to copy:
     *            - For ASCII and UTF8, this is the number of bytes.
     *            - For UTF32, this is the number of characters.
     */
    inline void
    buffer_memcpy(Buffer<enc> other, size_t len)
    {
        if (len == 0) {
            return;
        }
        switch (enc) {
            case ENCODING::ASCII:
            case ENCODING::UTF8:
                // for UTF8 we treat n_chars as number of bytes
                memcpy(other.buf, buf, len);
                break;
            case ENCODING::UTF32:
                memcpy(other.buf, buf, len * sizeof(npy_ucs4));
                break;
        }
    }

    /**
     * @brief Set a range of memory in the buffer to a specific character.
     *
     * @param fill_char The character to fill the buffer with.
     * @param n_chars The number of characters to set.
     *
     * @return The number of characters set in the buffer.
     */
    inline npy_intp
    buffer_memset(npy_ucs4 fill_char, size_t n_chars)
    {
        if (n_chars == 0) {
            return 0;
        }
        switch (enc) {
            case ENCODING::ASCII:
                memset(this->buf, (char)fill_char, n_chars);
                return (npy_intp) n_chars;
            case ENCODING::UTF32:
            {
                char *tmp = this->buf;
                for (size_t i = 0; i < n_chars; i++) {
                    *(npy_ucs4 *)tmp = fill_char;
                    tmp += sizeof(npy_ucs4);
                }
                return (npy_intp) n_chars;
            }
            case ENCODING::UTF8:
            {
                char utf8_c[4] = {0};
                char *tmp = this->buf;
                size_t num_bytes = ucs4_code_to_utf8_char(fill_char, utf8_c);
                for (size_t i = 0; i < n_chars; i++) {
                    memcpy(tmp, utf8_c, num_bytes);
                    tmp += num_bytes;
                }
                return (npy_intp) (num_bytes * n_chars);
            }
        }
    }

    /**
     * @brief Fill the buffer with zeros starting from a specified index.
     *
     * This method fills the buffer with zero bytes starting from the specified
     * index (`start_index`) to the end of the buffer.
     *
     * @param start_index The index from which to start filling with zeros.
     */
    inline void
    buffer_fill_with_zeros_after_index(size_t start_index)
    {
        Buffer<enc> offset = *this + start_index;
        for (char *tmp = offset.buf; tmp < after; tmp++) {
            *tmp = 0;
        }
    }

    /**
     * @brief Advance the buffer pointer by a specified number of characters or bytes.
     *
     * This method adjusts the buffer pointer based on the encoding:
     * - For ASCII and UTF32, it advances by `n` characters.
     * - For UTF8, it advances by `n` bytes.
     *
     * @param n The number of characters (or bytes) to advance.
     */
    inline void
    advance_chars_or_bytes(size_t n) {
        switch (enc) {
            case ENCODING::ASCII:
            case ENCODING::UTF32:
                *this += n;
                break;
            case ENCODING::UTF8:
                this->buf += n;
                break;
        }
    }

    /**
     * @brief Get the number of bytes for the next character in the buffer.
     *
     * This function determines the number of bytes required to represent
     * the next character based on the current encoding.
     *
     * @return The number of bytes for the next character.
     */
    inline size_t
    num_bytes_next_character() {
        switch (enc) {
            case ENCODING::ASCII:
                return 1;
            case ENCODING::UTF32:
                return 4;
            case ENCODING::UTF8:
                return num_bytes_for_utf8_character((unsigned char *)(*this).buf);
        }
    }

    /**
     * @brief Apply a unary function to each codepoint in the buffer.
     *
     * This function iterates over each codepoint in the buffer and applies
     * the specified unary function. If the function fails for any codepoint,
     * the iteration stops and returns false.
     *
     * @tparam f The unary function to be applied to each codepoint.
     *
     * @return true if the function succeeds for all codepoints, false otherwise.
     */
    template<IMPLEMENTED_UNARY_FUNCTIONS f>
    inline bool
    unary_loop()
    {
        size_t len = num_codepoints();
        if (len == 0) {
            return false;
        }

        Buffer<enc> tmp = *this;

        for (size_t i=0; i<len; i++) {
            bool result;

            call_buffer_member_function<f, enc, bool> cbmf;

            result = cbmf(tmp);

            if (!result) {
                return false;
            }
            tmp++;
        }
        return true;
    }

    /**
     * @brief Check if all characters in the buffer are alphabetic.
     *
     * @return true if all characters are alphabetic, false otherwise.
     */
    inline bool
    isalpha()
    {
        return unary_loop<IMPLEMENTED_UNARY_FUNCTIONS::ISALPHA>();
    }

    /**
     * @brief Check if all characters in the buffer are decimal digits.
     *
     * @return true if all characters are decimal digits, false otherwise.
     */
    inline bool
    isdecimal()
    {
        return unary_loop<IMPLEMENTED_UNARY_FUNCTIONS::ISDECIMAL>();
    }

    /**
     * @brief Check if all characters in the buffer are digits.
     *
     * @return true if all characters are digits, false otherwise.
     */
    inline bool
    isdigit()
    {
        return unary_loop<IMPLEMENTED_UNARY_FUNCTIONS::ISDIGIT>();
    }

    /**
     * @brief Check if the first character in the buffer is a whitespace.
     *
     * @return true if the first character is whitespace, false otherwise.
     */
    inline bool
    first_character_isspace()
    {
        switch (enc) {
            case ENCODING::ASCII:
                return NumPyOS_ascii_isspace(**this);
            case ENCODING::UTF32:
            case ENCODING::UTF8:
                return Py_UNICODE_ISSPACE(**this);
        }
    }

    /**
     * @brief Check if all characters in the buffer are whitespace.
     *
     * @return true if all characters are whitespace, false otherwise.
     */
    inline bool
    isspace()
    {
        return unary_loop<IMPLEMENTED_UNARY_FUNCTIONS::ISSPACE>();
    }

    /**
     * @brief Check if all characters in the buffer are alphanumeric.
     *
     * @return true if all characters are alphanumeric, false otherwise.
     */
    inline bool
    isalnum()
    {
        return unary_loop<IMPLEMENTED_UNARY_FUNCTIONS::ISALNUM>();
    }

    /**
     * @brief Check if the buffer contains only lowercase letters.
     *
     * This function iterates through the characters in the buffer
     * and determines whether they are all lowercase letters. It
     * returns true if at least one lowercase letter is present and
     * no uppercase or title case letters are found.
     *
     * @return True if the buffer contains only lowercase letters;
     *         otherwise, false.
     */
    inline bool
    islower()
    {
        size_t len = num_codepoints();
        if (len == 0) {
            return false;
        }

        Buffer<enc> tmp = *this;
        bool cased = false;
        for (size_t i = 0; i < len; i++) {
            if (codepoint_isupper<enc>(*tmp) || codepoint_istitle<enc>(*tmp)) {
                return false;
            }
            else if (!cased && codepoint_islower<enc>(*tmp)) {
                cased = true;
            }
            tmp++;
        }
        return cased;
    }

    /**
     * @brief Check if the buffer contains only uppercase letters.
     *
     * This function iterates through the characters in the buffer
     * and determines whether they are all uppercase letters. It
     * returns true if at least one uppercase letter is present and
     * no lowercase or title case letters are found.
     *
     * @return True if the buffer contains only uppercase letters;
     *         otherwise, false.
     */
    inline bool
    isupper()
    {
        size_t len = num_codepoints();
        if (len == 0) {
            return false;
        }

        Buffer<enc> tmp = *this;
        bool cased = false;
        for (size_t i = 0; i < len; i++) {
            if (codepoint_islower<enc>(*tmp) || codepoint_istitle<enc>(*tmp)) {
                return false;
            }
            else if (!cased && codepoint_isupper<enc>(*tmp)) {
                cased = true;
            }
            tmp++;
        }
        return cased;
    }

    /**
     * @brief Check if the buffer follows title case conventions.
     *
     * This function checks if the characters in the buffer are
     * formatted as title case. A title case string has each word's
     * first character capitalized, while other characters in the
     * word are lowercase. This function returns true if the buffer
     * adheres to this rule, and false otherwise.
     *
     * @return True if the buffer is in title case; otherwise, false.
     */
    inline bool
    istitle()
    {
        size_t len = num_codepoints();
        if (len == 0) {
            return false;
        }

        Buffer<enc> tmp = *this;
        bool cased = false;
        bool previous_is_cased = false;
        for (size_t i = 0; i < len; i++) {
            if (codepoint_isupper<enc>(*tmp) || codepoint_istitle<enc>(*tmp)) {
                if (previous_is_cased) {
                    return false;
                }
                previous_is_cased = true;
                cased = true;
            }
            else if (codepoint_islower<enc>(*tmp)) {
                if (!previous_is_cased) {
                    return false;
                }
                cased = true;
            }
            else {
                previous_is_cased = false;
            }
            tmp++;
        }
        return cased;
    }

    /**
     * @brief Check if all characters in the buffer are numeric.
     *
     * @return True if all characters are numeric; otherwise, false.
     */
    inline bool
    isnumeric()
    {
        return unary_loop<IMPLEMENTED_UNARY_FUNCTIONS::ISNUMERIC>();
    }

    /**
     * @brief Remove trailing whitespace and null characters from the buffer.
     *
     * This function modifies the buffer by removing any trailing whitespace
     * (spaces, tabs, ...) and null characters ('\0').
     *
     * @return The modified buffer with trailing whitespace removed.
     */
    inline Buffer<enc>
    rstrip()
    {
        Buffer<enc> tmp(after, 0);

        tmp--;
        while (tmp >= *this && (*tmp == '\0' || NumPyOS_ascii_isspace(*tmp))) {
            tmp--;
        }
        tmp++;

        after = tmp.buf;
        return *this;
    }

    /**
     * @brief Compare two buffers for equality or ordering.
     *
     * This function compares the current buffer with another buffer (`other`),
     * optionally removing trailing whitespace from both buffers if `rstrip` is true.
     *
     * @param other The buffer to compare against.
     * @param rstrip If true, trailing whitespace will be removed from both buffers
     *               before comparison.
     *
     * @return
     *   - A negative integer if the current buffer is less than `other`.
     *   - A positive integer if the current buffer is greater than `other`.
     *   - Zero if both buffers are equal.
     */
    inline int
    strcmp(Buffer<enc> other, bool rstrip)
    {
        Buffer tmp1 = rstrip ? this->rstrip() : *this;
        Buffer tmp2 = rstrip ? other.rstrip() : other;

        while (tmp1.buf < tmp1.after && tmp2.buf < tmp2.after) {
            if (*tmp1 < *tmp2) {
                return -1;
            }
            if (*tmp1 > *tmp2) {
                return 1;
            }
            tmp1++;
            tmp2++;
        }
        while (tmp1.buf < tmp1.after) {
            if (*tmp1) {
                return 1;
            }
            tmp1++;
        }
        while (tmp2.buf < tmp2.after) {
            if (*tmp2) {
                return -1;
            }
            tmp2++;
        }
        return 0;
    }

    /**
     * @brief Compare two buffers for equality or ordering
     *        without trimming whitespace.
     *
     * This function compares the current buffer with another buffer (`other`)
     * to determine their relative order. It does not modify the buffers
     * by trimming whitespace.
     *
     * @param other The buffer to compare with.
     *
     * @return An integer less than, equal to, or greater than zero if the
     *         current buffer is found to be less than, to match, or
     *         be greater than the `other` buffer, respectively.
     */
    inline int
    strcmp(Buffer<enc> other)
    {
        return strcmp(other, false);
    }
};


/**
 * @file_internal
 * @brief Functor to apply a specified unary function.
 *
 * @tparam f The unary function to apply.
 *           Support `ISALPHA`, `ISDECIMAL`, `ISDIGIT`,
 *           `ISSPACE`, `ISALNUM`, `ISNUMERIC`.
 * @tparam enc The encoding type of the buffer.
 * @tparam T The return type of the unary function.
 */
template <IMPLEMENTED_UNARY_FUNCTIONS f, ENCODING enc, typename T>
struct call_buffer_member_function {
    /**
     * @brief Applies the specified unary function to the given buffer.
     *
     * @param buf The buffer to process.
     * @return The result of the unary function.
     */
    T operator()(Buffer<enc> buf) {
        static_assert(f == IMPLEMENTED_UNARY_FUNCTIONS::ISALPHA ||
                      f == IMPLEMENTED_UNARY_FUNCTIONS::ISDECIMAL ||
                      f == IMPLEMENTED_UNARY_FUNCTIONS::ISDIGIT ||
                      f == IMPLEMENTED_UNARY_FUNCTIONS::ISSPACE ||
                      f == IMPLEMENTED_UNARY_FUNCTIONS::ISALNUM ||
                      f == IMPLEMENTED_UNARY_FUNCTIONS::ISNUMERIC,
                      "Invalid IMPLEMENTED_UNARY_FUNCTIONS value "
                      "in call_buffer_member_function.");

        switch (f) {
            case IMPLEMENTED_UNARY_FUNCTIONS::ISALPHA:
                return codepoint_isalpha<enc>(*buf);
            case IMPLEMENTED_UNARY_FUNCTIONS::ISDECIMAL:
                return codepoint_isdecimal(*buf);
            case IMPLEMENTED_UNARY_FUNCTIONS::ISDIGIT:
                return codepoint_isdigit<enc>(*buf);
            case IMPLEMENTED_UNARY_FUNCTIONS::ISSPACE:
                return codepoint_isspace<enc>(*buf);
            case IMPLEMENTED_UNARY_FUNCTIONS::ISALNUM:
                return codepoint_isalnum<enc>(*buf);
            case IMPLEMENTED_UNARY_FUNCTIONS::ISNUMERIC:
                return codepoint_isnumeric(*buf);
        }
    }
};

/**
 * @internal
 * @brief Moves the buffer forward by a specified number of elements.
 *
 * @tparam enc The buffer's encoding type.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param lhs The original buffer.
 * @param rhs The number of elements to move forward.
 *
 * @return A new `Buffer<enc>` at the updated position.
 */
template <ENCODING enc>
inline Buffer<enc>
operator+(Buffer<enc> lhs, npy_int64 rhs)
{
    switch (enc) {
        case ENCODING::ASCII:
            return Buffer<enc>(lhs.buf + rhs, lhs.after - lhs.buf - rhs);
        case ENCODING::UTF32:
            return Buffer<enc>(lhs.buf + rhs * sizeof(npy_ucs4),
                          lhs.after - lhs.buf - rhs * sizeof(npy_ucs4));
        case ENCODING::UTF8:
            char* buf = lhs.buf;
            for (int i=0; i<rhs; i++) {
                buf += num_bytes_for_utf8_character((unsigned char *)buf);
            }
            return Buffer<enc>(buf, (npy_int64)(lhs.after - buf));
    }
}

/**
 * @internal
 * @brief Computes the difference between two buffers.
 *
 * This function returns the difference in bytes for UTF-8
 * buffers, and the difference in characters for ASCII and UTF-32
 * buffers. It is valid for UTF-8 only if both buffers point to
 * the same string.
 *
 * @tparam enc The buffer's encoding type.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param lhs The first buffer to compare.
 * @param rhs The second buffer to compare.
 *
 * @return The difference between the two buffers in bytes for
 *         UTF-8 and in characters for ASCII and UTF-32.
 *
 * @note For UTF-8 strings, the result is only meaningful if
 *       both buffers originate from the same string.
 */
template <ENCODING enc>
inline std::ptrdiff_t
operator-(Buffer<enc> lhs, Buffer<enc> rhs)
{
    switch (enc) {
    case ENCODING::ASCII:
    case ENCODING::UTF8:
        // note for UTF8 strings this is nonsense unless we're comparing
        // two points in the same string
        return lhs.buf - rhs.buf;
    case ENCODING::UTF32:
        return (lhs.buf - rhs.buf) / (std::ptrdiff_t) sizeof(npy_ucs4);
    }
}

/**
 * @internal
 * @brief Moves the buffer backward by a specified number of elements.
 *
 * @tparam enc The buffer's encoding type.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param lhs The original buffer.
 * @param rhs The number of elements to move backward.
 *
 * @return A new `Buffer<enc>` at the updated position.
 */
template <ENCODING enc>
inline Buffer<enc>
operator-(Buffer<enc> lhs, npy_int64 rhs)
{
    switch (enc) {
        case ENCODING::ASCII:
            return Buffer<enc>(lhs.buf - rhs, lhs.after - lhs.buf + rhs);
        case ENCODING::UTF32:
            return Buffer<enc>(lhs.buf - rhs * sizeof(npy_ucs4),
                          lhs.after - lhs.buf + rhs * sizeof(npy_ucs4));
        case ENCODING::UTF8:
            char* buf = lhs.buf;
            buf = (char *)find_previous_utf8_character((unsigned char *)buf, rhs);
            return Buffer<enc>(buf, (npy_int64)(lhs.after - buf));
    }

}

/**
 * @internal
 * @brief Compares two buffers for equality.
 *
 * @tparam enc The buffer's encoding type.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param lhs The first buffer.
 * @param rhs The second buffer.
 *
 * @return `true` if the buffers are equal, `false` otherwise.
 */
template <ENCODING enc>
inline bool
operator==(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return lhs.buf == rhs.buf;
}

/**
 * @internal
 * @brief Compares two buffers for inequality.
 *
 * @tparam enc The buffer's encoding type.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param lhs The first buffer.
 * @param rhs The second buffer.
 *
 * @return `true` if the buffers are not equal, `false` otherwise.
 */
template <ENCODING enc>
inline bool
operator!=(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return !(rhs == lhs);
}

/**
 * @internal
 * @brief Checks if the first buffer is less than the second.
 *
 * @tparam enc The buffer's encoding type.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param lhs The first buffer.
 * @param rhs The second buffer.
 *
 * @return `true` if the first buffer is less than the second,
 *         `false` otherwise.
 */
template <ENCODING enc>
inline bool
operator<(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return lhs.buf < rhs.buf;
}

/**
 * @internal
 * @brief Checks if the first buffer is greater than the second.
 *
 * @tparam enc The buffer's encoding type.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param lhs The first buffer.
 * @param rhs The second buffer.
 *
 * @return `true` if the first buffer is greater than the second,
 *         `false` otherwise.
 */
template <ENCODING enc>
inline bool
operator>(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return rhs < lhs;
}

/**
 * @internal
 * @brief Checks if the first buffer is less than or equal to the second.
 *
 * @tparam enc The buffer's encoding type.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param lhs The first buffer.
 * @param rhs The second buffer.
 *
 * @return `true` if the first buffer is less than or equal to the second,
 *         `false` otherwise.
 */
template <ENCODING enc>
inline bool
operator<=(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return !(lhs > rhs);
}

/**
 * @internal
 * @brief Checks if the first buffer is greater than or equal to the second.
 *
 * @tparam enc The buffer's encoding type.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param lhs The first buffer.
 * @param rhs The second buffer.
 *
 * @return `true` if the first buffer is greater than or equal to the second,
 *         `false` otherwise.
 */
template <ENCODING enc>
inline bool
operator>=(Buffer<enc> lhs, Buffer<enc> rhs)
{
    return !(lhs < rhs);
}

/*
 * Helper to fixup start/end slice values.
 *
 * This function is taken from CPython's Unicode module
 * (https://github.com/python/cpython/blob/0b718e6407da65b838576a2459d630824ca62155/Objects/bytes_methods.c#L495)
 * in order to remain compatible with how CPython handles
 * start/end arguments to str function like find/rfind etc.
 */

/**
 * @internal
 * @brief Adjusts the start and end offsets for slicing based on the buffer length.
 *
 * This function modifies the provided start and end indices to ensure they
 * are within the valid range for a buffer of the specified length. It handles
 * negative indices by wrapping them around and ensures that the end index
 * does not exceed the buffer length.
 *
 * @param start Pointer to the starting index to adjust.
 * @param end Pointer to the ending index to adjust.
 * @param len Length of the buffer.
 */
static inline void
adjust_offsets(npy_int64 *start, npy_int64 *end, size_t len)
{
    npy_int64 temp_len = static_cast<npy_int64>(len);

    if (*end > temp_len) {
        *end = temp_len;
    }
    else if (*end < 0) {
        *end += temp_len;
        if (*end < 0) {
            *end = 0;
        }
    }

    if (*start < 0) {
        *start += temp_len;
        if (*start < 0) {
            *start = 0;
        }
    }
}

/**
 * @internal
 * @brief Searches for the substring `buf2` within the string
 *        `buf1` in the specified range.
 *
 * - This function adjusts the start and end indices to ensure they are
 * within valid ranges.
 * - If the length of `buf2` exceeds the search range in `buf1`, the
 * function returns -1.
 * - If `buf2` is empty, the function returns the start index.
 *
 * @tparam enc The encoding type.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param buf1 The main buffer containing the string to search in.
 * @param buf2 The buffer containing the substring to search for.
 * @param start The starting index for the search in `buf1`.
 * @param end The ending index for the search in `buf1`.
 *
 * @return The index of the first occurrence of `buf2` in `buf1`,
 *         -1 if not found, or -2 if an error is raised.
 * @throws PyExc_ValueError If the string to search (in/for) exceeds the
 *         allowed maximum, the function will raise `PyExc_ValueError`
 *         with error message: "(target/pattern) string is too long".
 *
 * @note The search starts from the index specified by `start` (inclusive)
 *       and ends before the index specified by `end` (exclusive).
 */
template <ENCODING enc>
static inline npy_intp
string_find(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    npy_int64 len1 = buf1.num_codepoints();
    npy_int64 len2 = buf2.num_codepoints();
    if (len1 > PY_SSIZE_T_MAX || len1 < 0) {
        npy_gil_error(PyExc_ValueError, "target string is too long");
        return (npy_intp) -2;
    }
    if (len2 > PY_SSIZE_T_MAX || len2 < 0) {
        npy_gil_error(PyExc_ValueError, "pattern string is too long");
        return (npy_intp) -2;
    }

    adjust_offsets(&start, &end, len1);
    if (end - start < len2) {
        return (npy_intp) -1;
    }
    if (len2 == 0) {
        return (npy_intp) start;
    }

    char *start_loc = NULL;
    char *end_loc = NULL;
    if (enc == ENCODING::UTF8) {
        find_start_end_locs(buf1.buf, (buf1.after - buf1.buf), start, end,
                            &start_loc, &end_loc);
    }
    else {
        start_loc = (buf1 + start).buf;
        end_loc = (buf1 + end).buf;
    }

    if (len2 == 1) {
        npy_intp result;
        switch (enc) {
            case ENCODING::UTF8:
            {
                if (num_bytes_for_utf8_character((const unsigned char *)buf2.buf) > 1) {
                    goto multibyte_search;
                }
                // fall through to the ASCII case because this is a one-byte character
            }
            case ENCODING::ASCII:
            {
                char ch = *buf2;
                CheckedIndexer<char> ind(start_loc, end_loc - start_loc);
                result = (npy_intp) find_char(ind, end_loc - start_loc, ch);
                if (enc == ENCODING::UTF8 && result > 0) {
                    result = utf8_character_index(
                            start_loc, start_loc - buf1.buf, start, result,
                            buf1.after - start_loc);
                }
                break;
            }
            case ENCODING::UTF32:
            {
                npy_ucs4 ch = *buf2;
                CheckedIndexer<npy_ucs4> ind((npy_ucs4 *)(buf1 + start).buf, end-start);
                result = (npy_intp) find_char(ind, end - start, ch);
                break;
            }
        }
        if (result == -1) {
            return (npy_intp) -1;
        }
        else {
            return result + (npy_intp) start;
        }
    }

  multibyte_search:

    npy_intp pos;
    switch(enc) {
        case ENCODING::UTF8:
            pos = fastsearch(start_loc, end_loc - start_loc, buf2.buf,
                    buf2.after - buf2.buf, -1, FAST_SEARCH);
            // pos is the byte index, but we need the character index
            if (pos > 0) {
                pos = utf8_character_index(start_loc, start_loc - buf1.buf,
                        start, pos, buf1.after - start_loc);
            }
            break;
        case ENCODING::ASCII:
            pos = fastsearch(start_loc, end - start, buf2.buf, len2, -1, FAST_SEARCH);
            break;
        case ENCODING::UTF32:
            pos = fastsearch((npy_ucs4 *)start_loc, end - start,
                             (npy_ucs4 *)buf2.buf, len2, -1, FAST_SEARCH);
            break;
    }

    if (pos >= 0) {
        pos += start;
    }
    return pos;
}

/**
 * @internal
 * @brief Finds the starting position of a substring within a string.
 *
 * This function attempts to find the position of the substring
 * represented by `buf2` within the string represented by `buf1`,
 * between the `start` and `end` positions. If the substring is
 * not found, it raises a Python `ValueError` and returns -2 to
 * indicate an exception. Otherwise, it returns the index of the
 * substring.
 *
 * @tparam enc The encoding type.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param buf1 The buffer containing the string to search in.
 * @param buf2 The buffer containing the substring to search for.
 * @param start The starting position within `buf1` to begin searching.
 * @param end The ending position within `buf1` to stop searching.
 *
 * @return The starting position of the substring within the string,
 *         or -2 if not found or an error is raised.
 * @throws PyExc_ValueError If the substring is not found within the
 *         specified range, this function raises a `ValueError` with
 *         the error message: "substring not found".
 * @throws PyExc_ValueError If the string to search (in/for) exceeds the
 *         allowed maximum, the function will raise `PyExc_ValueError`
 *         with error message: "(target/pattern) string is too long".
 *
 * @note The search starts from the index specified by `start` (inclusive)
 *       and ends before the index specified by `end` (exclusive).
 */
template <ENCODING enc>
static inline npy_intp
string_index(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    npy_intp pos = string_find(buf1, buf2, start, end);
    if (pos == -1) {
        npy_gil_error(PyExc_ValueError, "substring not found");
        return -2;
    }
    return pos;
}

/**
 * @internal
 * @brief Searches for the substring `buf2` within the string
 *        `buf1` in the specified range, searching backwards.
 *
 * - This function adjusts the start and end indices to ensure they are
 * within valid ranges.
 * - If the length of `buf2` exceeds the search range in `buf1`, the
 * function returns -1.
 * - If `buf2` is empty, the function returns the end index.
 *
 * @tparam enc The encoding type.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param buf1 The main buffer containing the string to search in.
 * @param buf2 The buffer containing the substring to search for.
 * @param start The starting index for the search in `buf1`.
 * @param end The ending index for the search in `buf1`.
 *
 * @return The index of the last occurrence of `buf2` in `buf1`,
 *         -1 if not found, or -2 if an error is raised.
 * @throws PyExc_ValueError If the string to search (in/for) exceeds the
 *         allowed maximum, the function will raise `PyExc_ValueError`
 *         with error message: "(target/pattern) string is too long".
 *
 * @note The search starts from the index specified by `start` (inclusive)
 *       and ends before the index specified by `end` (exclusive).
 */
template <ENCODING enc>
static inline npy_intp
string_rfind(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    npy_int64 len1 = buf1.num_codepoints();
    npy_int64 len2 = buf2.num_codepoints();
    if (len1 > PY_SSIZE_T_MAX || len1 < 0) {
        npy_gil_error(PyExc_ValueError, "target string is too long");
        return (npy_intp) -2;
    }
    if (len2 > PY_SSIZE_T_MAX || len2 < 0) {
        npy_gil_error(PyExc_ValueError, "pattern string is too long");
        return (npy_intp) -2;
    }

    adjust_offsets(&start, &end, len1);
    if (end - start < len2) {
        return (npy_intp) -1;
    }

    if (len2 == 0) {
        return (npy_intp) end;
    }

    char *start_loc = NULL;
    char *end_loc = NULL;
    if (enc == ENCODING::UTF8) {
        find_start_end_locs(buf1.buf, (buf1.after - buf1.buf), start, end,
                            &start_loc, &end_loc);
    }
    else {
        start_loc = (buf1 + start).buf;
        end_loc = (buf1 + end).buf;
    }

    if (len2 == 1) {
        npy_intp result;
        switch (enc) {
            case ENCODING::UTF8:
            {
                if (num_bytes_for_utf8_character((const unsigned char *)buf2.buf) > 1) {
                    goto multibyte_search;
                }
                // fall through to the ASCII case because this is a one-byte character
            }
            case ENCODING::ASCII:
            {
                char ch = *buf2;
                CheckedIndexer<char> ind(start_loc, end_loc - start_loc);
                result = (npy_intp) rfind_char(ind, end_loc - start_loc, ch);
                if (enc == ENCODING::UTF8 && result > 0) {
                    result = utf8_character_index(
                            start_loc, start_loc - buf1.buf, start, result,
                            buf1.after - start_loc);
                }
                break;
            }
            case ENCODING::UTF32:
            {
                npy_ucs4 ch = *buf2;
                CheckedIndexer<npy_ucs4> ind((npy_ucs4 *)(buf1 + start).buf, end - start);
                result = (npy_intp) rfind_char(ind, end - start, ch);
                break;
            }
        }
        if (result == -1) {
            return (npy_intp) -1;
        }
        else {
            return result + (npy_intp) start;
        }
    }

  multibyte_search:

    npy_intp pos;
    switch (enc) {
        case ENCODING::UTF8:
            pos = fastsearch(start_loc, end_loc - start_loc,
                    buf2.buf, buf2.after - buf2.buf, -1, FAST_RSEARCH);
            // pos is the byte index, but we need the character index
            if (pos > 0) {
                pos = utf8_character_index(start_loc, start_loc - buf1.buf,
                        start, pos, buf1.after - start_loc);
            }
            break;
        case ENCODING::ASCII:
            pos = (npy_intp) fastsearch(start_loc, end - start, buf2.buf, len2, -1, FAST_RSEARCH);
            break;
        case ENCODING::UTF32:
            pos = (npy_intp) fastsearch((npy_ucs4 *)start_loc, end - start,
                                        (npy_ucs4 *)buf2.buf, len2, -1, FAST_RSEARCH);
            break;
    }
    if (pos >= 0) {
        pos += start;
    }
    return pos;
}

/**
 * @internal
 * @brief Finds the last occurrence of a substring within a specified range.
 *
 * This function searches for the last occurrence of the substring `buf2`
 * within the buffer `buf1` starting from the specified `start` position
 * and ending at the `end` position. If the substring is not found, it
 * raises a Python `ValueError` and returns -2 to indicate an exception.
 * Otherwise, it returns the index of the substring.
 *
 * @tparam enc The encoding type.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param buf1 The buffer containing the string to search in.
 * @param buf2 The buffer containing the substring to search for.
 * @param start The starting position within `buf1` to begin searching.
 * @param end The ending position within `buf1` to stop searching.
 *
 * @return The index of the last occurrence of the substring,
 *         or -2 if not found.
 * @throws PyExc_ValueError If the substring is not found within the
 *         specified range, this function raises a `ValueError` with
 *         the error message: "substring not found".
 * @throws PyExc_ValueError If the string to search (in/for) exceeds the
 *         allowed maximum, the function will raise `PyExc_ValueError`
 *         with error message: "(target/pattern) string is too long".
 *
 * @note The search starts from the index specified by `start` (inclusive)
 *       and ends before the index specified by `end` (exclusive).
 */
template <ENCODING enc>
static inline npy_intp
string_rindex(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    npy_intp pos = string_rfind(buf1, buf2, start, end);
    if (pos == -1) {
        npy_gil_error(PyExc_ValueError, "substring not found");
        return -2;
    }
    return pos;
}

/**
 * @internal
 * @brief Count the occurrences of a substring within a specified range.
 *
 * This function counts how many times the substring `buf2` appears
 * within the buffer `buf1` between the specified `start` and `end`
 * indices. This function adjusts the start and end indices to ensure
 * they are within valid ranges.
 *
 * @tparam enc The encoding type of the buffers.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param buf1 The buffer to search within.
 * @param buf2 The substring to search for.
 * @param start The starting index for the search (inclusive).
 * @param end The ending index for the search (exclusive).
 *
 * @return The number of occurrences of the substring, or -2
 * @throws PyExc_ValueError If the string to search (in/for) exceeds the
 *         allowed maximum, the function will raise `PyExc_ValueError`
 *         with error message: "(target/pattern) string is too long".
 *
 * @note The search starts from the index specified by `start` (inclusive)
 *       and ends before the index specified by `end` (exclusive).
 */
template <ENCODING enc>
static inline npy_intp
string_count(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end)
{
    npy_int64 len1 = buf1.num_codepoints();
    npy_int64 len2 = buf2.num_codepoints();
    if (len1 > PY_SSIZE_T_MAX || len1 < 0) {
        npy_gil_error(PyExc_ValueError, "target string is too long");
        return (npy_intp) -2;
    }
    if (len2 > PY_SSIZE_T_MAX || len2 < 0) {
        npy_gil_error(PyExc_ValueError, "pattern string is too long");
        return (npy_intp) -2;
    }

    adjust_offsets(&start, &end, len1);
    if (end < start || end - start < len2) {
        return (npy_intp) 0;
    }

    if (len2 == 0) {
        return (end - start) < PY_SSIZE_T_MAX ? end - start + 1 : PY_SSIZE_T_MAX;
    }

    char *start_loc = NULL;
    char *end_loc = NULL;
    if (enc == ENCODING::UTF8) {
        find_start_end_locs(buf1.buf, (buf1.after - buf1.buf), start, end,
                            &start_loc, &end_loc);
    }
    else {
        start_loc = (buf1 + start).buf;
        end_loc = (buf1 + end).buf;
    }

    npy_intp count = 0;
    switch (enc) {
        case ENCODING::UTF8:
            count = fastsearch(start_loc, end_loc - start_loc, buf2.buf,
                               buf2.after - buf2.buf, PY_SSIZE_T_MAX,
                               FAST_COUNT);
            break;
        case ENCODING::ASCII:
            count = (npy_intp) fastsearch(start_loc, end - start, buf2.buf, len2,
                                          PY_SSIZE_T_MAX, FAST_COUNT);
            break;
        case ENCODING::UTF32:
            count = (npy_intp) fastsearch((npy_ucs4 *)start_loc, end - start,
                                          (npy_ucs4 *)buf2.buf, len2,
                                          PY_SSIZE_T_MAX, FAST_COUNT);
            break;
    }

    if (count < 0) {
        return 0;
    }
    return count;
}


/**
 * @internal
 * @enum STRING_SIDE
 * @brief Enumeration for the starting position of the search.
 */
enum class STRING_SIDE {
    FRONT, ///< Start from the front of the buffer.
    BACK   ///< Start from the back of the buffer.
};

/**
 * @internal
 * @brief Check if the specified buffer ends with the given substring.
 *
 * This function adjusts the start and end indices to ensure they are
 * within valid ranges. The search can be performed from either the
 * front or the back of the buffer depending on the specified direction.
 * This function adjusts the start and end indices to ensure they are
 * within valid ranges. If `buf2` is empty, the function returns `1`.
 *
 * @tparam enc The encoding type of the buffers.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param buf1 The buffer to check against.
 * @param buf2 The substring to check for.
 * @param start The starting index for the search (inclusive).
 * @param end The ending index for the search (exclusive).
 * @param direction The direction to search from (either front or back).
 *
 * @return `npy_bool` indicating whether `buf1` ends with `buf2`.
 *         It will return `NPY_FALSE` if an error is raised.
 * @throws PyExc_ValueError If the string to search (in/for) exceeds the
 *         allowed maximum, the function will raise `PyExc_ValueError`
 *         with error message: "(target/pattern) string is too long".
 *
 * @note The search starts from the index specified by `start` (inclusive)
 *       and ends before the index specified by `end` (exclusive).
 */
template <ENCODING enc>
inline npy_bool
tail_match(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 start, npy_int64 end,
          STRING_SIDE direction)
{
    npy_int64 len1 = buf1.num_codepoints();
    npy_int64 len2 = buf2.num_codepoints();
    if (len1 > PY_SSIZE_T_MAX || len1 < 0) {
        npy_gil_error(PyExc_ValueError, "target string is too long");
        return NPY_FALSE;
    }
    if (len2 > PY_SSIZE_T_MAX || len2 < 0) {
        npy_gil_error(PyExc_ValueError, "pattern string is too long");
        return NPY_FALSE;
    }

    adjust_offsets(&start, &end, len1);
    end -= len2;
    if (end < start) {
        return NPY_FALSE;
    }

    if (len2 == 0) {
        return NPY_TRUE;
    }

    size_t offset;
    size_t end_sub = len2 - 1;
    if (direction == STRING_SIDE::BACK) {
        // The end index has been adjusted by subtracting the length of buf2.
        offset = end;
    }
    else {
        offset = start;
    }

    size_t size2 = len2;
    if (enc == ENCODING::UTF8) {
        size2 = (buf2.after - buf2.buf);
    }

    Buffer start_buf = (buf1 + offset);
    Buffer end_buf = start_buf + end_sub;
    if (*start_buf == *buf2 && *end_buf == *(buf2 + end_sub)) {
        return !start_buf.buffer_memcmp(buf2, size2);
    }

    return NPY_FALSE;
}


/**
 * @internal
 * @enum STRIP_TYPE
 * @brief Enumeration for string strip operations.
 *
 * This enum class specifies the type of string stripping to be
 * performed.
 */
enum class STRIP_TYPE {
    LEFT_STRIP,  ///< Remove whitespace or specified characters
                 ///< from the left side of the string.
    RIGHT_STRIP, ///< Remove whitespace or specified characters
                 ///< from the right side of the string.
    BOTH_STRIP   ///< Remove whitespace or specified characters
                 ///< from both sides of the string.
};

/**
 * @internal
 * @brief Strip whitespace characters from the given buffer.
 *
 * This function removes whitespace characters from the specified
 * buffer based on the defined strip type. It can remove
 * characters from the left, right, or both sides of the buffer.
 *
 * @tparam enc The encoding type of the buffer.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param buf The input buffer from which whitespace will be stripped.
 * @param out The output buffer to hold the result.
 * @param strip_type The type of strip operation to perform (left, right,
 *                   or both).
 *
 * @return The number of bytes written to the output buffer for UTF-8,
 *         and of characters for ASCII and UTF32.
 *
 * @note If the input buffer is empty, the output buffer will be filled
 *       with zeros (if not UTF8), and the function will return 0.
 */
template <ENCODING enc>
static inline size_t
string_strip_whitespace(Buffer<enc> buf, Buffer<enc> out, STRIP_TYPE strip_type)
{
    size_t len = buf.num_codepoints();
    if (len == 0) {
        if (enc != ENCODING::UTF8) {
            out.buffer_fill_with_zeros_after_index(0);
        }
        return 0;
    }

    size_t new_start = 0;

    size_t num_bytes = (buf.after - buf.buf);
    Buffer traverse_buf = Buffer<enc>(buf.buf, num_bytes);

    if (strip_type != STRIP_TYPE::RIGHT_STRIP) {
        while (new_start < len) {
            if (!traverse_buf.first_character_isspace()) {
                break;
            }
            num_bytes -= traverse_buf.num_bytes_next_character();
            new_start++;
            traverse_buf++;  // may go one beyond buffer
        }
    }

    size_t new_stop = len;  // New stop is a range (beyond last char)

    if (strip_type != STRIP_TYPE::LEFT_STRIP) {
        if (enc == ENCODING::UTF8) {
            traverse_buf = Buffer<enc>(buf.after, 0) - 1;
        }
        else {
            traverse_buf = buf + (new_stop - 1);
        }
        while (new_stop > new_start) {
            if (*traverse_buf != 0 && !traverse_buf.first_character_isspace()) {
                break;
            }

            num_bytes -= traverse_buf.num_bytes_next_character();
            new_stop--;

            // Do not step to character -1: can't find its start for utf-8.
            if (new_stop > 0) {
                traverse_buf--;
            }
        }
    }

    Buffer offset_buf = buf + new_start;
    if (enc == ENCODING::UTF8) {
        offset_buf.buffer_memcpy(out, num_bytes);
        return num_bytes;
    }
    offset_buf.buffer_memcpy(out, new_stop - new_start);
    out.buffer_fill_with_zeros_after_index(new_stop - new_start);
    return new_stop - new_start;
}

/**
 * @internal
 * @brief Strips characters from the given buffer.
 *
 * This function removes characters specified in the `buf2` from the
 * `buf1` based on the provided `strip_type`. The result is written
 * to the `out` buffer. It can remove characters from the left, right,
 * or both sides of the buffer.
 *
 * @tparam enc The encoding type of the buffer.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param buf1 The input buffer from which characters will be stripped.
 * @param buf2 The buffer containing characters to be stripped from `buf1`.
 * @param out The output buffer where the result will be stored.
 * @param strip_type The type of stripping to be performed (left, right, or both).
 *
 * @return The length of the resulting string in the output buffer.
 *
 * @note If the encoding is not UTF8, the output buffer will be filled
 * with zeros after the index of the resulting string.
 */
template <ENCODING enc>
static inline size_t
string_strip_chars(Buffer<enc> buf1, Buffer<enc> buf2, Buffer<enc> out, STRIP_TYPE strip_type)
{
    size_t len1 = buf1.num_codepoints();
    if (len1 == 0) {
        if (enc != ENCODING::UTF8) {
            out.buffer_fill_with_zeros_after_index(0);
        }
        return 0;
    }

    size_t len2 = buf2.num_codepoints();
    if (len2 == 0) {
        if (enc == ENCODING::UTF8) {
            buf1.buffer_memcpy(out, (buf1.after - buf1.buf));
            return buf1.after - buf1.buf;
        }
        buf1.buffer_memcpy(out, len1);
        out.buffer_fill_with_zeros_after_index(len1);
        return len1;
    }

    size_t new_start = 0;

    size_t num_bytes = (buf1.after - buf1.buf);
    Buffer traverse_buf = Buffer<enc>(buf1.buf, num_bytes);

    if (strip_type != STRIP_TYPE::RIGHT_STRIP) {
        for (; new_start < len1; traverse_buf++) {
            Py_ssize_t res = 0;
            size_t current_point_bytes = traverse_buf.num_bytes_next_character();
            switch (enc) {
                case ENCODING::ASCII:
                {
                    CheckedIndexer<char> ind(buf2.buf, len2);
                    res = find_char<char>(ind, len2, *traverse_buf);
                    break;
                }
                case ENCODING::UTF8:
                {
                    if (current_point_bytes == 1) {
                        CheckedIndexer<char> ind(buf2.buf, len2);
                        res = find_char<char>(ind, len2, *traverse_buf);
                    } else {
                        res = fastsearch(buf2.buf, buf2.after - buf2.buf,
                                traverse_buf.buf, current_point_bytes,
                                -1, FAST_SEARCH);
                    }
                    break;
                }
                case ENCODING::UTF32:
                {
                    CheckedIndexer<npy_ucs4> ind((npy_ucs4 *)buf2.buf, len2);
                    res = find_char<npy_ucs4>(ind, len2, *traverse_buf);
                    break;
                }
            }
            if (res < 0) {
                break;
            }
            num_bytes -= traverse_buf.num_bytes_next_character();
            new_start++;
        }
    }

    size_t new_stop = len1;  // New stop is a range (beyond last char)

    if (strip_type != STRIP_TYPE::LEFT_STRIP) {
        if (enc == ENCODING::UTF8) {
            traverse_buf = Buffer<enc>(buf1.after, 0) - 1;
        }
        else {
            traverse_buf = buf1 + (new_stop - 1);
        }
        while (new_stop > new_start) {
            size_t current_point_bytes = traverse_buf.num_bytes_next_character();
            Py_ssize_t res = 0;
            switch (enc) {
                case ENCODING::ASCII:
                {
                    CheckedIndexer<char> ind(buf2.buf, len2);
                    res = find_char<char>(ind, len2, *traverse_buf);
                    break;
                }
                case ENCODING::UTF8:
                {
                    if (current_point_bytes == 1) {
                        CheckedIndexer<char> ind(buf2.buf, len2);
                        res = find_char<char>(ind, len2, *traverse_buf);
                    } else {
                        res = fastsearch(buf2.buf, buf2.after - buf2.buf,
                                traverse_buf.buf, current_point_bytes,
                                -1, FAST_RSEARCH);
                    }
                    break;
                }
                case ENCODING::UTF32:
                {
                    CheckedIndexer<npy_ucs4> ind((npy_ucs4 *)buf2.buf, len2);
                    res = find_char<npy_ucs4>(ind, len2, *traverse_buf);
                    break;
                }
            }
            if (res < 0) {
                break;
            }
            num_bytes -= current_point_bytes;;
            new_stop--;
            // Do not step to character -1: can't find it's start for utf-8.
            if (new_stop > 0) {
                traverse_buf--;
            }
        }
    }

    Buffer offset_buf = buf1 + new_start;
    if (enc == ENCODING::UTF8) {
        offset_buf.buffer_memcpy(out, num_bytes);
        return num_bytes;
    }
    offset_buf.buffer_memcpy(out, new_stop - new_start);
    out.buffer_fill_with_zeros_after_index(new_stop - new_start);
    return new_stop - new_start;
}


/**
 * @file_internal
 * @brief Finds the index of the first occurrence of a slice to be replaced.
 *
 * This function searches for a slice (specified by `buf2`) in another buffer
 * (specified by `buf1`). It returns the index of the first occurrence of the
 * slice in the buffer.
 *
 * @tparam char_type The character type of the buffers.
 * @param buf1 The input buffer where the search will be performed.
 * @param len1 The length of the input buffer.
 * @param buf2 The buffer containing the slice to be found.
 * @param len2 The length of the slice buffer.
 *
 * @return The index of the first occurrence of the slice in the input buffer,
 *         or 0 if the slice buffer is empty.
 */
template <typename char_type>
static inline npy_intp
find_slice_for_replace(CheckedIndexer<char_type> buf1, npy_intp len1,
                       CheckedIndexer<char_type> buf2, npy_intp len2)
{
    if (len2 == 0) {
        return 0;
    }
    if (len2 == 1) {
        return (npy_intp) find_char(buf1, len1, *buf2);
    }
    return (npy_intp) fastsearch(buf1.buffer, len1, buf2.buffer, len2,
            -1, FAST_SEARCH);
}

/**
 * @internal
 * @brief Replaces occurrences of a substring in a buffer with another substring.
 *
 * This function searches for a specified substring (buf2) in an input buffer
 * (buf1) and replaces it with another substring (buf3). The number of
 * replacements is limited by the count parameter.
 *
 * @tparam enc The encoding type of the buffer.
 *             Supported encodings are ASCII, UTF32, and UTF8.
 * @param buf1 The input buffer where replacements will occur.
 * @param buf2 The buffer containing the substring to be replaced.
 * @param buf3 The buffer containing the substring to replace with.
 * @param count The maximum number of replacements to perform.
 * @param out The output buffer where the result will be stored.
 *
 * @return The total length of the resulting string in the output buffer.
 */
template <ENCODING enc>
static inline size_t
string_replace(Buffer<enc> buf1, Buffer<enc> buf2, Buffer<enc> buf3, npy_int64 count,
               Buffer<enc> out)
{
    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();
    size_t len3 = buf3.num_codepoints();

    char *start;
    if (enc == ENCODING::UTF8) {
        start = buf1.after;
    }
    else if (enc == ENCODING::UTF32) {
        start = buf1.buf + sizeof(npy_ucs4) * len1;
    }
    else {
        start = buf1.buf + len1;
    }
    Buffer<enc> end1(start, 0);
    size_t span2, span3;

    switch(enc) {
        case ENCODING::ASCII:
        case ENCODING::UTF32:
        {
            span2 = len2;
            span3 = len3;
            break;
        }
        case ENCODING::UTF8:
        {
            span2 = buf2.after - buf2.buf;
            span3 = buf3.after - buf3.buf;
            break;
        }
    }

    size_t ret = 0;

    // Only try to replace if replacements are possible.
    if (count <= 0                      // There's nothing to replace.
        || len1 < len2                  // Input is too small to have a match.
        || (len2 <= 0 && len3 <= 0)     // Match and replacement strings both empty.
        || (len2 == len3 && buf2.strcmp(buf3) == 0)) {  // Match and replacement are the same.

        goto copy_rest;
    }

    if (len2 > 0) {
        for (npy_int64 time = 0; time < count; time++) {
            npy_intp pos;
            switch (enc) {
                case ENCODING::ASCII:
                case ENCODING::UTF8:
                {
                    CheckedIndexer<char> ind1(buf1.buf, end1 - buf1);
                    CheckedIndexer<char> ind2(buf2.buf, span2);
                    pos = find_slice_for_replace(ind1, end1 - buf1, ind2, span2);
                    break;
                }
                case ENCODING::UTF32:
                {
                    CheckedIndexer<npy_ucs4> ind1((npy_ucs4 *)buf1.buf, end1 - buf1);
                    CheckedIndexer<npy_ucs4> ind2((npy_ucs4 *)buf2.buf, span2);
                    pos = find_slice_for_replace(ind1, end1 - buf1, ind2, span2);
                    break;
                }
            }
            if (pos < 0) {
                break;
            }

            buf1.buffer_memcpy(out, pos);
            ret += pos;
            out.advance_chars_or_bytes(pos);
            buf1.advance_chars_or_bytes(pos);

            buf3.buffer_memcpy(out, span3);
            ret += span3;
            out.advance_chars_or_bytes(span3);
            buf1.advance_chars_or_bytes(span2);
        }
    }
    else {  // If match string empty, interleave.
        while (count > 0) {
            buf3.buffer_memcpy(out, span3);
            ret += span3;
            out.advance_chars_or_bytes(span3);

            if (--count <= 0) {
                break;
            }

            switch (enc) {
                case ENCODING::ASCII:
                case ENCODING::UTF32:
                    buf1.buffer_memcpy(out, 1);
                    ret += 1;
                    break;
                case ENCODING::UTF8:
                    size_t n_bytes = buf1.num_bytes_next_character();
                    buf1.buffer_memcpy(out, n_bytes);
                    ret += n_bytes;
                    break;
            }
            buf1 += 1;
            out += 1;
        }
    }

copy_rest:
    buf1.buffer_memcpy(out, end1 - buf1);
    ret += end1 - buf1;
    if (enc == ENCODING::UTF8) {
        return ret;
    }
    out.buffer_fill_with_zeros_after_index(end1 - buf1);
    return ret;
}


/**
 * @internal
 * @brief Computes the length of a string after expanding tabs into spaces.
 *
 * This function calculates the length of a string represented
 * by a `Buffer<enc>`, after replacing tab characters ('\t') with
 * spaces.
 *
 * @tparam enc The encoding of the input string,
 *             which is specified by the `Buffer<enc>` type.
 * @param buf A buffer containing the input string, represented as a
 *            `Buffer<enc>`, where `enc` indicates the string encoding.
 * @param tabsize The number of spaces used to replace a tab character
 *                ('\t'). If `tabsize` is zero or less, tabs are not expanded.
 *
 * @return The length of the string after expanding tabs into spaces.
 *         If the computed length exceeds, the function raises a
 *         `PyExc_OverflowError` and returns -1.
 * @throws PyExc_OverflowError If the resulting string length exceeds,
 *         the function will raise `PyExc_OverflowError` with error message:
 *         "new string is too long".
 */
template <ENCODING enc>
static inline npy_intp
string_expandtabs_length(Buffer<enc> buf, npy_int64 tabsize)
{
    size_t len = buf.num_codepoints();

    npy_intp new_len = 0, line_pos = 0;

    Buffer<enc> tmp = buf;
    for (size_t i = 0; i < len; i++) {
        npy_ucs4 ch = *tmp;
        if (ch == '\t') {
            if (tabsize > 0) {
                npy_intp incr = tabsize - (line_pos % tabsize);
                line_pos += incr;
                new_len += incr;
            }
        }
        else {
            line_pos += 1;
            size_t n_bytes = tmp.num_bytes_next_character();
            new_len += n_bytes;
            if (ch == '\n' || ch == '\r') {
                line_pos = 0;
            }
        }
        if (new_len < 0) {
            npy_gil_error(PyExc_OverflowError, "new string is too long");
            return -1;
        }
        tmp++;
    }
    return new_len;
}

/**
 * @internal
 * @brief Expands tab characters ('\t') in a string buffer to
 *        spaces and writes the result to an output buffer.
 *
 * This function replaces each tab character ('\t') in the input
 * buffer (`buf`) with spaces, as determined by the `tabsize` parameter,
 * and writes the expanded string into the output buffer (`out`).
 *
 * @tparam enc The encoding of the input and output string,
 *             specified by the `Buffer<enc>` type.
 *
 * @param buf The input string buffer,
 *            containing the string with tabs to expand.
 * @param tabsize The number of spaces used to replace a tab character ('\t').
 *                If `tabsize` is zero or less, tabs are not expanded.
 * @param out The output string buffer, where the expanded string will
 *            be written. The buffer must be pre-allocated to have sufficient
 *            space to hold the expanded string.
 *
 * @return The total length of the expanded string written into the output buffer.
 *
 * @note If the output buffer does not have enough space for the expanded string,
 *       this function may result in an overflow, and it is the caller's
 *       responsibility to ensure that the buffer is large enough.
 */
template <ENCODING enc>
static inline npy_intp
string_expandtabs(Buffer<enc> buf, npy_int64 tabsize, Buffer<enc> out)
{
    size_t len = buf.num_codepoints();

    npy_intp new_len = 0, line_pos = 0;

    Buffer<enc> tmp = buf;
    for (size_t i = 0; i < len; i++) {
        npy_ucs4 ch = *tmp;
        if (ch == '\t') {
            if (tabsize > 0) {
                npy_intp incr = tabsize - (line_pos % tabsize);
                line_pos += incr;
                new_len += out.buffer_memset((npy_ucs4) ' ', incr);
                out += incr;
            }
        }
        else {
            line_pos++;
            new_len += out.buffer_memset(ch, 1);
            out++;
            if (ch == '\n' || ch == '\r') {
                line_pos = 0;
            }
        }
        tmp++;
    }
    return new_len;
}

/**
 * @internal
 * @enum ALIGN_POSITION
 * @brief Defines alignment positions.
 */
enum class ALIGN_POSITION {
    CENTER,  ///< Center alignment
    LEFT,    ///< Left alignment
    RIGHT    ///< Right alignment
};

/**
 * @internal
 * @brief Pads a string with a specified character to a given width,
 *        aligning it according to the specified position.
 *
 * This function pads the input string (`buf`) with the specified `fill`
 * character to achieve the desired `width`. The padding can be applied to the
 * left, right, or equally to both sides depending on the alignment `pos`.
 * The result is written to the output buffer (`out`).
 *
 * @tparam enc The encoding of the input and output buffers,
 *             specified by `Buffer<enc>`.
 * @param buf The input string buffer to be padded.
 * @param width The target width for the padded string.
 *              If the input string is already wider than `width`,
 *              no padding is added.
 * @param fill The character used for padding.
 * @param pos The position where the padding should be applied
 *            (`LEFT`, `RIGHT`, or `CENTER`).
 * @param out The output buffer where the padded string is written.
 *            The buffer should be large enough to hold the padded result.
 *
 * @return The width of the final padded string. Returns -1 and raises
 * `PyExc_OverflowError` if the final string exceeds the maximum allowed size.
 * @throws PyExc_OverflowError If the padded string length exceeds the
 *         allowed maximum, the function will raise `PyExc_OverflowError`
 *         with error message: "padded string is too long".
 *
 * @note If the output buffer does not have enough space for the expanded string,
 *       this function may result in an overflow, and it is the caller's
 *       responsibility to ensure that the buffer is large enough.
 */

template <ENCODING enc>
static inline npy_intp
string_pad(Buffer<enc> buf, npy_int64 width, npy_ucs4 fill, ALIGN_POSITION pos, Buffer<enc> out)
{
    size_t final_width = width > 0 ? width : 0;
    if (final_width > PY_SSIZE_T_MAX) {
        npy_gil_error(PyExc_OverflowError, "padded string is too long");
        return -1;
    }

    size_t len_codepoints = buf.num_codepoints();
    size_t len_bytes = buf.after - buf.buf;

    size_t len;
    if (enc == ENCODING::UTF8) {
        len = len_bytes;
    }
    else {
        len = len_codepoints;
    }

    if (len_codepoints >= final_width) {
        buf.buffer_memcpy(out, len);
        return (npy_intp) len;
    }

    size_t left, right;
    if (pos == ALIGN_POSITION::CENTER) {
        size_t pad = final_width - len_codepoints;
        left = pad / 2 + (pad & final_width & 1);
        right = pad - left;
    }
    else if (pos == ALIGN_POSITION::LEFT) {
        left = 0;
        right = final_width - len_codepoints;
    }
    else {
        left = final_width - len_codepoints;
        right = 0;
    }

    assert(left <= PY_SSIZE_T_MAX - len && right <= PY_SSIZE_T_MAX - (left + len));

    if (left > 0) {
        out.advance_chars_or_bytes(out.buffer_memset(fill, left));
    }

    buf.buffer_memcpy(out, len);
    out += len_codepoints;

    if (right > 0) {
        out.advance_chars_or_bytes(out.buffer_memset(fill, right));
    }

    return (npy_intp)final_width;
}

/**
 * @internal
 * @brief Pads the input string with leading zeros ('0')
 *        to achieve a specified width, while ensuring
 *        any leading sign ('+' or '-') remains at the start.
 *
 * This function expands the input string (`buf`) to the specified `width` by
 * adding leading zeros if necessary. If the string contains a leading sign
 * ('+' or '-'), the zeros are inserted after the sign. The padded result is
 * written to the output buffer (`out`).
 *
 * @tparam enc The encoding of the input and output buffers,
 *             specified by `Buffer<enc>`.
 * @param buf The input string buffer to be zero-filled.
 * @param width The target width of the padded string.
 *              If the input string is already wider than `width`,
 *              no padding is added.
 * @param out The output buffer where the zero-filled string is written.
 *            The buffer should have enough space for the padded result.
 *
 * @return The total length of the padded string, or -1 if an error occurs.
 * @throws PyExc_OverflowError If the padded string length exceeds the
 *         allowed maximum, the function will raise `PyExc_OverflowError`
 *         with error message: "padded string is too long".
 */
template <ENCODING enc>
static inline npy_intp
string_zfill(Buffer<enc> buf, npy_int64 width, Buffer<enc> out)
{
    size_t final_width = width > 0 ? width : 0;

    npy_ucs4 fill = '0';
    npy_intp new_len = string_pad(buf, width, fill, ALIGN_POSITION::RIGHT, out);
    if (new_len == -1) {
        return -1;
    }

    size_t offset = final_width - buf.num_codepoints();
    Buffer<enc> tmp = out + offset;

    npy_ucs4 c = *tmp;
    if (c == '+' || c == '-') {
        tmp.buffer_memset(fill, 1);
        out.buffer_memset(c, 1);
    }

    return new_len;
}

/**
 * @internal
 * @brief Partitions an input string into three parts based on a separator and
 * an index.
 *
 * This function partitions the input string (`buf1`) into three parts: the
 * substring before the separator (`out1`), the separator itself (`out2`), and
 * the substring after the separator (`out3`). The separator is provided as
 * `buf2`. If the separator is not found, the partitioning depends on the
 * specified starting position (`pos`).
 *
 * @tparam enc The encoding of the input and output buffers,
 *             specified by `Buffer<enc>`.
 *             This function does not support `ENCODING::UTF8`
 * @param buf1 The input string buffer to be partitioned.
 * @param buf2 The separator string buffer.
 * @param idx The index where the separator is found in `buf1`.
 *            If negative, the partition is based on `pos`.
 * @param out1 Output buffer for the part of the string before the separator.
 * @param out2 Output buffer for the separator.
 * @param out3 Output buffer for the part of the string after the separator.
 * @param final_len1 Pointer to store the length of the string
 *                   before the separator.
 * @param final_len2 Pointer to store the length of the separator.
 * @param final_len3 Pointer to store the length of the string
 *                   after the separator.
 * @param pos Specifies the behavior when the separator is not found:
 *            `FRONT` fills `out1` or `END` fills `out3`.
 *
 * @throws PyExc_ValueError If the separator is an empty string, the function
 *         will raise `PyExc_ValueError` with error message: "empty separator".
 * @throws PyExc_ValueError If buf1 (target string) and buf2 (separator string)
 *         length exceeds the allowed maximum, the function will raise
 *         `PyExc_ValueError` with error message:
 *         "(target/separator) string is too long".
 * @throws PyExc_ValueError If idx is non-negative and exceeds the valid
 *         range for splitting, the function will raise `PyExc_ValueError`
 *         with error message: "input index is too large".
 *
 * @note This function does not perform any comparison between
 *       `buf1` and `buf2` to check if the content at `idx` matches
 *       the separator. It simply skips over a substring of the same
 *       length as `buf2` at position `idx` in `buf1` and assigns
 *       `buf2` to `out2`. `final_len*` is less than `0` if an error is raised.
 */
template <ENCODING enc>
static inline void
string_partition(Buffer<enc> buf1, Buffer<enc> buf2, npy_int64 idx,
                 Buffer<enc> out1, Buffer<enc> out2, Buffer<enc> out3,
                 npy_intp *final_len1, npy_intp *final_len2, npy_intp *final_len3,
                 STRING_SIDE pos)
{
    // StringDType uses an ufunc that implements the find-part as well
    assert(enc != ENCODING::UTF8);

    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();

    if (len1 > PY_SSIZE_T_MAX) {
        npy_gil_error(PyExc_ValueError, "target string is too long");
        *final_len1 = *final_len2 = *final_len3 = -1;
        return;
    }
    if (idx >= 0) {
        if (len2 > len1) {
            npy_gil_error(PyExc_ValueError, "separator string is too long");
            *final_len1 = *final_len2 = *final_len3 = -1;
            return;
        }
        if ((size_t)idx > len1 - len2) {
            npy_gil_error(PyExc_ValueError, "input index is too large");
            *final_len1 = *final_len2 = *final_len3 = -1;
            return;
        }
    }

    if (len2 == 0) {
        npy_gil_error(PyExc_ValueError, "empty separator");
        *final_len1 = *final_len2 = *final_len3 = -1;
        return;
    }

    if (idx < 0) {
        if (pos == STRING_SIDE::FRONT) {
            buf1.buffer_memcpy(out1, len1);
            *final_len1 = len1;
            *final_len2 = *final_len3 = 0;
        }
        else {
            buf1.buffer_memcpy(out3, len1);
            *final_len1 = *final_len2 = 0;
            *final_len3 = len1;
        }
        return;
    }

    buf1.buffer_memcpy(out1, idx);
    *final_len1 = idx;
    buf2.buffer_memcpy(out2, len2);
    *final_len2 = len2;
    (buf1 + idx + len2).buffer_memcpy(out3, len1 - idx - len2);
    *final_len3 = len1 - idx - len2;
}


#endif /* _NPY_CORE_SRC_UMATH_STRING_BUFFER_H_ */
