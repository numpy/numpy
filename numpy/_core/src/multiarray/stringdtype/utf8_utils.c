#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"
#include "utf8_utils.h"

// Given UTF-8 bytes in *c*, sets *code* to the corresponding unicode
// codepoint for the next character, returning the size of the character in
// bytes. Does not do any validation or error checking: assumes *c* is valid
// utf-8
NPY_NO_EXPORT size_t
utf8_char_to_ucs4_code(const unsigned char *c, Py_UCS4 *code)
{
    if (c[0] <= 0x7F) {
        // 0zzzzzzz -> 0zzzzzzz
        *code = (Py_UCS4)(c[0]);
        return 1;
    }
    else if (c[0] <= 0xDF) {
        // 110yyyyy 10zzzzzz -> 00000yyy yyzzzzzz
        *code = (Py_UCS4)(((c[0] << 6) + c[1]) - ((0xC0 << 6) + 0x80));
        return 2;
    }
    else if (c[0] <= 0xEF) {
        // 1110xxxx 10yyyyyy 10zzzzzz -> xxxxyyyy yyzzzzzz
        *code = (Py_UCS4)(((c[0] << 12) + (c[1] << 6) + c[2]) -
                          ((0xE0 << 12) + (0x80 << 6) + 0x80));
        return 3;
    }
    else {
        // 11110www 10xxxxxx 10yyyyyy 10zzzzzz -> 000wwwxx xxxxyyyy yyzzzzzz
        *code = (Py_UCS4)(((c[0] << 18) + (c[1] << 12) + (c[2] << 6) + c[3]) -
                          ((0xF0 << 18) + (0x80 << 12) + (0x80 << 6) + 0x80));
        return 4;
    }
}

NPY_NO_EXPORT const unsigned char*
find_previous_utf8_character(const unsigned char *c, size_t nchar)
{
    while (nchar > 0) {
        do
        {
            // this assumes well-formed UTF-8 and does not check if we go
            // before the start of the string
            c--;
        // the first byte of a UTF8 character either has
        // the topmost bit clear or has both topmost bits set
        } while ((*c & 0xC0) == 0x80);
        nchar--;
    }
    return c;
}


NPY_NO_EXPORT int
num_utf8_bytes_for_codepoint(uint32_t code)
{
    if (code <= 0x7F) {
        return 1;
    }
    else if (code <= 0x07FF) {
        return 2;
    }
    else if (code <= 0xFFFF) {
        if ((code >= 0xD800) && (code <= 0xDFFF)) {
            // surrogates are invalid UCS4 code points
            return -1;
        }
        return 3;
        }
    else if (code <= 0x10FFFF) {
        return 4;
    }
    else {
        // codepoint is outside the valid unicode range
        return -1;
    }
}

// Find the number of bytes, *utf8_bytes*, needed to store the string
// represented by *codepoints* in UTF-8. The array of *codepoints* is
// *max_length* long, but may be padded with null codepoints. *num_codepoints*
// is the number of codepoints that are not trailing null codepoints. Returns
// 0 on success and -1 when an invalid code point is found.
NPY_NO_EXPORT int
utf8_size(const Py_UCS4 *codepoints, long max_length, size_t *num_codepoints,
          size_t *utf8_bytes)
{
    size_t ucs4len = max_length;

    while (ucs4len > 0 && codepoints[ucs4len - 1] == 0) {
        ucs4len--;
    }
    // ucs4len is now the number of codepoints that aren't trailing nulls.

    size_t num_bytes = 0;

    for (size_t i = 0; i < ucs4len; i++) {
        Py_UCS4 code = codepoints[i];
        int codepoint_bytes = num_utf8_bytes_for_codepoint((uint32_t)code);
        if (codepoint_bytes == -1) {
            return -1;
        }
        num_bytes += codepoint_bytes;
    }

    *num_codepoints = ucs4len;
    *utf8_bytes = num_bytes;

    return 0;
}

// Converts UCS4 code point *code* to 4-byte character array *c*. Assumes *c*
// is a zero-filled 4 byte array and *code* is a valid codepoint and does not
// do any error checking! Returns the number of bytes in the UTF-8 character.
NPY_NO_EXPORT size_t
ucs4_code_to_utf8_char(Py_UCS4 code, char *c)
{
    if (code <= 0x7F) {
        // 0zzzzzzz -> 0zzzzzzz
        c[0] = (char)code;
        return 1;
    }
    else if (code <= 0x07FF) {
        // 00000yyy yyzzzzzz -> 110yyyyy 10zzzzzz
        c[0] = (0xC0 | (code >> 6));
        c[1] = (0x80 | (code & 0x3F));
        return 2;
    }
    else if (code <= 0xFFFF) {
        // xxxxyyyy yyzzzzzz -> 110yyyyy 10zzzzzz
        c[0] = (0xe0 | (code >> 12));
        c[1] = (0x80 | ((code >> 6) & 0x3f));
        c[2] = (0x80 | (code & 0x3f));
        return 3;
    }
    else {
        // 00wwwxx xxxxyyyy yyzzzzzz -> 11110www 10xxxxxx 10yyyyyy 10zzzzzz
        c[0] = (0xf0 | (code >> 18));
        c[1] = (0x80 | ((code >> 12) & 0x3f));
        c[2] = (0x80 | ((code >> 6) & 0x3f));
        c[3] = (0x80 | (code & 0x3f));
        return 4;
    }
}

/*******************************************************************************/
// Everything until the closing /***/ block below is a copy of the
// Bjoern Hoerhmann DFA UTF-8 validator
// License: MIT
// Copyright (c) 2008-2009 Bjoern Hoehrmann <bjoern@hoehrmann.de>
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// See http://bjoern.hoehrmann.de/utf-8/decoder/dfa/ for details.
//
// in principle could use something like simdutf to accelerate this

#define UTF8_ACCEPT 0
#define UTF8_REJECT 1

static const uint8_t utf8d[] = {
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 00..1f
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 20..3f
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 40..5f
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 60..7f
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, // 80..9f
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7, // a0..bf
  8,8,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, // c0..df
  0xa,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x4,0x3,0x3, // e0..ef
  0xb,0x6,0x6,0x6,0x5,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8, // f0..ff
  0x0,0x1,0x2,0x3,0x5,0x8,0x7,0x1,0x1,0x1,0x4,0x6,0x1,0x1,0x1,0x1, // s0..s0
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1, // s1..s2
  1,2,1,1,1,1,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1, // s3..s4
  1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,3,1,1,1,1,1,1, // s5..s6
  1,3,1,1,1,1,1,3,1,3,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // s7..s8
};

static uint32_t inline
utf8_decode(uint32_t* state, uint32_t* codep, uint32_t byte) {
    uint32_t type = utf8d[byte];

    *codep = (*state != UTF8_ACCEPT) ?
            (byte & 0x3fu) | (*codep << 6) :
            (0xff >> type) & (byte);

    *state = utf8d[256 + *state*16 + type];
    return *state;
}

/*******************************************************************************/

// calculate the size in bytes required to store a UTF-8 encoded version of the
// UTF-32 encoded string stored in **s**, which is **max_bytes** long.
NPY_NO_EXPORT Py_ssize_t
utf8_buffer_size(const uint8_t *s, size_t max_bytes)
{
    uint32_t codepoint;
    uint32_t state = 0;
    size_t num_bytes = 0;
    Py_ssize_t encoded_size_in_bytes = 0;

    // ignore trailing nulls
    while (max_bytes > 0 && s[max_bytes - 1] == 0) {
        max_bytes--;
    }

    if (max_bytes == 0) {
        return 0;
    }

    for (; num_bytes < max_bytes; ++s)
    {
        utf8_decode(&state, &codepoint, *s);
        if (state == UTF8_REJECT)
        {
            return -1;
        }
        else if(state == UTF8_ACCEPT)
        {
            encoded_size_in_bytes += num_utf8_bytes_for_codepoint(codepoint);
        }
        num_bytes += 1;
    }

    if (state != UTF8_ACCEPT) {
        return -1;
    }
    return encoded_size_in_bytes;
}


// calculate the number of UTF-32 code points in the UTF-8 encoded string
// stored in **s**, which is **max_bytes** long.
NPY_NO_EXPORT int
num_codepoints_for_utf8_bytes(const unsigned char *s, size_t *num_codepoints, size_t max_bytes)
{
    uint32_t codepoint;
    uint32_t state = 0;
    size_t num_bytes = 0;
    *num_codepoints = 0;

    // ignore trailing nulls
    while (max_bytes > 0 && s[max_bytes - 1] == 0) {
        max_bytes--;
    }

    if (max_bytes == 0) {
        return UTF8_ACCEPT;
    }

    for (; num_bytes < max_bytes; ++s)
    {
        utf8_decode(&state, &codepoint, *s);
        if (state == UTF8_REJECT)
        {
            return state;
        }
        else if(state == UTF8_ACCEPT)
        {
            *num_codepoints += 1;
        }
        num_bytes += 1;
    }

    return state != UTF8_ACCEPT;
}

NPY_NO_EXPORT void
find_start_end_locs(char* buf, size_t buffer_size, npy_int64 start_index, npy_int64 end_index,
                    char **start_loc, char **end_loc) {
    size_t bytes_consumed = 0;
    size_t num_codepoints = 0;
    if (num_codepoints == (size_t) start_index) {
        *start_loc = buf;
    }
    if (num_codepoints == (size_t) end_index) {
        *end_loc = buf;
    }
    while (bytes_consumed < buffer_size && num_codepoints < (size_t) end_index) {
        size_t num_bytes = num_bytes_for_utf8_character((const unsigned char*)buf);
        num_codepoints += 1;
        bytes_consumed += num_bytes;
        buf += num_bytes;
        if (num_codepoints == (size_t) start_index) {
            *start_loc = buf;
        }
        if (num_codepoints == (size_t) end_index) {
            *end_loc = buf;
        }
    }
    assert(start_loc != NULL);
    assert(end_loc != NULL);
}

NPY_NO_EXPORT size_t
utf8_character_index(
        const char* start_loc, size_t start_byte_offset, size_t start_index,
        size_t search_byte_offset, size_t buffer_size)
{
    size_t bytes_consumed = 0;
    size_t cur_index = start_index;
    while (bytes_consumed < buffer_size && bytes_consumed < search_byte_offset) {
        size_t num_bytes = num_bytes_for_utf8_character((const unsigned char*)start_loc);
        cur_index += 1;
        bytes_consumed += num_bytes;
        start_loc += num_bytes;
    }
    return cur_index - start_index;
}
