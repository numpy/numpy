#ifndef _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_UTF8_UTILS_H_
#define _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_UTF8_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT size_t
utf8_char_to_ucs4_code(const unsigned char *c, Py_UCS4 *code);

static inline int num_bytes_for_utf8_character(const unsigned char *c)
{
    // adapted from https://github.com/skeeto/branchless-utf8
    // the first byte of a UTF-8 character encodes the length of the character
    static const char LENGTHS_LUT[] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 4, 0
    };
    return LENGTHS_LUT[c[0] >> 3];
}

NPY_NO_EXPORT const unsigned char*
find_previous_utf8_character(const unsigned char *c, size_t nchar);

NPY_NO_EXPORT int
num_utf8_bytes_for_codepoint(uint32_t code);

NPY_NO_EXPORT int
num_codepoints_for_utf8_bytes(const unsigned char *s, size_t *num_codepoints, size_t max_bytes);

NPY_NO_EXPORT int
utf8_size(const Py_UCS4 *codepoints, long max_length, size_t *num_codepoints,
          size_t *utf8_bytes);

NPY_NO_EXPORT size_t
ucs4_code_to_utf8_char(Py_UCS4 code, char *c);

NPY_NO_EXPORT Py_ssize_t
utf8_buffer_size(const uint8_t *s, size_t max_bytes);

NPY_NO_EXPORT void
find_start_end_locs(char* buf, size_t buffer_size, npy_int64 start_index, npy_int64 end_index,
                    char **start_loc, char **end_loc);

NPY_NO_EXPORT size_t
utf8_character_index(
        const char* start_loc, size_t start_byte_offset, size_t start_index,
        size_t search_byte_offset, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_UTF8_UTILS_H_ */
