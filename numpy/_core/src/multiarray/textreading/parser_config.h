
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_PARSER_CONFIG_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_PARSER_CONFIG_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /*
     *  Field delimiter character.
     *  Typically ',', ' ', '\t', ignored if `delimiter_is_whitespace` is true.
     */
    Py_UCS4 delimiter;

    /*
     *  Character used to quote fields.
     *  Typically '"' or "'".  To disable quoting we set this to UINT_MAX
     *  (which is not a valid unicode character and thus cannot occur in the
     *  file; the same is used for all other characters if necessary).
     */
    Py_UCS4 quote;

    /*
     *  Character(s) that indicates the start of a comment.
     *  Typically '#', '%' or ';'.
     *  When encountered in a line and not inside quotes, all character
     *  from the comment character(s) to the end of the line are ignored.
     */
    Py_UCS4 comment;

    /*
     *  Ignore whitespace at the beginning of a field (outside/before quotes).
     *  Is (and must be) set if `delimiter_is_whitespace`.
     */
    bool ignore_leading_whitespace;

    /*
     * If true, the delimiter is ignored and any unicode whitespace is used
     * for splitting (same as `string.split()` in Python). In that case
     * `ignore_leading_whitespace` should also be set.
     */
    bool delimiter_is_whitespace;

    /*
     *  The imaginary unit character. Default is `j`.
     */
    Py_UCS4 imaginary_unit;

     /*
      * Data should be encoded as `latin1` when using python converter
      * (implementing `loadtxt` default Python 2 compatibility mode).
      * The c byte converter is used when the user requested `dtype="S"`.
      * In this case we go via `dtype=object`, however, loadtxt allows latin1
      * while normal object to string casts only accept ASCII, so it ensures
      * that that the object array already contains bytes and not strings.
      */
     bool python_byte_converters;
     bool c_byte_converters;
} parser_config;


#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_PARSER_CONFIG_H_ */
