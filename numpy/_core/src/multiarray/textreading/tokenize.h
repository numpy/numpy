
#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_TOKENIZE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_TOKENIZE_H_

#include <Python.h>
#include "numpy/ndarraytypes.h"

#include "textreading/stream.h"
#include "textreading/parser_config.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    /* Initialization of fields */
    TOKENIZE_INIT,
    TOKENIZE_CHECK_QUOTED,
    /* Main field parsing states */
    TOKENIZE_UNQUOTED,
    TOKENIZE_UNQUOTED_WHITESPACE,
    TOKENIZE_QUOTED,
    /* Handling of two character control sequences (except "\r\n") */
    TOKENIZE_QUOTED_CHECK_DOUBLE_QUOTE,
    /* Line end handling */
    TOKENIZE_LINE_END,
    TOKENIZE_EAT_CRLF,  /* "\r\n" support (carriage return, line feed) */
    TOKENIZE_GOTO_LINE_END,
} tokenizer_parsing_state;


typedef struct {
    size_t offset;
    bool quoted;
} field_info;


typedef struct {
    tokenizer_parsing_state state;
    /* Either TOKENIZE_UNQUOTED or TOKENIZE_UNQUOTED_WHITESPACE: */
    tokenizer_parsing_state unquoted_state;
    int unicode_kind;
    int buf_state;
    /* the buffer we are currently working on */
    char *pos;
    char *end;
    /*
     * Space to copy words into.  Due to `add_field` not growing the buffer
     * but writing a \0 termination, the buffer must always be two larger
     * (add_field can be called twice if a row ends in a delimiter: "123,").
     * The first byte beyond the current word is always NUL'ed on write, the
     * second byte is there to allow easy appending of an additional empty
     * word at the end (this word is also NUL terminated).
     */
    npy_intp field_buffer_length;
    npy_intp field_buffer_pos;
    Py_UCS4 *field_buffer;

    /*
     * Fields, including information about the field being quoted.  This
     * always includes one "additional" empty field.  The length of a field
     * is equal to `fields[i+1].offset - fields[i].offset - 1`.
     *
     * The tokenizer assumes at least one field is allocated.
     */
    npy_intp num_fields;
    npy_intp fields_size;
    field_info *fields;
} tokenizer_state;


NPY_NO_EXPORT void
npy_tokenizer_clear(tokenizer_state *ts);


NPY_NO_EXPORT int
npy_tokenizer_init(tokenizer_state *ts, parser_config *config);

NPY_NO_EXPORT int
npy_tokenize(stream *s, tokenizer_state *ts, parser_config *const config);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_TOKENIZE_H_ */
