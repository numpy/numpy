
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"

#include "textreading/stream.h"
#include "textreading/tokenize.h"
#include "textreading/parser_config.h"
#include "textreading/growth.h"

/*
    How parsing quoted fields works:

    For quoting to be activated, the first character of the field
    must be the quote character (after taking into account
    ignore_leading_spaces).  While quoting is active, delimiters
    are treated as regular characters, not delimiters.  Quoting is
    deactivated by the second occurrence of the quote character.  An
    exception is the occurrence of two consecutive quote characters,
    which is treated as a literal occurrence of a single quote character.
    E.g. (with delimiter=',' and quote='"'):
        12.3,"New York, NY","3'2"""
    The second and third fields are `New York, NY` and `3'2"`.

    If a non-delimiter occurs after the closing quote, the quote is
    ignored and parsing continues with quoting deactivated.  Quotes
    that occur while quoting is not activated are not handled specially;
    they become part of the data.
    E.g:
        12.3,"ABC"DEF,XY"Z
    The second and third fields are `ABCDEF` and `XY"Z`.

    Note that the second field of
        12.3,"ABC"   ,4.5
    is `ABC   `.  Currently there is no option to ignore whitespace
    at the end of a field.
*/


template <typename UCS>
static inline int
copy_to_field_buffer(tokenizer_state *ts,
        const UCS *chunk_start, const UCS *chunk_end)
{
    npy_intp chunk_length = chunk_end - chunk_start;
    /* Space for length +1 termination, +2 additional padding for add_field */
    npy_intp size = chunk_length + ts->field_buffer_pos + 3;

    if (NPY_UNLIKELY(ts->field_buffer_length < size)) {
        npy_intp alloc_size = grow_size_and_multiply(&size, 32, sizeof(Py_UCS4));
        if (alloc_size < 0) {
            PyErr_Format(PyExc_ValueError,
                    "line too long to handle while reading file.");
            return -1;
        }
        Py_UCS4 *grown = (Py_UCS4 *)PyMem_Realloc(ts->field_buffer, alloc_size);
        if (grown == nullptr) {
            PyErr_NoMemory();
            return -1;
        }
        ts->field_buffer_length = size;
        ts->field_buffer = grown;
    }

    Py_UCS4 *write_pos = ts->field_buffer + ts->field_buffer_pos;
    for (; chunk_start < chunk_end; chunk_start++, write_pos++) {
        *write_pos = (Py_UCS4)*chunk_start;
    }
    *write_pos = '\0';  /* always ensure we end with NUL */
    ts->field_buffer_pos += chunk_length;
    return 0;
}


static inline int
add_field(tokenizer_state *ts)
{
    /* The previous field is done, advance to keep a NUL byte at the end */
    ts->field_buffer_pos += 1;

    if (NPY_UNLIKELY(ts->num_fields + 1 > ts->fields_size)) {
        npy_intp size = ts->num_fields;

        npy_intp alloc_size = grow_size_and_multiply(
                &size, 4, sizeof(field_info));
        if (alloc_size < 0) {
            /* Check for a size overflow, path should be almost impossible. */
            PyErr_Format(PyExc_ValueError,
                    "too many columns found; cannot read file.");
            return -1;
        }
        field_info *fields = (field_info *)PyMem_Realloc(ts->fields, alloc_size);
        if (fields == nullptr) {
            PyErr_NoMemory();
            return -1;
        }
        ts->fields = fields;
        ts->fields_size = size;
    }

    ts->fields[ts->num_fields].offset = ts->field_buffer_pos;
    ts->fields[ts->num_fields].quoted = false;
    ts->num_fields += 1;
    /* Ensure this (currently empty) word is NUL terminated. */
    ts->field_buffer[ts->field_buffer_pos] = '\0';
    assert(ts->field_buffer_length > ts->field_buffer_pos);
    return 0;
}


template <typename UCS>
static inline int
tokenizer_core(tokenizer_state *ts, parser_config *const config)
{
    UCS *pos = (UCS *)ts->pos;
    UCS *stop = (UCS *)ts->end;
    UCS *chunk_start;

    if (ts->state == TOKENIZE_CHECK_QUOTED) {
        /* before we can check for quotes, strip leading whitespace */
        if (config->ignore_leading_whitespace) {
            while (pos < stop && Py_UNICODE_ISSPACE(*pos) &&
                        *pos != '\r' && *pos != '\n') {
                pos++;
            }
            if (pos == stop) {
                ts->pos = (char *)pos;
                return 0;
            }
        }

        /* Setting chunk effectively starts the field */
        if (*pos == config->quote) {
            ts->fields[ts->num_fields - 1].quoted = true;
            ts->state = TOKENIZE_QUOTED;
            pos++;  /* TOKENIZE_QUOTED is OK with pos == stop */
        }
        else {
            /* Set to TOKENIZE_QUOTED or TOKENIZE_QUOTED_WHITESPACE */
            ts->state = ts->unquoted_state;
        }
    }

    switch (ts->state) {
        case TOKENIZE_UNQUOTED:
            chunk_start = pos;
            for (; pos < stop; pos++) {
                if (*pos == '\r') {
                    ts->state = TOKENIZE_EAT_CRLF;
                    break;
                }
                else if (*pos == '\n') {
                    ts->state = TOKENIZE_LINE_END;
                    break;
                }
                else if (*pos == config->delimiter) {
                    ts->state = TOKENIZE_INIT;
                    break;
                }
                else if (*pos == config->comment) {
                    ts->state = TOKENIZE_GOTO_LINE_END;
                    break;
                }
            }
            if (copy_to_field_buffer(ts, chunk_start, pos) < 0) {
                return -1;
            }
            pos++;
            break;

        case TOKENIZE_UNQUOTED_WHITESPACE:
            /* Note, this branch is largely identical to `TOKENIZE_UNQUOTED` */
            chunk_start = pos;
            for (; pos < stop; pos++) {
                if (*pos == '\r') {
                    ts->state = TOKENIZE_EAT_CRLF;
                    break;
                }
                else if (*pos == '\n') {
                    ts->state = TOKENIZE_LINE_END;
                    break;
                }
                else if (Py_UNICODE_ISSPACE(*pos)) {
                    ts->state = TOKENIZE_INIT;
                    break;
                }
                else if (*pos == config->comment) {
                    ts->state = TOKENIZE_GOTO_LINE_END;
                    break;
                }
            }
            if (copy_to_field_buffer(ts, chunk_start, pos) < 0) {
                return -1;
            }
            pos++;
            break;

        case TOKENIZE_QUOTED:
            chunk_start = pos;
            for (; pos < stop; pos++) {
                if (*pos == config->quote) {
                    ts->state = TOKENIZE_QUOTED_CHECK_DOUBLE_QUOTE;
                    break;
                }
            }
            if (copy_to_field_buffer(ts, chunk_start, pos) < 0) {
                return -1;
            }
            pos++;
            break;

        case TOKENIZE_QUOTED_CHECK_DOUBLE_QUOTE:
            if (*pos == config->quote) {
                /* Copy the quote character directly from the config: */
                if (copy_to_field_buffer(ts,
                        &config->quote, &config->quote+1) < 0) {
                    return -1;
                }
                ts->state = TOKENIZE_QUOTED;
                pos++;
            }
            else {
                /* continue parsing as if unquoted */
                /* Set to TOKENIZE_UNQUOTED or TOKENIZE_UNQUOTED_WHITESPACE */
                ts->state = ts->unquoted_state;
            }
            break;

        case TOKENIZE_GOTO_LINE_END:
            if (ts->buf_state != BUFFER_MAY_CONTAIN_NEWLINE) {
                pos = stop;  /* advance to next buffer */
                ts->state = TOKENIZE_LINE_END;
                break;
            }
            for (; pos < stop; pos++) {
                if (*pos == '\r') {
                    ts->state = TOKENIZE_EAT_CRLF;
                    break;
                }
                else if (*pos == '\n') {
                    ts->state = TOKENIZE_LINE_END;
                    break;
                }
            }
            pos++;
            break;

        case TOKENIZE_EAT_CRLF:
            /* "Universal newline" support: remove \n in \r\n. */
            if (*pos == '\n') {
                pos++;
            }
            ts->state = TOKENIZE_LINE_END;
            break;

        default:
            assert(0);
    }

    ts->pos = (char *)pos;
    return 0;
}


/*
 * This tokenizer always copies the full "row" (all tokens).  This makes
 * two things easier:
 * 1. It means that every word is guaranteed to be followed by a NUL character
 *    (although it can include one as well).
 * 2. If usecols are used we can sniff the first row easier by parsing it
 *    fully.  Further, usecols can be negative so we may not know which row we
 *    need up-front.
 *
 * The tokenizer could grow the ability to skip fields and check the
 * maximum number of fields when known, it is unclear that this is worthwhile.
 *
 * Unlike some tokenizers, this one tries to work in chunks and copies
 * data in chunks as well.  The hope is that this makes multiple light-weight
 * loops rather than a single heavy one, to allow e.g. quickly scanning for the
 * end of a field.  Copying chunks also means we usually only check once per
 * field whether the buffer is large enough.
 * Different choices are possible, this one seems to work well, though.
 *
 * The core (main part) of the tokenizer is specialized for the three Python
 * unicode flavors UCS1, UCS2, and UCS4 as a worthwhile optimization.
 */
NPY_NO_EXPORT int
npy_tokenize(stream *s, tokenizer_state *ts, parser_config *const config)
{
    assert(ts->fields_size >= 2);
    assert(ts->field_buffer_length >= 2*(Py_ssize_t)sizeof(Py_UCS4));

    int finished_reading_file = 0;

    /* Reset to start of buffer */
    ts->field_buffer_pos = 0;
    ts->num_fields = 0;

    while (true) {
        /*
         * This loop adds new fields to the result (to make up a full row)
         * until the row ends (typically a line end or the file end)
         */
        if (ts->state == TOKENIZE_INIT) {
            /* Start a new field */
            if (add_field(ts) < 0) {
                return -1;
            }
            ts->state = TOKENIZE_CHECK_QUOTED;
        }

        if (NPY_UNLIKELY(ts->pos >= ts->end)) {
            if (ts->buf_state == BUFFER_IS_LINEND &&
                    ts->state != TOKENIZE_QUOTED) {
                /*
                 * Finished line, do not read anymore (also do not eat \n).
                 * If we are in a quoted field and the "line" does not end with
                 * a newline, the quoted field will not have it either.
                 * I.e. `np.loadtxt(['"a', 'b"'], dtype="S2", quotechar='"')`
                 * reads "ab". This matches `next(csv.reader(['"a', 'b"']))`.
                 */
                break;
            }
            /* fetch new data */
            ts->buf_state = stream_nextbuf(s,
                    &ts->pos, &ts->end, &ts->unicode_kind);
            if (ts->buf_state < 0) {
                return -1;
            }
            if (ts->buf_state == BUFFER_IS_FILEEND) {
                finished_reading_file = 1;
                ts->pos = ts->end;  /* stream should ensure this. */
                break;
            }
            else if (ts->pos == ts->end) {
                /* This must be an empty line (and it must be indicated!). */
                assert(ts->buf_state == BUFFER_IS_LINEND);
                break;
            }
        }
        int status;
        if (ts->unicode_kind == PyUnicode_1BYTE_KIND) {
            status = tokenizer_core<Py_UCS1>(ts, config);
        }
        else if (ts->unicode_kind == PyUnicode_2BYTE_KIND) {
            status = tokenizer_core<Py_UCS2>(ts, config);
        }
        else {
            assert(ts->unicode_kind == PyUnicode_4BYTE_KIND);
            status = tokenizer_core<Py_UCS4>(ts, config);
        }
        if (status < 0) {
            return -1;
        }

        if (ts->state == TOKENIZE_LINE_END) {
            break;
        }
    }

    /*
     * We have finished tokenizing a full row into fields, finalize result
     */
    if (ts->buf_state == BUFFER_IS_LINEND) {
        /* This line is "finished", make sure we don't touch it again: */
        ts->buf_state = BUFFER_MAY_CONTAIN_NEWLINE;
        if (NPY_UNLIKELY(ts->pos < ts->end)) {
            PyErr_SetString(PyExc_ValueError,
                    "Found an unquoted embedded newline within a single line of "
                    "input.  This is currently not supported.");
            return -1;
        }
    }

    /* Finish the last field (we "append" one to store the last ones length) */
    if (add_field(ts) < 0) {
        return -1;
    }
    ts->num_fields -= 1;

    /*
     * We always start a new field (at the very beginning and whenever a
     * delimiter was found).
     * This gives us two scenarios where we need to ignore the last field
     * if it is empty:
     * 1. If there is exactly one empty (unquoted) field, the whole line is
     *    empty.
     * 2. If we are splitting on whitespace we always ignore a last empty
     *    field to match Python's splitting: `" 1 ".split()`.
     *    (Zero fields are possible when we are only skipping lines)
     */
    if (ts->num_fields == 1 || (ts->num_fields > 0
                && ts->unquoted_state == TOKENIZE_UNQUOTED_WHITESPACE)) {
        size_t offset_last = ts->fields[ts->num_fields-1].offset;
        size_t end_last = ts->fields[ts->num_fields].offset;
        if (!ts->fields->quoted && end_last - offset_last == 1) {
            ts->num_fields--;
        }
    }
    ts->state = TOKENIZE_INIT;
    return finished_reading_file;
}


NPY_NO_EXPORT void
npy_tokenizer_clear(tokenizer_state *ts)
{
    PyMem_FREE(ts->field_buffer);
    ts->field_buffer = nullptr;
    ts->field_buffer_length = 0;

    PyMem_FREE(ts->fields);
    ts->fields = nullptr;
    ts->fields_size = 0;
}


/*
 * Initialize the tokenizer.  We may want to copy all important config
 * variables into the tokenizer.  This would improve the cache locality during
 * tokenizing.
 */
NPY_NO_EXPORT int
npy_tokenizer_init(tokenizer_state *ts, parser_config *config)
{
    /* State and buf_state could be moved into tokenize if we go by row */
    ts->buf_state = BUFFER_MAY_CONTAIN_NEWLINE;
    ts->state = TOKENIZE_INIT;
    if (config->delimiter_is_whitespace) {
        ts->unquoted_state = TOKENIZE_UNQUOTED_WHITESPACE;
    }
    else {
        ts->unquoted_state = TOKENIZE_UNQUOTED;
    }
    ts->num_fields = 0;

    ts->buf_state = 0;
    ts->pos = nullptr;
    ts->end = nullptr;

    ts->field_buffer = (Py_UCS4 *)PyMem_Malloc(32 * sizeof(Py_UCS4));
    if (ts->field_buffer == nullptr) {
        PyErr_NoMemory();
        return -1;
    }
    ts->field_buffer_length = 32;

    ts->fields = (field_info *)PyMem_Malloc(4 * sizeof(*ts->fields));
    if (ts->fields == nullptr) {
        PyMem_Free(ts->field_buffer);
        ts->field_buffer = nullptr;
        PyErr_NoMemory();
        return -1;
    }
    ts->fields_size = 4;
    return 0;
}
