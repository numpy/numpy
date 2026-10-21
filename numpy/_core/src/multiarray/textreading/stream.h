#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * When getting the next line, we hope that the buffer provider can already
 * give some information about the newlines, because for Python iterables
 * we definitely expect to get line-by-line buffers.
 *
 * BUFFER_IS_FILEEND must be returned when the end of the file is reached and
 * must NOT be returned together with a valid (non-empty) buffer.
 */
#define BUFFER_MAY_CONTAIN_NEWLINE 0
#define BUFFER_IS_LINEND 1
#define BUFFER_IS_FILEEND 2

/*
 * Base struct for streams.  We currently have two, a chunked reader for
 * filelikes and a line-by-line for any iterable.
 * As of writing, the chunked reader was only used for filelikes not already
 * opened.  That is to preserve the amount read in case of an error exactly.
 * If we drop this, we could read it more often (but not when `max_rows` is
 * used).
 *
 * The "streams" can extend this struct to store their own data (so it is
 * a very lightweight "object").
 */
typedef struct _stream {
    int (*stream_nextbuf)(void *sdata, char **start, char **end, int *kind);
    // Note that the first argument to stream_close is the stream pointer
    // itself, not the stream_data pointer.
    int (*stream_close)(struct _stream *strm);
} stream;


#define stream_nextbuf(s, start, end, kind)  \
        ((s)->stream_nextbuf((s), start, end, kind))
#define stream_close(s)    ((s)->stream_close((s)))

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_STREAM_H_ */
