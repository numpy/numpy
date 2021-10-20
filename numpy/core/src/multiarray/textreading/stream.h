#ifndef _STREAM_H_
#define _STREAM_H_

#include <stdint.h>

/*
 * When getting the next line, we hope that the buffer provider can already
 * give some information about the newlines, because for Python iterables
 * we definitely expect to get line-by-line buffers.
 */
#define BUFFER_MAY_CONTAIN_NEWLINE 0
#define BUFFER_IS_PARTIAL_LINE 1
#define BUFFER_IS_LINEND 2
#define BUFFER_IS_FILEEND 3

typedef struct _stream {
    void *stream_data;
    int (*stream_nextbuf)(void *sdata, char **start, char **end, int *kind);
    // Note that the first argument to stream_close is the stream pointer
    // itself, not the stream_data pointer.
    int (*stream_close)(struct _stream *strm);
} stream;


#define stream_nextbuf(s, start, end, kind)  \
        ((s)->stream_nextbuf((s)->stream_data, start, end, kind))
#define stream_close(s)    ((s)->stream_close((s)))

#endif
