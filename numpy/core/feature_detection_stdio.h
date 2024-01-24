#define _GNU_SOURCE
#include <stdio.h>
#include <fcntl.h>

#if 0 /* Only for setup_common.py, not the C compiler */
off_t ftello(FILE *stream);
int fseeko(FILE *stream, off_t offset, int whence);
int fallocate(int, int, off_t, off_t);
#endif
