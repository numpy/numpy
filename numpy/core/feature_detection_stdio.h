#include <fcntl.h>
#include <stdio.h>

off_t
ftello(FILE *stream);
int
fseeko(FILE *stream, off_t offset, int whence);
int
fallocate(int, int, off_t, off_t);
