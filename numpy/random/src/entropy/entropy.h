#ifndef _RANDOMDGEN__ENTROPY_H_
#define _RANDOMDGEN__ENTROPY_H_

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

extern void entropy_fill(void *dest, size_t size);

extern bool entropy_getbytes(void *dest, size_t size);

extern bool entropy_fallback_getbytes(void *dest, size_t size);

#endif
