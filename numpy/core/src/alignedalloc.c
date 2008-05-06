/*
 * Cross platform memory allocator with optional alignment
 *
 * Most of the code is from Steven Johnson (FFTW).
 *
 * TODO: 
 *  - Had some magic for debug aligned allocators (to detect mismatch)
 *  - some platforms set errno for errors, other not. According to man
 *  posix_memalign, this function does NOT set errno; according to MSDN,
 *  _align_memalloc does set errno. This seems logical seeing the different
 *  signatures of the functions, but I have not checked it.
 */

#ifndef _MULTIARRAYMODULE
#error Mmmh, looke like Python.h was not already included !
#endif

/* Are those four headers always available ? */
#include <stdlib.h>
#include <errno.h>
#include <stddef.h>		/* ptrdiff_t */
#include <string.h>		/* memmove */

#ifdef HAVE_STDINT_H
#include <stdint.h>		/* uintptr_t */
#else
#define uintptr_t size_t
#endif

#define NPY_ALIGNED_NOT_POWER_OF_TWO(n) (((n) & ((n) - 1)))
#define NPY_ALIGNED_UI(p) ((uintptr_t) (p))
#define NPY_ALIGNED_CP(p) ((char *) p)

#define NPY_ALIGNED_PTR_ALIGN(p0, alignment, offset)				\
            ((void *) (((NPY_ALIGNED_UI(p0) + (alignment + sizeof(void*)) + offset)	\
            & (~NPY_ALIGNED_UI(alignment - 1)))				\
               - offset))

/* pointer must sometimes be aligned; assume sizeof(void*) is a power of two */
#define NPY_ALIGNED_ORIG_PTR(p) \
        (*(((void **) (NPY_ALIGNED_UI(p) & (~NPY_ALIGNED_UI(sizeof(void*) - 1)))) - 1))

/* Default implementation: simply using malloc and co */
static void *_aligned_offset_malloc(size_t size, size_t alignment,
				    size_t offset)
{
	void *p0, *p;

	if (NPY_ALIGNED_NOT_POWER_OF_TWO(alignment)) {
		errno = EINVAL;
		return ((void *) 0);
	}
	if (size == 0) {
		return ((void *) 0);
	}
	if (alignment < sizeof(void *)) {
		alignment = sizeof(void *);
	}

	/* including the extra sizeof(void*) is overkill on a 32-bit
	   machine, since malloc is already 8-byte aligned, as long
	   as we enforce alignment >= 8 ...but oh well */

	p0 = malloc(size + (alignment + sizeof(void *)));
	if (!p0) {
		return ((void *) 0);
	}
	p = NPY_ALIGNED_PTR_ALIGN(p0, alignment, offset);
	NPY_ALIGNED_ORIG_PTR(p) = p0;
	return p;
}

void *_aligned_malloc(size_t size, size_t alignment)
{
	return _aligned_offset_malloc(size, alignment, 0);
}

void _aligned_free(void *memblock)
{
	if (memblock) {
		free(NPY_ALIGNED_ORIG_PTR(memblock));
	}
}

void *_aligned_realloc(void *memblock, size_t size, size_t alignment)
{
	void *p0, *p;
	ptrdiff_t shift;

	if (!memblock) {
		return _aligned_malloc(size, alignment);
	}
	if (NPY_ALIGNED_NOT_POWER_OF_TWO(alignment)) {
		goto bad;
	}
	if (size == 0) {
		_aligned_free(memblock);
		return ((void *) 0);
	}
	if (alignment < sizeof(void *)) {
		alignment = sizeof(void *);
	}

	p0 = NPY_ALIGNED_ORIG_PTR(memblock);
	if (memblock != NPY_ALIGNED_PTR_ALIGN(p0, alignment, 0)) {
		goto bad;	/* it is an error for the alignment to change */
	}
	shift = NPY_ALIGNED_CP(memblock) - NPY_ALIGNED_CP(p0);

	p0 = realloc(p0, size + (alignment + sizeof(void *)));
	if (!p0) {
		return ((void *) 0);
	}
	p = NPY_ALIGNED_PTR_ALIGN(p0, alignment, 0);

	/* relative shift of actual data may be different from before, ugh */
	if (shift != NPY_ALIGNED_CP(p) - NPY_ALIGNED_CP(p0)) {
		/* ugh, moves more than necessary if size is increased */
		memmove(NPY_ALIGNED_CP(p), NPY_ALIGNED_CP(p0) + shift,
			size);
	}

	NPY_ALIGNED_ORIG_PTR(p) = p0;
	return p;

bad:
	errno = EINVAL;
	return ((void *) 0);
}
