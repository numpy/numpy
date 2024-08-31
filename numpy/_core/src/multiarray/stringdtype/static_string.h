#ifndef NUMPY_CORE_SRC_MULTIARRAY_STATIC_STRING_H_
#define NUMPY_CORE_SRC_MULTIARRAY_STATIC_STRING_H_

#ifdef __cplusplus
extern "C" {
#endif


#include <stdlib.h>

// some types used by this header are defined in ndarraytypes.h

// one byte in size is reserved for flags and small string optimization
#define NPY_MAX_STRING_SIZE (((int64_t)1 << 8 * (sizeof(size_t) - 1)) - 1)

// Typedefs for allocator functions
typedef void *(*npy_string_malloc_func)(size_t size);
typedef void (*npy_string_free_func)(void *ptr);
typedef void *(*npy_string_realloc_func)(void *ptr, size_t size);

// Use these functions to create and destroy string allocators. Normally
// users won't use these directly and will use an allocator already
// attached to a dtype instance
NPY_NO_EXPORT npy_string_allocator *
NpyString_new_allocator(npy_string_malloc_func m, npy_string_free_func f,
                        npy_string_realloc_func r);

// Deallocates the internal buffer and the allocator itself.
NPY_NO_EXPORT void
NpyString_free_allocator(npy_string_allocator *allocator);

// Allocates a new buffer for *to_init*, which must be set to NULL before
// calling this function, filling the newly allocated buffer with the copied
// contents of the first *size* entries in *init*, which must be valid and
// initialized beforehand. Calling NpyString_free on *to_init* before calling
// this function on an existing string or copying the contents of
// NPY_EMPTY_STRING into *to_init* is sufficient to initialize it. Does not
// check if *to_init* is NULL or if the internal buffer is non-NULL, undefined
// behavior or memory leaks are possible if this function is passed a pointer
// to a an uninitialized struct, a NULL pointer, or an existing heap-allocated
// string.  Returns -1 if allocating the string would exceed the maximum
// allowed string size or exhaust available memory. Returns 0 on success.
NPY_NO_EXPORT int
NpyString_newsize(const char *init, size_t size,
                  npy_packed_static_string *to_init,
                  npy_string_allocator *allocator);

// Zeroes out the packed string and frees any heap allocated data. For
// arena-allocated data, checks if the data are inside the arena and
// will return -1 if not. Returns 0 on success.
NPY_NO_EXPORT int
NpyString_free(npy_packed_static_string *str, npy_string_allocator *allocator);

// Copies the contents of *in* into *out*. Allocates a new string buffer for
// *out*, NpyString_free *must* be called before this is called if *out*
// points to an existing string. Returns -1 if malloc fails. Returns 0 on
// success.
NPY_NO_EXPORT int
NpyString_dup(const npy_packed_static_string *in,
              npy_packed_static_string *out,
              npy_string_allocator *in_allocator,
              npy_string_allocator *out_allocator);

// Allocates a new string buffer for *out* with enough capacity to store
// *size* bytes of text. Does not do any initialization, the caller must
// initialize the string buffer after this function returns. Calling
// NpyString_free on *to_init* before calling this function on an existing
// string or initializing a new string with the contents of NPY_EMPTY_STRING
// is sufficient to initialize it. Does not check if *to_init* has already
// been initialized or if the internal buffer is non-NULL, undefined behavior
// or memory leaks are possible if this function is passed a NULL pointer or
// an existing heap-allocated string.  Returns 0 on success. Returns -1 if
// allocating the string would exceed the maximum allowed string size or
// exhaust available memory. Returns 0 on success.
NPY_NO_EXPORT int
NpyString_newemptysize(size_t size, npy_packed_static_string *out,
                       npy_string_allocator *allocator);

// Determine if *in* corresponds to a null string (e.g. an NA object). Returns
// -1 if *in* cannot be unpacked. Returns 1 if *in* is a null string and
// zero otherwise.
NPY_NO_EXPORT int
NpyString_isnull(const npy_packed_static_string *in);

// Compare two strings. Has the same semantics as if strcmp were passed
// null-terminated C strings with the contents of *s1* and *s2*.
NPY_NO_EXPORT int
NpyString_cmp(const npy_static_string *s1, const npy_static_string *s2);

// Returns the size of the string data in the packed string. Useful in
// situations where only the string size is needed and determining if it is a
// null or unpacking the string is unnecessary.
NPY_NO_EXPORT size_t
NpyString_size(const npy_packed_static_string *packed_string);

NPY_NO_EXPORT int
NpyString_share_memory(const npy_packed_static_string *s1, npy_string_allocator *a1,
                       const npy_packed_static_string *s2, npy_string_allocator *a2);

#ifdef __cplusplus
}
#endif

#endif /* NUMPY_CORE_SRC_MULTIARRAY_STATIC_STRING_H_ */
