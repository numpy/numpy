/* Static string API
 *
 * Strings can be stored in multiple ways. Initialization leaves them as
 * either initialized or missing, with initialization being inside an arena
 * except for short strings that fit inside the static string struct stored in
 * the array buffer (<=15 or 7 bytes, depending on architecture).
 *
 * If a string is replaced, it will be allocated on the heap if it cannot fit
 * inside the original short string or arena allocation.  If a string is set
 * to missing, the information about the previous allocation is kept, so
 * replacement with a new string can use the possible previous arena
 * allocation. Note that after replacement with a short string, any arena
 * information is lost, so a later replacement with a longer one will always
 * be on the heap.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <string.h>
#include <stdarg.h>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_2_compat.h"
#include "numpy/arrayobject.h"
#include "static_string.h"
#include "dtypemeta.h"

#if NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN

// the last and hence highest byte in vstring.size is reserved for flags
//
// SSSS SSSF

typedef struct _npy_static_vstring_t {
    size_t offset;
    size_t size_and_flags;
} _npy_static_vstring_t;

typedef struct _short_string_buffer {
    char buf[sizeof(_npy_static_vstring_t) - 1];
    unsigned char size_and_flags;
} _short_string_buffer;

#elif NPY_BYTE_ORDER == NPY_BIG_ENDIAN

// the first and hence highest byte in vstring.size is reserved for flags
//
// FSSS SSSS

typedef struct _npy_static_vstring_t {
    size_t size_and_flags;
    size_t offset;
} _npy_static_vstring_t;

typedef struct _short_string_buffer {
    unsigned char size_and_flags;
    char buf[sizeof(_npy_static_vstring_t) - 1];
} _short_string_buffer;

#endif

typedef union _npy_static_string_u {
    _npy_static_vstring_t vstring;
    _short_string_buffer direct_buffer;
} _npy_static_string_u;


// Flags defining whether a string exists and how it is stored.
// Inside the arena, long means not medium (i.e., >255 bytes), while
// outside, it means not short (i.e., >15 or 7, depending on arch).

// set for null strings representing missing data
#define NPY_STRING_MISSING 0x80        // 1000 0000
// set after an array entry is initialized for the first time
#define NPY_STRING_INITIALIZED 0x40    // 0100 0000
// The string data is managed by malloc/free on the heap
// Only set for strings that have been mutated to be longer
// than the original entry
#define NPY_STRING_OUTSIDE_ARENA 0x20  // 0010 0000
// A string that lives in the arena with a size longer
// than 255 bytes, so the size in the arena is stored in a size_t
#define NPY_STRING_LONG 0x10           // 0001 0000


// The last four bits of the flags byte is currently unused.
#define NPY_STRING_FLAG_MASK 0xF0      // 1111 0000
// short string sizes fit in a 4-bit integer.
#define NPY_SHORT_STRING_SIZE_MASK 0x0F  // 0000 1111
#define NPY_SHORT_STRING_MAX_SIZE \
    (sizeof(npy_static_string) - 1)      // 15 or 7 depending on arch
#define NPY_MEDIUM_STRING_MAX_SIZE 0xFF  // 255

#define STRING_FLAGS(string) \
    (((_npy_static_string_u *)string)->direct_buffer.size_and_flags & NPY_STRING_FLAG_MASK)
#define SHORT_STRING_SIZE(string) \
    (string->direct_buffer.size_and_flags & NPY_SHORT_STRING_SIZE_MASK)
#define HIGH_BYTE_MASK ((size_t)0XFF << 8 * (sizeof(size_t) - 1))
#define VSTRING_SIZE(string) (string->vstring.size_and_flags & ~HIGH_BYTE_MASK)

// Since this has no flags set, technically this is an uninitialized,
// arena-allocated medium-sized string with size zero. Practically, that
// doesn't matter because it is flagged as uninitialized.
// The nice part of this choice is a calloc'd array buffer (e.g. from
// np.empty) is filled with empty elements for free.
static const _npy_static_string_u empty_string_u = {
        .direct_buffer = {.size_and_flags = 0, .buf = {0}}};

static int
is_short_string(const npy_packed_static_string *s)
{
    // Initialized and outside arena, but not missing, or long.
    return STRING_FLAGS(s) == (NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA);
}

typedef struct npy_string_arena {
    size_t cursor;
    size_t size;
    char *buffer;
} npy_string_arena;

struct npy_string_allocator {
    npy_string_malloc_func malloc;
    npy_string_free_func free;
    npy_string_realloc_func realloc;
    npy_string_arena arena;
#if PY_VERSION_HEX < 0x30d00b3
    PyThread_type_lock *allocator_lock;
#else
    PyMutex allocator_lock;
#endif
};

static void
set_vstring_size(_npy_static_string_u *str, size_t size)
{
    unsigned char current_flags = str->direct_buffer.size_and_flags;
    str->vstring.size_and_flags = size;
    str->direct_buffer.size_and_flags = current_flags;
}

static char *
vstring_buffer(npy_string_arena *arena, _npy_static_string_u *string)
{
    if (STRING_FLAGS(string) & NPY_STRING_OUTSIDE_ARENA) {
        return (char *)string->vstring.offset;
    }
    if (arena->buffer == NULL) {
        return NULL;
    }
    return (char *)((size_t)arena->buffer + string->vstring.offset);
}

#define ARENA_EXPAND_FACTOR 1.25

static char *
arena_malloc(npy_string_arena *arena, npy_string_realloc_func r, size_t size)
{
    // one extra size_t to store the size of the allocation
    size_t string_storage_size;
    if (size <= NPY_MEDIUM_STRING_MAX_SIZE) {
        string_storage_size = size + sizeof(unsigned char);
    }
    else {
        string_storage_size = size + sizeof(size_t);
    }
    if ((arena->size - arena->cursor) <= string_storage_size) {
        // realloc the buffer so there is enough room
        // first guess is to double the size of the buffer
        size_t newsize;
        if (arena->size == 0) {
            newsize = string_storage_size;
        }
        else if (((ARENA_EXPAND_FACTOR * arena->size) - arena->cursor) >
                 string_storage_size) {
            newsize = ARENA_EXPAND_FACTOR * arena->size;
        }
        else {
            newsize = arena->size + string_storage_size;
        }
        if ((arena->cursor + size) >= newsize) {
            // need extra room beyond the expansion factor, leave some padding
            newsize = ARENA_EXPAND_FACTOR * (arena->cursor + size);
        }
        // passing a NULL buffer to realloc is the same as malloc
        char *newbuf = r(arena->buffer, newsize);
        if (newbuf == NULL) {
            return NULL;
        }
        memset(newbuf + arena->cursor, 0, newsize - arena->cursor);
        arena->buffer = newbuf;
        arena->size = newsize;
    }
    char *ret;
    if (size <= NPY_MEDIUM_STRING_MAX_SIZE) {
        unsigned char *size_loc =
                (unsigned char *)&arena->buffer[arena->cursor];
        *size_loc = size;
        ret = &arena->buffer[arena->cursor + sizeof(char)];
    }
    else {
        char *size_ptr = (char *)&arena->buffer[arena->cursor];
        memcpy(size_ptr, &size, sizeof(size_t));
        ret = &arena->buffer[arena->cursor + sizeof(size_t)];
    }
    arena->cursor += string_storage_size;
    return ret;
}

static int
arena_free(npy_string_arena *arena, _npy_static_string_u *str)
{
    size_t size = VSTRING_SIZE(str);
    // When calling this function, string and arena should not be empty.
    assert (size > 0);
    assert (!(arena->size == 0 && arena->cursor == 0 && arena->buffer == NULL));

    char *ptr = vstring_buffer(arena, str);
    if (ptr == NULL) {
        return -1;
    }
    uintptr_t buf_start = (uintptr_t)arena->buffer;
    uintptr_t ptr_loc = (uintptr_t)ptr;
    uintptr_t end_loc = ptr_loc + size;
    uintptr_t buf_end = buf_start + arena->size;
    if (ptr_loc < buf_start || ptr_loc > buf_end || end_loc > buf_end) {
        return -1;
    }

    memset(ptr, 0, size);

    return 0;
}

static npy_string_arena NEW_ARENA = {0, 0, NULL};

NPY_NO_EXPORT npy_string_allocator *
NpyString_new_allocator(npy_string_malloc_func m, npy_string_free_func f,
                        npy_string_realloc_func r)
{
    npy_string_allocator *allocator = m(sizeof(npy_string_allocator));
    if (allocator == NULL) {
        return NULL;
    }
#if PY_VERSION_HEX < 0x30d00b3
    PyThread_type_lock *allocator_lock = PyThread_allocate_lock();
    if (allocator_lock == NULL) {
        f(allocator);
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate thread lock");
        return NULL;
    }
    allocator->allocator_lock = allocator_lock;
#else
    memset(&allocator->allocator_lock, 0, sizeof(PyMutex));
#endif
    allocator->malloc = m;
    allocator->free = f;
    allocator->realloc = r;
    // arena buffer gets allocated in arena_malloc
    allocator->arena = NEW_ARENA;

    return allocator;
}

NPY_NO_EXPORT void
NpyString_free_allocator(npy_string_allocator *allocator)
{
    npy_string_free_func f = allocator->free;

    if (allocator->arena.buffer != NULL) {
        f(allocator->arena.buffer);
    }
#if PY_VERSION_HEX < 0x30d00b3
    if (allocator->allocator_lock != NULL) {
        PyThread_free_lock(allocator->allocator_lock);
    }
#endif

    f(allocator);
}

/*NUMPY_API
 * Acquire the mutex locking the allocator attached to *descr*.
 *
 * NpyString_release_allocator must be called on the allocator returned
 * by this function exactly once.
 *
 * Note that functions requiring the GIL should not be called while the
 * allocator mutex is held, as doing so may cause deadlocks.
 */
NPY_NO_EXPORT npy_string_allocator *
NpyString_acquire_allocator(const PyArray_StringDTypeObject *descr)
{
#if PY_VERSION_HEX < 0x30d00b3
    if (!PyThread_acquire_lock(descr->allocator->allocator_lock, NOWAIT_LOCK)) {
        PyThread_acquire_lock(descr->allocator->allocator_lock, WAIT_LOCK);
    }
#else
    PyMutex_Lock(&descr->allocator->allocator_lock);
#endif
    return descr->allocator;
}

/*NUMPY_API
 * Simultaneously acquire the mutexes locking the allocators attached to
 * multiple descriptors.
 *
 * Writes a pointer to the associated allocator in the allocators array for
 * each StringDType descriptor in the array. If any of the descriptors are not
 * StringDType instances, write NULL to the allocators array for that entry.
 *
 * *n_descriptors* is the number of descriptors in the descrs array that
 * should be examined. Any descriptor after *n_descriptors* elements is
 * ignored. A buffer overflow will happen if the *descrs* array does not
 * contain n_descriptors elements.
 *
 * If pointers to the same descriptor are passed multiple times, only acquires
 * the allocator mutex once but sets identical allocator pointers appropriately.
 *
 * The allocator mutexes must be released after this function returns, see
 * NpyString_release_allocators.
 *
 * Note that functions requiring the GIL should not be called while the
 * allocator mutex is held, as doing so may cause deadlocks.
 */
NPY_NO_EXPORT void
NpyString_acquire_allocators(size_t n_descriptors,
                             PyArray_Descr *const descrs[],
                             npy_string_allocator *allocators[])
{
    for (size_t i=0; i<n_descriptors; i++) {
        if (NPY_DTYPE(descrs[i]) != &PyArray_StringDType) {
            allocators[i] = NULL;
            continue;
        }
        int allocators_match = 0;
        for (size_t j=0; j<i; j++) {
            if (allocators[j] == NULL) {
                continue;
            }
            if (((PyArray_StringDTypeObject *)descrs[i])->allocator ==
                ((PyArray_StringDTypeObject *)descrs[j])->allocator)
            {
                allocators[i] = allocators[j];
                allocators_match = 1;
                break;
            }
        }
        if (!allocators_match) {
            allocators[i] = NpyString_acquire_allocator(
                    (PyArray_StringDTypeObject *)descrs[i]);
        }
    }
}

/*NUMPY_API
 * Release the mutex locking an allocator. This must be called exactly once
 * after acquiring the allocator mutex and all operations requiring the
 * allocator are done.
 *
 * If you need to release multiple allocators, see
 * NpyString_release_allocators, which can correctly handle releasing the
 * allocator once when given several references to the same allocator.
 */
NPY_NO_EXPORT void
NpyString_release_allocator(npy_string_allocator *allocator)
{
#if PY_VERSION_HEX < 0x30d00b3
    PyThread_release_lock(allocator->allocator_lock);
#else
    PyMutex_Unlock(&allocator->allocator_lock);
#endif
}

/*NUMPY_API
 * Release the mutexes locking N allocators.
 *
 * *length* is the length of the allocators array. NULL entries are ignored.
 *
 * If pointers to the same allocator are passed multiple times, only releases
 * the allocator mutex once.
 */
NPY_NO_EXPORT void
NpyString_release_allocators(size_t length, npy_string_allocator *allocators[])
{
    for (size_t i=0; i<length; i++) {
        if (allocators[i] == NULL) {
            continue;
        }
        int matches = 0;
        for (size_t j=0; j<i; j++) {
            if (allocators[i] == allocators[j]) {
                matches = 1;
                break;
            }
        }
        if (!matches) {
            NpyString_release_allocator(allocators[i]);
        }
    }
}

static const char EMPTY_STRING[] = "";

/*NUMPY_API
 * Extract the packed contents of *packed_string* into *unpacked_string*.
 *
 * The *unpacked_string* is a read-only view onto the *packed_string* data and
 * should not be used to modify the string data. If *packed_string* is the
 * null string, sets *unpacked_string.buf* to the NULL pointer. Returns -1 if
 * unpacking the string fails, returns 1 if *packed_string* is the null
 * string, and returns 0 otherwise.
 *
 * A useful pattern is to define a stack-allocated npy_static_string instance
 * initialized to {0, NULL} and pass a pointer to the stack-allocated unpacked
 * string to this function.  This function can be used to simultaneously
 * unpack a string and determine if it is a null string.
 */
NPY_NO_EXPORT int
NpyString_load(npy_string_allocator *allocator,
               const npy_packed_static_string *packed_string,
               npy_static_string *unpacked_string)
{
    if (NpyString_isnull(packed_string)) {
        unpacked_string->size = 0;
        unpacked_string->buf = NULL;
        return 1;
    }

    _npy_static_string_u *string_u = (_npy_static_string_u *)packed_string;

    if (is_short_string(packed_string)) {
        unpacked_string->size = SHORT_STRING_SIZE(string_u);
        unpacked_string->buf = string_u->direct_buffer.buf;
    }

    else {
        size_t size = VSTRING_SIZE(string_u);
        const char *buf = EMPTY_STRING;
        if (size > 0) {
            npy_string_arena *arena = &allocator->arena;
            if (arena == NULL) {
                return -1;
            }
            buf = vstring_buffer(arena, string_u);
            if (buf == NULL) {
                return -1;
            }
        }
        unpacked_string->size = size;
        unpacked_string->buf = buf;
    }

    return 0;
}

// Helper for allocating strings that will live on the heap or in the arena
// buffer. Determines whether this is a newly allocated array and the string
// should be appended to an existing arena buffer, new data for an existing
// arena string that is being mutated, or new data for an existing short
// string that is being mutated
static char *
heap_or_arena_allocate(npy_string_allocator *allocator,
                       _npy_static_string_u *to_init_u, size_t size,
                       int *on_heap)
{
    unsigned char *flags = &to_init_u->direct_buffer.size_and_flags;
    if (!(*flags & NPY_STRING_OUTSIDE_ARENA)) {
        // Arena allocation or re-allocation.
        npy_string_arena *arena = &allocator->arena;
        if (arena == NULL) {
            return NULL;
        }
        if (*flags == 0) {
            // string isn't previously allocated, so add to existing arena allocation
            char *ret = arena_malloc(arena, allocator->realloc, sizeof(char) * size);
            if (size < NPY_MEDIUM_STRING_MAX_SIZE) {
                *flags = NPY_STRING_INITIALIZED;
            }
            else {
                *flags = NPY_STRING_INITIALIZED | NPY_STRING_LONG;
            }
            return ret;
        }
        // String was in arena. See if there is still space.
        // The size is stored "behind" the beginning of the allocation.
        char *buf = vstring_buffer(arena, to_init_u);
        if (buf == NULL) {
            return NULL;
        }
        size_t alloc_size;
        if (*flags & NPY_STRING_LONG) {
            // Long string size not necessarily memory-aligned, so use memcpy.
            size_t *size_loc = (size_t *)((uintptr_t)buf - sizeof(size_t));
            memcpy(&alloc_size, size_loc, sizeof(size_t));
        }
        else {
            // medium string size is stored in a char so direct access is OK.
            alloc_size = (size_t) * (unsigned char *)(buf - 1);
        }
        if (size <= alloc_size) {
            // we have room!
            return buf;
        }
        // No room, fall through to heap allocation.
    }
    // Heap allocate
    *on_heap = 1;
    *flags = NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA | NPY_STRING_LONG;
    return allocator->malloc(sizeof(char) * size);
}

static int
heap_or_arena_deallocate(npy_string_allocator *allocator,
                         _npy_static_string_u *str_u)
{
    assert (VSTRING_SIZE(str_u) > 0); // should not get here with empty string.

    if (STRING_FLAGS(str_u) & NPY_STRING_OUTSIDE_ARENA) {
        // It's a heap string (not in the arena buffer) so it needs to be
        // deallocated with free(). For heap strings the offset is a raw
        // address so this cast is safe.
        allocator->free((char *)str_u->vstring.offset);
        str_u->vstring.offset = 0;
    }
    else {
        // In the arena buffer.
        npy_string_arena *arena = &allocator->arena;
        if (arena == NULL) {
            return -1;
        }
        if (arena_free(arena, str_u) < 0) {
            return -1;
        }
    }
    return 0;
}

// A regular empty string is just a short string with a size of 0.  But if a
// string was already initialized on the arena, we just set the vstring size to
// 0, so that we still can use the arena if the string gets reset again.
NPY_NO_EXPORT int
NpyString_pack_empty(npy_packed_static_string *out)
{
    _npy_static_string_u *out_u = (_npy_static_string_u *)out;
    unsigned char *flags = &out_u->direct_buffer.size_and_flags;
    if (*flags & NPY_STRING_OUTSIDE_ARENA) {
        // This also sets short string size to 0.
        *flags = NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA;
    }
    else {
        set_vstring_size(out_u, 0);
    }
    return 0;
}

NPY_NO_EXPORT int
NpyString_newemptysize(size_t size, npy_packed_static_string *out,
                       npy_string_allocator *allocator)
{
    if (size == 0) {
        return NpyString_pack_empty(out);
    }
    if (size > NPY_MAX_STRING_SIZE) {
        return -1;
    }

    _npy_static_string_u *out_u = (_npy_static_string_u *)out;

    if (size > NPY_SHORT_STRING_MAX_SIZE) {
        int on_heap = 0;
        char *buf = heap_or_arena_allocate(allocator, out_u, size, &on_heap);

        if (buf == NULL) {
            return -1;
        }

        if (on_heap) {
            out_u->vstring.offset = (size_t)buf;
        }
        else {
            npy_string_arena *arena = &allocator->arena;
            if (arena == NULL) {
                return -1;
            }
            out_u->vstring.offset = (size_t)buf - (size_t)arena->buffer;
        }
        set_vstring_size(out_u, size);
    }
    else {
        // Size can be no larger than 7 or 15, depending on CPU architecture.
        // In either case, the size data is in at most the least significant 4
        // bits of the byte so it's safe to | with the flags.  All other
        // metadata for the previous allocation are wiped, since setting the
        // short string will overwrite the previous size and offset.
        out_u->direct_buffer.size_and_flags =
            NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA | size;
    }

    return 0;
}

NPY_NO_EXPORT int
NpyString_newsize(const char *init, size_t size,
                  npy_packed_static_string *to_init,
                  npy_string_allocator *allocator)
{
    if (NpyString_newemptysize(size, to_init, allocator) < 0) {
        return -1;
    }

    if (size == 0) {
        return 0;
    }

    _npy_static_string_u *to_init_u = ((_npy_static_string_u *)to_init);

    char *buf = NULL;

    if (size > NPY_SHORT_STRING_MAX_SIZE) {
        buf = vstring_buffer(&allocator->arena, to_init_u);
    }
    else {
        buf = to_init_u->direct_buffer.buf;
    }

    memcpy(buf, init, size);

    return 0;
}

NPY_NO_EXPORT int
NpyString_free(npy_packed_static_string *str, npy_string_allocator *allocator)
{
    _npy_static_string_u *str_u = (_npy_static_string_u *)str;
    unsigned char *flags = &str_u->direct_buffer.size_and_flags;
    // Unconditionally remove flag indicating something was missing.
    // For that case, the string should have been deallocated already, but test
    // anyway in case we later implement the option to mask the string.
    *flags &= ~NPY_STRING_MISSING;
    if (is_short_string(str)) {
        if (SHORT_STRING_SIZE(str_u) > 0) {
            // zero buffer and set flags for initialized out-of-arena empty string.
            memcpy(str_u, &empty_string_u, sizeof(_npy_static_string_u));
            *flags = NPY_STRING_OUTSIDE_ARENA | NPY_STRING_INITIALIZED;
        }
    }
    else {
        if (VSTRING_SIZE(str_u) > 0) {
            // Deallocate string and set size to 0 to indicate that happened.
            if (heap_or_arena_deallocate(allocator, str_u) < 0) {
                return -1;
            }
            set_vstring_size(str_u, 0);
        }
    }
    return 0;
}

/*NUMPY_API
 * Pack the null string into a npy_packed_static_string
 *
 * Pack the null string into *packed_string*. Returns 0 on success and -1
 * on failure.
*/
NPY_NO_EXPORT int
NpyString_pack_null(npy_string_allocator *allocator,
                    npy_packed_static_string *packed_string)
{
    _npy_static_string_u *str_u = (_npy_static_string_u *)packed_string;
    if (NpyString_free(packed_string, allocator) < 0) {
        return -1;
    }
    // preserve the flags because we allow mutation, so we need metadata about
    // the original allocation associated with this string in order to
    // determine if there is space in the arena allocation for a new string
    // after a user mutates this one to a non-NULL value.
    str_u->direct_buffer.size_and_flags |= NPY_STRING_MISSING;
    return 0;
}

NPY_NO_EXPORT int
NpyString_dup(const npy_packed_static_string *in,
              npy_packed_static_string *out,
              npy_string_allocator *in_allocator,
              npy_string_allocator *out_allocator)
{
    if (NpyString_isnull(in)) {
        return NpyString_pack_null(out_allocator, out);
    }
    size_t size = NpyString_size(in);
    if (size == 0) {
        return NpyString_pack_empty(out);
    }
    if (is_short_string(in)) {
        memcpy(out, in, sizeof(_npy_static_string_u));
        return 0;
    }
    _npy_static_string_u *in_u = (_npy_static_string_u *)in;
    char *in_buf = NULL;
    npy_string_arena *arena = &in_allocator->arena;
    int used_malloc = 0;
    if (in_allocator == out_allocator && !is_short_string(in)) {
        in_buf = in_allocator->malloc(size);
        memcpy(in_buf, vstring_buffer(arena, in_u), size);
        used_malloc = 1;
    }
    else {
        in_buf = vstring_buffer(arena, in_u);
    }
    int ret =
            NpyString_newsize(in_buf, size, out, out_allocator);
    if (used_malloc) {
        in_allocator->free(in_buf);
    }
    return ret;
}

NPY_NO_EXPORT int
NpyString_isnull(const npy_packed_static_string *s)
{
    return (STRING_FLAGS(s) & NPY_STRING_MISSING) == NPY_STRING_MISSING;
}

NPY_NO_EXPORT int
NpyString_cmp(const npy_static_string *s1, const npy_static_string *s2)
{
    size_t minsize = s1->size < s2->size ? s1->size : s2->size;

    int cmp = 0;

    if (minsize != 0) {
        cmp = strncmp(s1->buf, s2->buf, minsize);
    }

    if (cmp == 0) {
        if (s1->size > minsize) {
            return 1;
        }
        if (s2->size > minsize) {
            return -1;
        }
    }

    return cmp;
}

NPY_NO_EXPORT size_t
NpyString_size(const npy_packed_static_string *packed_string)
{
    if (NpyString_isnull(packed_string)) {
        return 0;
    }

    _npy_static_string_u *string_u = (_npy_static_string_u *)packed_string;

    if (is_short_string(packed_string)) {
        return string_u->direct_buffer.size_and_flags &
               NPY_SHORT_STRING_SIZE_MASK;
    }

    return VSTRING_SIZE(string_u);
}

/*NUMPY_API
 * Pack a string buffer into a npy_packed_static_string
 *
 * Copy and pack the first *size* entries of the buffer pointed to by *buf*
 * into the *packed_string*. Returns 0 on success and -1 on failure.
*/
NPY_NO_EXPORT int
NpyString_pack(npy_string_allocator *allocator,
               npy_packed_static_string *packed_string, const char *buf,
               size_t size)
{
    if (NpyString_free(packed_string, allocator) < 0) {
        return -1;
    }
    return NpyString_newsize(buf, size, packed_string, allocator);
}

NPY_NO_EXPORT int
NpyString_share_memory(const npy_packed_static_string *s1, npy_string_allocator *a1,
                       const npy_packed_static_string *s2, npy_string_allocator *a2) {
    if (a1 != a2 ||
        is_short_string(s1) || is_short_string(s2) ||
        NpyString_isnull(s1) || NpyString_isnull(s2)) {
        return 0;
    }

    _npy_static_string_u *s1_u = (_npy_static_string_u *)s1;
    _npy_static_string_u *s2_u = (_npy_static_string_u *)s2;

    if (vstring_buffer(&a1->arena, s1_u) == vstring_buffer(&a2->arena, s2_u))
    {
        return 1;
    }
    return 0;
}
