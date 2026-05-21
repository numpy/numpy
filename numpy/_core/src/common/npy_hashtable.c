/* Lock-free hash table implementation for identity based keys
 * (C arrays of pointers) used for ufunc dispatching cache.
 *
 * This cache does not do any reference counting of the stored objects,
 * and the stored pointers must remain valid while in the cache.
 * The cache entries cannot be changed or deleted once added, only new
 * entries can be added. It is thread safe and lock-free for reading, and
 * uses a mutex for writing (adding new entries). See below for the details
 * of thread safety.
 *
 * The actual hash table is stored in the `buckets` struct which contains
 * a flexible array member for the keys and values. It avoids multiple
 * atomic operations as resizing the hash table only requires a single atomic
 * store to swap in the new buckets pointer.
 *
 * Thread safety notes for free-threading builds:
 * - Reading from the cache (getting items) is lock-free and thread safe.
 *   The reader reads the current `buckets` pointer using an atomic load
 *   with memory_order_acquire order. This ensures that the reader
 *   synchronizes with any concurrent writers that may be resizing the cache.
 *   The value of item is then read using an atomic load with memory_order_acquire
 *   order so that it sees the key written by the writer before the value.
 *
 * - Writing to the cache (adding new items) uses ``tb->mutex`` mutex to
 *   ensure only one thread writes at a time. The new items are added
 *   concurrently with readers and synchronized using atomic operations.
 *   The key is stored first (using memcpy), and then the value is stored
 *   using an atomic store with memory_order_release order so that
 *   the store of key is visible to readers that see the value.
 *
 * - Resizing the cache uses the same mutex to ensure only one thread
 *   resizes at a time. The new larger cache is built while holding the
 *   mutex, and then swapped in using an atomic operation. Because,
 *   readers can be reading from the old cache while the new one is
 *   swapped in, the old cache is not free immediately. Instead, it is
 *   kept in a linked list of old caches using the `prev` pointer in the
 *   `buckets` struct. The old caches are only freed when the identity
 *   hash table is deallocated, ensuring that no readers are using them
 *   anymore.
 */

#include "npy_hashtable.h"

#include "templ_common.h"
#include <stdatomic.h>

// It is defined here instead of header to avoid flexible array member warning in C++.
struct buckets {
    struct buckets *prev; /* linked list of old buckets */
    npy_intp size;        /* current size */
    npy_intp nelem;       /* number of elements */
    PyObject *array[];    /* array of keys and values */
};

#if SIZEOF_PY_UHASH_T > 4
#define _NpyHASH_XXPRIME_1 ((Py_uhash_t)11400714785074694791ULL)
#define _NpyHASH_XXPRIME_2 ((Py_uhash_t)14029467366897019727ULL)
#define _NpyHASH_XXPRIME_5 ((Py_uhash_t)2870177450012600261ULL)
#define _NpyHASH_XXROTATE(x) ((x << 31) | (x >> 33))  /* Rotate left 31 bits */
#else
#define _NpyHASH_XXPRIME_1 ((Py_uhash_t)2654435761UL)
#define _NpyHASH_XXPRIME_2 ((Py_uhash_t)2246822519UL)
#define _NpyHASH_XXPRIME_5 ((Py_uhash_t)374761393UL)
#define _NpyHASH_XXROTATE(x) ((x << 13) | (x >> 19))  /* Rotate left 13 bits */
#endif

#ifdef Py_GIL_DISABLED
#define FT_ATOMIC_LOAD_PTR_ACQUIRE(ptr) \
    atomic_load_explicit((_Atomic(void *) *)&(ptr), memory_order_acquire)
#define FT_ATOMIC_STORE_PTR_RELEASE(ptr, val) \
    atomic_store_explicit((_Atomic(void *) *)&(ptr), (void *)(val), memory_order_release)
#else
#define FT_ATOMIC_LOAD_PTR_ACQUIRE(ptr) (ptr)
#define FT_ATOMIC_STORE_PTR_RELEASE(ptr, val) (ptr) = (val)
#endif

/*
 * This hashing function is basically the Python tuple hash with the type
 * identity hash inlined. The tuple hash itself is a reduced version of xxHash.
 *
 * Users cannot control pointers, so we do not have to worry about DoS attacks?
 */
static inline Py_hash_t
identity_list_hash(PyObject *const *v, int len)
{
    Py_uhash_t acc = _NpyHASH_XXPRIME_5;
    for (int i = 0; i < len; i++) {
        /*
         * Lane is the single item hash, which for us is the rotated pointer.
         * Identical to the python type hash (pointers end with 0s normally).
         */
        size_t y = (size_t)v[i];
        Py_uhash_t lane = (y >> 4) | (y << (8 * SIZEOF_VOID_P - 4));
        acc += lane * _NpyHASH_XXPRIME_2;
        acc = _NpyHASH_XXROTATE(acc);
        acc *= _NpyHASH_XXPRIME_1;
    }
    return acc;
}
#undef _NpyHASH_XXPRIME_1
#undef _NpyHASH_XXPRIME_2
#undef _NpyHASH_XXPRIME_5
#undef _NpyHASH_XXROTATE


static inline PyObject **
find_item_buckets(struct buckets *buckets, int key_len, PyObject *const *key,
                  PyObject **pvalue)
{
    Py_hash_t hash = identity_list_hash(key, key_len);
    npy_uintp perturb = (npy_uintp)hash;
    npy_intp mask = buckets->size - 1;
    npy_intp bucket = (npy_intp)hash & mask;

    while (1) {
        PyObject **item = &(buckets->array[bucket * (key_len + 1)]);
        PyObject *val = FT_ATOMIC_LOAD_PTR_ACQUIRE(item[0]);
        if (pvalue != NULL) {
            *pvalue = val;
        }
        if (val == NULL) {
            /* The item is not in the cache; return the empty bucket */
            return item;
        }
        if (memcmp(item+1, key, key_len * sizeof(PyObject *)) == 0) {
            /* This is a match, so return the item/bucket */
            return item;
        }
        /* Hash collision, perturb like Python (must happen rarely!) */
        perturb >>= 5;  /* Python uses the macro PERTURB_SHIFT == 5 */
        bucket = mask & (bucket * 5 + perturb + 1);
    }
}


static inline PyObject **
find_item(PyArrayIdentityHash const *tb, PyObject *const *key, PyObject **pvalue)
{
    struct buckets *buckets = FT_ATOMIC_LOAD_PTR_ACQUIRE(tb->buckets);
    return find_item_buckets(buckets, tb->key_len, key, pvalue);
}


NPY_NO_EXPORT PyArrayIdentityHash *
PyArrayIdentityHash_New(int key_len)
{
    PyArrayIdentityHash *res = (PyArrayIdentityHash *)PyMem_Malloc(sizeof(PyArrayIdentityHash));
    if (res == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    assert(key_len > 0);
    res->key_len = key_len;

    npy_intp initial_size = 4;  /* Start with a size of 4 */

    res->buckets = PyMem_Calloc(1, sizeof(struct buckets)
                                    + initial_size * (key_len + 1) * sizeof(PyObject *));
    if (res->buckets == NULL) {
        PyErr_NoMemory();
        PyMem_Free(res);
        return NULL;
    }
    res->buckets->prev = NULL;
    res->buckets->size = initial_size;
    res->buckets->nelem = 0;

#ifdef Py_GIL_DISABLED
    res->mutex = (PyMutex){0};
#endif
    return res;
}


NPY_NO_EXPORT void
PyArrayIdentityHash_Dealloc(PyArrayIdentityHash *tb)
{
    struct buckets *b = tb->buckets;
#ifdef Py_GIL_DISABLED
    // free all old buckets
    while (b != NULL) {
        struct buckets *prev = b->prev;
        PyMem_Free(b);
        b = prev;
    }
#else
    assert(b->prev == NULL);
    PyMem_Free(b);
#endif
    PyMem_Free(tb);
}


static int
_resize_if_necessary(PyArrayIdentityHash *tb)
{
#ifdef Py_GIL_DISABLED
    assert(PyMutex_IsLocked(&tb->mutex));
#endif
    struct buckets *old_buckets = tb->buckets;
    int key_len = tb->key_len;
    npy_intp prev_size = old_buckets->size;
    assert(prev_size > 0);

    if ((old_buckets->nelem + 1) * 2 <= old_buckets->size) {
        /* No resize necessary if load factor is not more than 0.5 */
        return 0;
    }

    /* Double in size */
    npy_intp new_size = old_buckets->size * 2;

    npy_intp alloc_size;
    if (npy_mul_sizes_with_overflow(&alloc_size, new_size, key_len + 1)) {
        return -1;
    }
    struct buckets *new_buckets = (struct buckets *)PyMem_Calloc(
        1, sizeof(struct buckets) + alloc_size * sizeof(PyObject *));
    if (new_buckets == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    new_buckets->size = new_size;
    new_buckets->nelem = 0;
    for (npy_intp i = 0; i < prev_size; i++) {
        PyObject **item = &old_buckets->array[i * (key_len + 1)];
        if (item[0] != NULL) {
            PyObject **tb_item = find_item_buckets(new_buckets, key_len, item + 1, NULL);
            memcpy(tb_item+1, item+1, key_len * sizeof(PyObject *));
            new_buckets->nelem++;
            tb_item[0] = item[0];
        }
    }
#ifdef Py_GIL_DISABLED
    new_buckets->prev = old_buckets;
#else
    PyMem_Free(old_buckets);
#endif
    FT_ATOMIC_STORE_PTR_RELEASE(tb->buckets, new_buckets);
    return 0;
}


/**
 * Set an item in the identity hash table if it does not already exist.
 * If it does exist, return the existing item.
 *
 * @param tb The mapping.
 * @param key The key, must be a C-array of pointers of the length
 *        corresponding to the mapping.
 * @param value Normally a Python object, no reference counting is done
 *        and it should not be NULL.
 * @param result The resulting value, either the existing one or the
 *        newly added value.
 * @returns 0 on success, -1 with a MemoryError set on failure.
 */
static inline int
PyArrayIdentityHash_SetItemDefaultLockHeld(PyArrayIdentityHash *tb,
        PyObject *const *key, PyObject *default_value, PyObject **result)
{
#ifdef Py_GIL_DISABLED
    assert(PyMutex_IsLocked(&tb->mutex));
#endif
    assert(default_value != NULL);
    if (_resize_if_necessary(tb) < 0) {
        return -1;
    }

    PyObject **tb_item = find_item(tb, key, NULL);
    if (tb_item[0] == NULL) {
        memcpy(tb_item+1, key, tb->key_len * sizeof(PyObject *));
        tb->buckets->nelem++;
        FT_ATOMIC_STORE_PTR_RELEASE(tb_item[0], default_value);
        *result = default_value;
    } else {
        *result = tb_item[0];
    }

    return 0;
}

NPY_NO_EXPORT int
PyArrayIdentityHash_SetItemDefault(PyArrayIdentityHash *tb,
        PyObject *const *key, PyObject *default_value, PyObject **result)
{
#ifdef Py_GIL_DISABLED
    PyMutex_Lock(&tb->mutex);
#endif
    int ret = PyArrayIdentityHash_SetItemDefaultLockHeld(tb, key, default_value, result);
#ifdef Py_GIL_DISABLED
    PyMutex_Unlock(&tb->mutex);
#endif
    return ret;
}


NPY_NO_EXPORT PyObject *
PyArrayIdentityHash_GetItem(PyArrayIdentityHash *tb, PyObject *const *key)
{
    PyObject *value = NULL;
    find_item(tb, key, &value);
    return value;
}
