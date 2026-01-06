/*
 * This functionality is designed specifically for the ufunc machinery to
 * dispatch based on multiple DTypes.  Since this is designed to be used
 * as purely a cache, it currently does no reference counting.
 * Even though this is a cache, there is currently no maximum size.  It may
 * make sense to limit the size, or count collisions:  If too many collisions
 * occur, we could grow the cache, otherwise, just replace an old item that
 * was presumably not used for a long time.
 *
 * If a different part of NumPy requires a custom hashtable, the code should
 * be reused with care since specializing it more for the ufunc dispatching
 * case is likely desired.
 */

#include "npy_hashtable.h"

#include "templ_common.h"
#include <stdatomic.h>



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
find_item_buckets(struct buckets *buckets, int key_len, PyObject *const *key)
{
    Py_hash_t hash = identity_list_hash(key, key_len);
    npy_uintp perturb = (npy_uintp)hash;
    npy_intp mask = buckets->size - 1;
    npy_intp bucket = (npy_intp)hash & mask;

    while (1) {
        PyObject **item = &(buckets->array[bucket * (key_len + 1)]);
        PyObject *value = FT_ATOMIC_LOAD_PTR_ACQUIRE(item[0]);
        if (value == NULL) {
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
find_item(PyArrayIdentityHash const *tb, PyObject *const *key)
{
    struct buckets *buckets = FT_ATOMIC_LOAD_PTR_ACQUIRE(tb->buckets);
    return find_item_buckets(buckets, tb->key_len, key);
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
    npy_intp new_size, prev_size = old_buckets->size;
    assert(prev_size > 0);

    if ((old_buckets->nelem + 1) * 2 > prev_size) {
        /* Double in size */
        new_size = prev_size * 2;
    }
    else {
        new_size = prev_size;
        while ((old_buckets->nelem + 8) * 2 < new_size / 2) {
            /*
             * Should possibly be improved.  However, we assume that we
             * almost never shrink.  Still if we do, do not shrink as much
             * as possible to avoid growing right away.
             */
            new_size /= 2;
        }
        assert(new_size >= 4);
    }
    if (new_size == prev_size) {
        return 0;
    }

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
            PyObject **tb_item = find_item_buckets(new_buckets, key_len, item + 1);
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
 * Add an item to the identity cache.  The storage location must not change
 * unless the cache is cleared.
 *
 * @param tb The mapping.
 * @param key The key, must be a C-array of pointers of the length
 *        corresponding to the mapping.
 * @param value Normally a Python object, no reference counting is done.
 *        use NULL to clear an item.  If the item does not exist, no
 *        action is performed for NULL.
 * @param replace If 1, allow replacements. If replace is 0 an error is raised
 *        if the stored value is different from the value to be cached. If the
 *        value to be cached is identical to the stored value, the value to be
 *        cached is ignored and no error is raised.
 * @returns 0 on success, -1 with a MemoryError or RuntimeError (if an item
 *        is added which is already in the cache and replace is 0).  The
 *        caller should avoid the RuntimeError.
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

    PyObject **tb_item = find_item(tb, key);
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
    PyObject **tb_item = find_item(tb, key);
    return FT_ATOMIC_LOAD_PTR_ACQUIRE(tb_item[0]);
}
