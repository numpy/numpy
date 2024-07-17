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

#include "templ_common.h"
#include "npy_hashtable.h"



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
// TODO: replace with PyMutex when it is public
#define LOCK_TABLE(tb)                                      \
    if (!PyThread_acquire_lock(tb->mutex, NOWAIT_LOCK)) {   \
        PyThread_acquire_lock(tb->mutex, WAIT_LOCK);        \
    }
#define UNLOCK_TABLE(tb) PyThread_release_lock(tb->mutex);
#define INITIALIZE_LOCK(tb)                     \
    tb->mutex = PyThread_allocate_lock();       \
    if (tb->mutex == NULL) {                    \
        PyErr_NoMemory();                       \
        PyMem_Free(res);                        \
        return NULL;                            \
    }
#define FREE_LOCK(tb)                           \
    if (tb->mutex != NULL) {                    \
        PyThread_free_lock(tb->mutex);          \
    }
#else
// the GIL serializes access to the table so no need
// for locking if it is enabled
#define LOCK_TABLE(tb)
#define UNLOCK_TABLE(tb)
#define INITIALIZE_LOCK(tb)
#define FREE_LOCK(tb)
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
find_item(PyArrayIdentityHash const *tb, PyObject *const *key)
{
    Py_hash_t hash = identity_list_hash(key, tb->key_len);
    npy_uintp perturb = (npy_uintp)hash;
    npy_intp bucket;
    npy_intp mask = tb->size - 1 ;
    PyObject **item;

    bucket = (npy_intp)hash & mask;
    while (1) {
        item = &(tb->buckets[bucket * (tb->key_len + 1)]);

        if (item[0] == NULL) {
            /* The item is not in the cache; return the empty bucket */
            return item;
        }
        if (memcmp(item+1, key, tb->key_len * sizeof(PyObject *)) == 0) {
            /* This is a match, so return the item/bucket */
            return item;
        }
        /* Hash collision, perturb like Python (must happen rarely!) */
        perturb >>= 5;  /* Python uses the macro PERTURB_SHIFT == 5 */
        bucket = mask & (bucket * 5 + perturb + 1);
    }
}


NPY_NO_EXPORT PyArrayIdentityHash *
PyArrayIdentityHash_New(int key_len)
{
    PyArrayIdentityHash *res = PyMem_Malloc(sizeof(PyArrayIdentityHash));
    if (res == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    assert(key_len > 0);
    res->key_len = key_len;
    res->size = 4;  /* Start with a size of 4 */
    res->nelem = 0;

    INITIALIZE_LOCK(res);

    res->buckets = PyMem_Calloc(4 * (key_len + 1), sizeof(PyObject *));
    if (res->buckets == NULL) {
        PyErr_NoMemory();
        PyMem_Free(res);
        return NULL;
    }
    return res;
}


NPY_NO_EXPORT void
PyArrayIdentityHash_Dealloc(PyArrayIdentityHash *tb)
{
    PyMem_Free(tb->buckets);
    FREE_LOCK(tb);
    PyMem_Free(tb);
}


static int
_resize_if_necessary(PyArrayIdentityHash *tb)
{
    npy_intp new_size, prev_size = tb->size;
    PyObject **old_table = tb->buckets;
    assert(prev_size > 0);

    if ((tb->nelem + 1) * 2 > prev_size) {
        /* Double in size */
        new_size = prev_size * 2;
    }
    else {
        new_size = prev_size;
        while ((tb->nelem + 8) * 2 < new_size / 2) {
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
    if (npy_mul_sizes_with_overflow(&alloc_size, new_size, tb->key_len + 1)) {
        return -1;
    }
    tb->buckets = PyMem_Calloc(alloc_size, sizeof(PyObject *));
    if (tb->buckets == NULL) {
        tb->buckets = old_table;
        PyErr_NoMemory();
        return -1;
    }

    tb->size = new_size;
    for (npy_intp i = 0; i < prev_size; i++) {
        PyObject **item = &old_table[i * (tb->key_len + 1)];
        if (item[0] != NULL) {
            PyObject **tb_item = find_item(tb, item + 1);
            tb_item[0] = item[0];
            memcpy(tb_item+1, item+1, tb->key_len * sizeof(PyObject *));
        }
    }
    PyMem_Free(old_table);
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
NPY_NO_EXPORT int
PyArrayIdentityHash_SetItem(PyArrayIdentityHash *tb,
        PyObject *const *key, PyObject *value, int replace)
{
    LOCK_TABLE(tb);
    if (value != NULL && _resize_if_necessary(tb) < 0) {
        /* Shrink, only if a new value is added. */
        UNLOCK_TABLE(tb);
        return -1;
    }

    PyObject **tb_item = find_item(tb, key);
    if (value != NULL) {
        if (tb_item[0] != NULL && tb_item[0] != value && !replace) {
            UNLOCK_TABLE(tb);
            PyErr_SetString(PyExc_RuntimeError,
                    "Identity cache already includes an item with this key.");
            return -1;
        }
        tb_item[0] = value;
        memcpy(tb_item+1, key, tb->key_len * sizeof(PyObject *));
        tb->nelem += 1;
    }
    else {
        /* Clear the bucket -- just the value should be enough though. */
        memset(tb_item, 0, (tb->key_len + 1) * sizeof(PyObject *));
    }

    UNLOCK_TABLE(tb);
    return 0;
}


NPY_NO_EXPORT PyObject *
PyArrayIdentityHash_GetItem(PyArrayIdentityHash const *tb, PyObject *const *key)
{
    LOCK_TABLE(tb);
    PyObject *res = find_item(tb, key)[0];
    UNLOCK_TABLE(tb);
    return res;
}
