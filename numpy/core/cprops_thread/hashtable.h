#ifndef _CP_HASHTABLE_H
#define _CP_HASHTABLE_H

/*
 * hash table implementation
 */

/**
 * @addtogroup cp_hashtable
 * @ingroup collection
 * @copydoc collection
 *
 */
/** @{ */
/**
 * @file
 * generic synchronized hashtable. 
 *
 * Here is an example of using a cp_hashtable to create a lookup for 'bar'
 * items using 'foo' keys:
 *
 * <code><pre>
 * unsigned long foo_hash_code(void *fooptr) 
 * {
 *     return ((foo *) fooptr)->id; 
 * }    
 *
 * // compare function
 * int foo_compare(void *f1, void *f2)
 * {
 *     return ((foo *) f1)->id != ((foo *) f2)->id;
 * }
 *
 * ...
 * 
 *     cp_hashtable *t;
 *     foo *foo1, *f;
 *     bar *bar1, *bar2;
 *     
 *     t = cp_hashtable_create(10, foo_hash_code, foo_compare);
 *     if (t == NULL) 
 *     {
 *         perror("can\'t create cp_hashtable");
 *         exit(1);
 *     }
 * 
 *     cp_hashtable_put(foo1, bar1, t);
 * 
 *     ...
 *     f = foo_create(...);
 *     ...
 * 
 *     if ((bar2 = (bar *) cp_hashtable_get(f, t))) 
 *         printf("%s maps to %s\n", foo_get_name(f), bar_get_name(bar2));
 *     else 
 *         printf("%s is not mapped\n", foo_get_name(f));
 *
 *     ...
 * 
 *     cp_hashtable_destroy(t);
 * </pre></code>
 * 
 * Note the strcmp like semantics of the compare function. The comparison
 * should return 0 for identical keys.
 * <p>
 * @see hash.h (util.collection) for function prototypes 
 * and convenience functions.
 * @see cp_hashtable_create, cp_hashtable_destroy, cp_hashtable_destroy_deep, cp_hashtable_put,
 *      cp_hashtable_get, cp_hashtable_contains, cp_hashtable_remove_deep
 */

#include "common.h"

__BEGIN_DECLS

#include "collection.h"

/*
 * cp_hashtable interface for members of mapping collections.
 */

#include "cp_config.h"

/**
 * 1000000001 is a prime. HASH_SEED is used by cp_hash_string(). 
 */
#define HASH_SEED 1000000001L

#ifndef CP_HASHTABLE_DEFAULT_MIN_FILL_FACTOR
#define CP_HASHTABLE_DEFAULT_MIN_FILL_FACTOR   5
#endif

#ifndef CP_HASHTABLE_DEFAULT_MAX_FILL_FACTOR
#define CP_HASHTABLE_DEFAULT_MAX_FILL_FACTOR  70
#endif

/* ----------------------------------------------------------------- 
 * Function prototypes
 * ----------------------------------------------------------------- */
/**
 * the hash function takes (void *) and returns unsigned long.
 *
 * Create a function with the name <class_name>_hash_code()
 */
typedef unsigned long (*cp_hashfunction)(void *);


/* ------------------------------------------------------------------------ 
 * hash function prototypes for primitives and cp_strings
 * ------------------------------------------------------------------------ */


/**
 * hash function for int keys
 * @param key pointer to the int
 * @return hash code of the key
 */
CPROPS_DLL 
unsigned long cp_hash_int(void *key);


/**
 * comparator for int keys
 * @param key1 pointer the first int
 * @param key2 pointer the second int
 * @retval 0 if key1 equals key2;
 * @retval <0 if key1 is less than key2;
 * @retval >0 if key1 is greater than key2

 */
CPROPS_DLL 
int cp_hash_compare_int(void *key1, void *key2);

/**
 * hash function for long keys
 */
CPROPS_DLL 
unsigned long cp_hash_long(void *key);

/**
 * comparator for long keys
 * 
 * @param key1 pointer the first long
 * @param key2 pointer the second long
 * @retval 0 if key1 equals key2;
 * @retval <0 if key1 is less than key2;
 * @retval >0 if key1 is greater than key2
 */
CPROPS_DLL 
int cp_hash_compare_long(void *key1, void *key2);

/**
 * hash function for pointer keys
 */
CPROPS_DLL 
unsigned long cp_hash_addr(void *addr);

/**
 * comparator for pointer keys
 * 
 * @param key1 pointer the first pointer
 * @param key2 pointer the second pointer
 * @retval 0 if key1 equals key2;
 * @retval <0 if key1 is less than key2;
 * @retval >0 if key1 is greater than key2
 */
CPROPS_DLL 
int cp_hash_compare_addr(void *key1, void *key2);

/**
 * hash function for (char *) keys
 * @param key pointer to the cp_string
 * @return hash code of the key
 */
CPROPS_DLL 
unsigned long cp_hash_string(void *key);

/**
 * case insensitive hash function for (char *) keys
 * @param key pointer to the cp_string
 * @return hash code of the key
 */
CPROPS_DLL 
unsigned long cp_hash_istring(void *key);


/**
 * copy function for cp_string copy tables
 */
CPROPS_DLL 
void *cp_hash_copy_string(void *element);

/**
 * comparator for (char *) keys
 * 
 * @param key1 pointer to the first cp_string
 * @param key2 pointer to the second cp_string
 * @retval 0 if key1 equals key2
 * @retval <>0 otherwise
 */
CPROPS_DLL 
int cp_hash_compare_string(void *key1, void *key2);


/**
 * comparator for (char *) keys
 * 
 * @param key1 pointer to the first cp_string
 * @param key2 pointer to the second cp_string
 * @retval 0 if key1 equals key2
 * @retval <>0 otherwise
 */
CPROPS_DLL 
int cp_hash_compare_istring(void *key1, void *key2);



/**
 * Internal object that implements a key, value pair plus linked list.
 * 
 * Entries which are stored under the same index in the hashtable are stored
 * in a linked list. The algorithm of the hashtable has to ensure that the lists
 * do not get too long.
 */
typedef CPROPS_DLL struct _cp_hashtable_entry 
{
    void *key;                   /**< key (original) needed for comparisons */
    void *value;                 /**< the item value being stored */
    unsigned long hashcode;      /**< save calculated hash-code of the key */
    struct _cp_hashtable_entry *next; /**< link to next entry */
} cp_hashtable_entry;

/**
 * data structure of generic synchronized cp_hashtable
 */
typedef CPROPS_DLL struct _cp_hashtable
{
    cp_hashtable_entry **table;     /**< array of pointers to entries */
    long table_size;                /**< size of the table */

    unsigned long items;            /**< number of items in the table */
    int mode;                       /**< collection mode @see collection.h */
    cp_hashfunction   hash_fn;      /**< pointer to hash function */
    cp_compare_fn     compare_fn;   /**< pointer to compare function */
    cp_copy_fn        copy_key;     /**< pointer to key copy function */
    cp_copy_fn        copy_value;   /**< pointer to value copy function */
	cp_destructor_fn  free_key;
	cp_destructor_fn  free_value;

    cp_lock           *lock;        /**< lock */
	cp_thread 		  txowner;    	/**< lock owner */
	int				  txtype;       /**< lock type */

	int min_size;					/**< table resize lower limit */
	int fill_factor_min;			/**< minimal fill factor in percent */
	int fill_factor_max;			/**< maximal fill factor in percent */

    cp_hashtable_entry **resize_table; /**< temp table for resizing */
    int resizing;                   /**< resize running flag */
    unsigned long resize_len;       /**< resize table length */
    cp_thread resize_thread;        /**< run resize in a separate thread  */
    cp_mutex *resize_lock;			/**< for synchronizing resize operation */
} cp_hashtable;


/**
 * creates a new cp_hashtable.
 *
 * by default there is no memory management for table content; insertion, 
 * removal and retrieval operations are synchronized; and the table will 
 * automatically resize when the fill factor goes over 70% or under 5%.
 * @param size_hint an estimate for the initial storage requirements. The table 
 *
 * handles the storage appropriately when items become too tight. 
 * @param hashfn a hash code function. This should ideally produce different
 * results for different keys.
 * @param compare_fn the comparator for your key type. 
 * 
 * @return a pointer to the newly created cp_hashtable.
 */
CPROPS_DLL 
cp_hashtable *
	cp_hashtable_create(unsigned long size_hint, 
						cp_hashfunction hashfn, 
						cp_compare_fn compare_fn);

/**
 * creates a new cp_hashtable with the specified mode.
 */
#define cp_hashtable_create_by_mode(mode, size_hint, cp_hashfn, compare_fn) \
        cp_hashtable_create_by_option((mode), (size_hint), (cp_hashfn), (compare_fn), NULL, NULL, NULL, NULL)

/**
 * creates a new cp_hashtable with COLLECTION_MODE_DEEP | COLLECTION_MODE_COPY.
 * @param size_hint an estimate for the initial storage requirements. The table 
 * handles the storage appropriately when items become too tight. 
 * @param hashfn a hash code function. This should ideally produce different
 * results for different keys.
 * @param compare_fn the comparator for your key type. 
 * 
 * @return a pointer to the newly created cp_hashtable.
 */
CPROPS_DLL 
cp_hashtable *
	cp_hashtable_create_copy_mode(unsigned long size_hint, 
								  cp_hashfunction hash_fn, 
								  cp_compare_fn compare_fn, 
             					  cp_copy_fn copy_key, 
								  cp_destructor_fn free_key,
								  cp_copy_fn copy_value,
								  cp_destructor_fn free_value);    

/**
 * create a new table, fully specifying all parameters.
 * @param size_hint   initial capacity
 * @param hash_fn     hash function
 * @param compare_fn  key comparison function
 * @param copy_key    function to return new copies of keys
 * @param copy_value  function to return new copies of values
 * @param mode        mode flags
 * 
 * @return new created cp_hashtable. Returns NULL case of error.
 */
CPROPS_DLL 
cp_hashtable *
	cp_hashtable_create_by_option(int mode, unsigned long size_hint, 
								  cp_hashfunction hash_fn, 
								  cp_compare_fn compare_fn, 
								  cp_copy_fn copy_key, 
								  cp_destructor_fn free_key,
								  cp_copy_fn copy_value,
								  cp_destructor_fn free_value);

/**
 * deletes a cp_hashtable according to the current mode settings
 * @param table        object to delete
 */
CPROPS_DLL 
void cp_hashtable_destroy(cp_hashtable *table);

/**
 * deletes a cp_hashtable. Pointers to the keys and values are not released. Use
 * table if the keys and values you entered in the table should not be released
 * by the cp_hashtable.
 * @param table        object to delete
 */
CPROPS_DLL 
void cp_hashtable_destroy_shallow(cp_hashtable *table);

/**
 * deletes a cp_hashtable. Keys and values entered in the cp_hashtable are released.
 * @param table        object to delete
 */
CPROPS_DLL 
void cp_hashtable_destroy_deep(cp_hashtable *table);


/** 
 * Deep destroy with custom destructors for keys and values. NULL function 
 * pointers are not invoked. 
 */
CPROPS_DLL 
void cp_hashtable_destroy_custom(cp_hashtable *table, cp_destructor_fn dk, cp_destructor_fn dv);

/** 
 * by default the get, put and remove functions as well as set and unset mode
 * perform their own locking. Other functions do not synchronize, since it is
 * assumed they would be called in a single cp_thread context - the initialization  * and deletion functions in particular. You can of course set 
 * COLLECTION_MODE_NOSYNC and perform your own synchronization.<p>
 * 
 * The current implementation uses a queued read/write lock where blocked
 * cp_threads are guaranteed to be woken by the order in which they attempted
 *
 * the following macros are defined for convenience:<p>
 * <ul>
 * <li> cp_hashtable_rdlock(table) - get a read lock on the table </li>
 * <li> cp_hashtable_wrlock(table) - get a write lock on the table </li>
 * </ul>
 * @param table        cp_hashtable to lock
 * @param type COLLECTION_LOCK_READ or COLLECTION_LOCK_WRITE
 */ 
CPROPS_DLL 
int cp_hashtable_lock(cp_hashtable *table, int type);

/** unlock the table */
CPROPS_DLL 
int cp_hashtable_unlock(cp_hashtable *table);

/** macro to get a read lock on the table
 */
#define cp_hashtable_rdlock(table) cp_hashtable_lock((table), COLLECTION_LOCK_READ)

/** macro to get a write lock on the table */
#define cp_hashtable_wrlock(table) cp_hashtable_lock((table), COLLECTION_LOCK_WRITE)

/**
 * returns the current operation mode. See cp_hashtable_set_mode for a list of
 * mode bits and their effects.
 */
CPROPS_DLL 
int cp_hashtable_get_mode(cp_hashtable *table);

/**
 * set the operation mode as a bit set of the following options:
 * <ul>
 *  <li> COLLECTION_MODE_DEEP - release memory when removing references from table    </li>
 *  <li> COLLECTION_MODE_MULTIPLE_PUT - allow multiple values for a key    </li>
 *  <li> COLLECTION_MODE_COPY - keep copies rather than references    </li>
 *  <li> COLLECTION_MODE_NOSYNC - the table will not perform its own synchronization.    </li>
 *  <li> COLLECTION_MODE_NORESIZE - the table will not resize automatically.    </li>
 * </ul>
 * 
 * The parameter bits are flipped on. If the current mode is 
 * COLLECTION_MODE_DEEP and you want to change it, call
 * cp_hashtable_unset_mode(table, COLLECTION_MODE_DEEP).
 */
CPROPS_DLL 
int cp_hashtable_set_mode(cp_hashtable *table, int mode);


/**
 * unset the mode bits defined by mode
 */
CPROPS_DLL 
int cp_hashtable_unset_mode(cp_hashtable *table, int mode);


/**
 * the internal table will not be resized to less than min_size
 */
CPROPS_DLL 
int cp_hashtable_set_min_size(cp_hashtable *table, int min_size);

/**
 * a resize is triggered when the table contains more items than
 * table_size * fill_factor / 100
 */
CPROPS_DLL 
int cp_hashtable_set_max_fill_factor(cp_hashtable *table, int fill_factor);

/**
 * a resize is triggered when the table contains less items than
 * table_size * fill_factor / 100
 */
CPROPS_DLL 
int cp_hashtable_set_min_fill_factor(cp_hashtable *table, int fill_factor);

/**
 * attempts to retrieve the value assigned to the key 'key'. To return
 * multiple values the table mode must be set to COLLECTION_MODE_MULTIPLE_VALUES, 
 * otherwise the only first value for the given key will be returned. 
 * @retval (void*)value to the value if found
 * @retval NULL otherwise 
 */
CPROPS_DLL 
void *cp_hashtable_get(cp_hashtable *table, void *key);

/** 
 * retrieve the value or values for key 'key'. the 'mode' parameter sets the
 * mode for the current operation.
 */
CPROPS_DLL 
void *cp_hashtable_get_by_option(cp_hashtable *table, void *key, int mode);

/**
 * Internal put method. 
 */
CPROPS_DLL 
void *cp_hashtable_put_by_option(cp_hashtable *table, void *key, void *value, int mode);

/**
 * the key 'key' will be assigned to the value 'value'. The new value will 
 * override an old value if one exists. The old value will not be deallocated.
 * If you would need the old value to be released call cp_hashtable_put_safe instead.
 */
CPROPS_DLL 
void *cp_hashtable_put(cp_hashtable *table, void *key, void *value);

/**
 * same as cp_hashtable_put(table, key, value) except that an old value is released if it
 * exists.
 */
CPROPS_DLL 
void *cp_hashtable_put_safe(cp_hashtable *table, void *key, void *value);

/**
 * same as cp_hashtable_put(table, key, value) except that it inserts a copy
 * of the key and the value object.
 */
CPROPS_DLL 
void *cp_hashtable_put_copy(cp_hashtable *table, void *key, void *value);

/**
 * Attempts to remove the mapping for key from the table.
 *
 * @param table   the object
 * @param key    Key to search for.
 * @retval value retrieved by the key (that was removed)
 * @retval NULL  if the table does not contain the requested key.
 */
CPROPS_DLL 
void *cp_hashtable_remove(cp_hashtable *table, void *key);

/** remove all entries with current mode */
CPROPS_DLL 
int cp_hashtable_remove_all(cp_hashtable *table);

/**
 * removes a mapping from the table, and deallocates the memory for the mapped
 * key and value.
 * 
 * @param table   the object
 * @param key    Key to search for.
 * @return 1 if the operation was successful, 0 otherwise
 */
CPROPS_DLL 
int cp_hashtable_remove_deep(cp_hashtable *table, void *key);

/**
 * Check if there is an entry with matching key.
 *
 * @param table   the object
 * @param key    Key to search for.
 * @return 1 if table contains key, 0 otherwise
 */
CPROPS_DLL 
int cp_hashtable_contains(cp_hashtable *table, void *key);

/**
 * get an array containing all keys mapped in table table.
 * @note It is the responsibility of the caller to free the returned array. 
 * @note The keys themselves must not be changed or deleted (read-only).
 */
CPROPS_DLL 
void **cp_hashtable_get_keys(cp_hashtable *table);

/**
 * get an array containing all values in the table.
 * @note It is the responsibility of the caller to free the returned array. 
 * @note The values themselves must not be changed or deleted (read-only).
 */
CPROPS_DLL 
void **cp_hashtable_get_values(cp_hashtable *table);

/**
 * Get the number of entries in the collection.
 * @return the number of key mappings currently in the table.
 */
CPROPS_DLL 
unsigned long cp_hashtable_count(cp_hashtable *table);

/**
 * Check if the collection is empty.
 * @retval true/1 if the collection is empty
 * @retval false/0 if the collection has entries
 */
#define cp_hashtable_is_empty(table) (cp_hashtable_count(table) == 0)

/**
 * @return a prime greater than <code>size_request</code>
 */
CPROPS_DLL 
unsigned long cp_hashtable_choose_size(unsigned long size_request);

__END_DECLS
/** @} */
#endif

