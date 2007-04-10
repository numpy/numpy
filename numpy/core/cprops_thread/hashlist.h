#ifndef _CP_HASHLIST_H
#define _CP_HASHLIST_H

/** @{ */
/**
 * @file
 *
 * a mapping traversable by iterator - who could want more. access elements 
 * sequentially or by key. 
 */
/* ----------------------------------------------------------------- */

#include "common.h"

__BEGIN_DECLS

#include "cp_config.h"

#include "collection.h"
#include "hashtable.h"

/*************/

typedef CPROPS_DLL struct _cp_hashlist_entry
{
    void *key;                         /**< key of stored element */
    void *value;                       /**< stored element (content) */
    struct _cp_hashlist_entry *next;      /**< next entry */
    struct _cp_hashlist_entry *prev;      /**< previous entry */
    struct _cp_hashlist_entry *bucket;    /**< next record in bucket */
} cp_hashlist_entry;


/**
 * Main object that holds the endpoints of the hash-list and the hash-table.
 * 
 * It also stores the hash, compare and copy methods and the default
 * operation mode.
 */
typedef CPROPS_DLL struct _cp_hashlist
{
    cp_hashlist_entry **table;    /**< table of entries in the hash-table */
    cp_hashlist_entry *head;      /**< first item in the list */
    cp_hashlist_entry *tail;      /**< last item in the list */
    unsigned long table_size;     /**< size of the hash-table */
    unsigned long items;          /**< number of items in the collection */

    cp_hashfunction  hash_fn;     /**< pointer to hash function */
    cp_compare_fn    compare_fn;  /**< pointer to compare function */
    cp_copy_fn       copy_key; 	  /**< pointer to key copy function */
    cp_copy_fn       copy_value;  /**< pointer to value copy function */
	cp_destructor_fn free_key;
	cp_destructor_fn free_value;

    int mode;                     /**< operation mode (see collection.h) */
    cp_lock *lock;				  /**< lock */
	cp_thread txowner;			  /**< lock owner */
	int txtype;                   /**< lock type */

	unsigned long min_size;
	int fill_factor_min;		  /**< minimal fill factor in percent */
	int fill_factor_max;		  /**< maximal fill factor in percent */
} cp_hashlist;

/**
 * iterator for hashlists
 */
typedef CPROPS_DLL struct _cp_hashlist_iterator
{
    cp_hashlist *list;        /**< link to the list */
    cp_hashlist_entry **pos;   /**< current position */

    int lock_type;           /**< locking mode */
} cp_hashlist_iterator;


/*
 * Removes the entry with matching key and destroys it. 
 *
 * @param list the object
 * @param key  Key which is searched.
 * @param mode locking mode
 * @retval value that was stored in the entry.
 * @retval NULL if key was not found
 */
CPROPS_DLL 
void *cp_hashlist_remove_by_option(cp_hashlist *list, void *key, int mode);

/**
 * Removes the first entry and destroys it. 
 *
 * @param list the object
 * @param mode locking mode
 * @return Element that was stored in the first entry.
 */
CPROPS_DLL 
void *cp_hashlist_remove_head_by_option(cp_hashlist *list, int mode);

/**
 * Removes the last entry and destroys it. 
 *
 * @param list the object
 * @param mode locking mode
 * @return Element that was stored in the first entry.
 */
CPROPS_DLL 
void *cp_hashlist_remove_tail_by_option(cp_hashlist *list, int mode);

/*************/

/** default constructor */
#define cp_hashlist_create(size_hint, hash_fn, compare_fn) \
        cp_hashlist_create_by_option(0, (size_hint), \
									 (hash_fn), (compare_fn), \
									 NULL, NULL, NULL, NULL)

/**
 * constructor with parameters.
 *
 * @param operation mode bitmap (see collection.h)
 * @param size_hint initial size of the cp_hashtable 
 * @param hash_fn hash method  
 * @param compare_fn  hash compare method  
 */
CPROPS_DLL 
cp_hashlist *cp_hashlist_create_by_mode(int mode, 
										unsigned long size_hint, 
										cp_hashfunction hash_fn, 
										cp_compare_fn compare_fn);

/**
 * Constructor for copy mode.
 *
 * @param mode Bitmap of operation mode (see collection.h)
 * @param size_hint initial  size of the cp_hashtable 
 * @param hash_fn  hash method  
 * @param compare_fn   hash compare method  
 * @param copy_key   copy key method  
 * @param copy_value copy value method  
 */
CPROPS_DLL 
cp_hashlist *
	cp_hashlist_create_by_option(int mode, unsigned long size_hint, 
		                   		 cp_hashfunction hash_fn, 
								 cp_compare_fn compare_fn,
								 cp_copy_fn copy_key, 
								 cp_destructor_fn free_key,
								 cp_copy_fn copy_value,
								 cp_destructor_fn free_value);

/**
 * Destroy the list with the mode stored in the list.
 */
CPROPS_DLL 
void cp_hashlist_destroy(cp_hashlist *);

/**
 * Destroy the list with the mode stored in the list plus COLLECTION_MODE_DEEP.
 */
CPROPS_DLL 
void cp_hashlist_destroy_deep(cp_hashlist *);

/**
 * Destroy the object with the specified mode (override default).
 *
 * @param list the object
 * @param mode locking mode
 */
CPROPS_DLL 
void cp_hashlist_destroy_by_option(cp_hashlist *list, int mode);

/**
 * This function does exactly what you would think it does. Before it didn't - 
 * that was a bug. Now it does.
 */
CPROPS_DLL 
void cp_hashlist_destroy_custom(cp_hashlist *list, 
								cp_destructor_fn dk, 
								cp_destructor_fn dv);

/**
 * iterates over the list and calls the callback function on each item.
 * 
 * @param list the hashlist
 * @param dk if not NULL, this function is invoked with the key for each entry
 * @param dv if not NULL, this function is invoked with the value for each entry
 */
CPROPS_DLL 
int cp_hashlist_callback(cp_hashlist *list, 
						 int (*cb)(void *key, void *value, void *id),
						 void *id);

/**  find out what mode your cp_hashlist is running in */
CPROPS_DLL 
int cp_hashlist_get_mode(cp_hashlist *list);

/** set the mode on your cp_hashlist */
CPROPS_DLL 
int cp_hashlist_set_mode(cp_hashlist *list, int mode);

/** unset mode bits on list */
CPROPS_DLL 
int cp_hashlist_unset_mode(cp_hashlist *list, int mode);


/**
 * the internal table will not be resized to less than min_size
 */
CPROPS_DLL 
int cp_hashlist_set_min_size(cp_hashlist *list, unsigned long min_size);

/**
 * a resize is triggered when the table contains more items than
 * table_size * fill_factor / 100
 */
CPROPS_DLL 
int cp_hashlist_set_max_fill_factor(cp_hashlist *list, int fill_factor);

/**
 * a resize is triggered when the table contains less items than
 * table_size * fill_factor / 100 if table_size > min_size
 */
CPROPS_DLL 
int cp_hashlist_set_min_fill_factor(cp_hashlist *list, int fill_factor);


/* **************************************************************************
 *                                                                          *
 *                            iterator functions                            *
 *                                                                          *
 ************************************************************************** */

/**
 * Create a new iterator and initialize it at the beginning.
 *
 * @param list list to iterate over
 * @param lock_mode locking mode to use
 * @return new iterator object
 */
CPROPS_DLL 
cp_hashlist_iterator *cp_hashlist_create_iterator(cp_hashlist *list, int lock_mode);

/**
 * initialize the iterator at the beginning
 *
 * set the iterator at the beginning of the list and lock the list in the
 * mode specified in type.
 */
CPROPS_DLL 
int cp_hashlist_iterator_head(cp_hashlist_iterator *iterator);
//int cp_hashlist_iterator_head(cp_hashlist_iterator *iterator, cp_hashlist *l, int lock_mode);

CPROPS_DLL 
int cp_hashlist_iterator_init(cp_hashlist_iterator *iterator, 
							  cp_hashlist *list, int type);
/**
 * Initialize the Iterator at the end.
 *
 * Set the iterator at the end of the list and lock the list in the
 * mode specified in type.
 */
CPROPS_DLL 
int cp_hashlist_iterator_init_tail(cp_hashlist_iterator *iterator, cp_hashlist *l, int lock_mode);

/**
 * set iterator at list tail
 */
CPROPS_DLL 
int cp_hashlist_iterator_tail(cp_hashlist_iterator *iterator);

/**
 * set iterator position at first occurence of given key
 */
CPROPS_DLL 
int cp_hashlist_iterator_to_key(cp_hashlist_iterator *iterator, void *key);

/**
 * iterator destructor
 */
CPROPS_DLL 
int cp_hashlist_iterator_destroy(cp_hashlist_iterator *iterator);

/**
 * Unlock the list of the Iterator.
 *
 * If the locking mode is COLLECTION_LOCK_NONE, do nothing.
 */
CPROPS_DLL 
int cp_hashlist_iterator_release(cp_hashlist_iterator *iterator);


/**
 * Go to the next entry in the list and return the content.
 * 
 * @retval entry the next entry object.
 * @retval NULL if reading beyond end or from empty list.
 */
CPROPS_DLL 
cp_hashlist_entry *cp_hashlist_iterator_next(cp_hashlist_iterator *iterator);

/**
 * Go to the next entry in the list and return the key.
 * 
 * @return object of the next entry.
 * @retval NULL if reading beyond end or from empty list.
 */
CPROPS_DLL 
void *cp_hashlist_iterator_next_key(cp_hashlist_iterator *iterator);

/**
 * Go to the next entry in the list and return the content.
 * 
 * @return object of the next entry.
 * @retval NULL if reading beyond end or from empty list.
 */
CPROPS_DLL 
void *cp_hashlist_iterator_next_value(cp_hashlist_iterator *iterator);

/**
 * Go to the previous entry in the list and return the content.
 * 
 * @retval entry the previous entry object.
 * @retval NULL if reading beyond beginning or from empty list.
 */
CPROPS_DLL 
cp_hashlist_entry *cp_hashlist_iterator_prev(cp_hashlist_iterator *iterator);

/**
 * Go to the previous entry in the list and return the key.
 * 
 * @return object of the previous entry.
 * @retval NULL if reading beyond beginning or from empty list.
 */
CPROPS_DLL 
void *cp_hashlist_iterator_prev_key(cp_hashlist_iterator *iterator);

/**
 * Go to the previous entry in the list and return the content.
 * 
 * @return object of the previous entry.
 * @retval NULL if reading beyond beginning or from empty list.
 */
CPROPS_DLL 
void *cp_hashlist_iterator_prev_value(cp_hashlist_iterator *iterator);

/**
 * return the entry at the current iterator position
 */
CPROPS_DLL 
cp_hashlist_entry *cp_hashlist_iterator_curr(cp_hashlist_iterator *iterator);

/**
 * return the key at the current iterator position
 */
CPROPS_DLL 
void *cp_hashlist_iterator_curr_key(cp_hashlist_iterator *iterator);

/**
 * return the value at the current iterator position
 */
CPROPS_DLL 
void *cp_hashlist_iterator_curr_value(cp_hashlist_iterator *iterator);


/**
 * add a mapping before the current iterator position
 */
CPROPS_DLL 
cp_hashlist_entry *cp_hashlist_iterator_insert(cp_hashlist_iterator *iterator, 
								  			   void *key, 
								  			   void *value);
/**
 * add a mapping after the current iterator position
 */
CPROPS_DLL 
cp_hashlist_entry *cp_hashlist_iterator_append(cp_hashlist_iterator *iterator, 
											   void *key,
											   void *value);

/**
 * remove the mapping at the current iterator position
 */
CPROPS_DLL 
void *cp_hashlist_iterator_remove(cp_hashlist_iterator *iterator);


/* ------------------------ end iterator functions ------------------------ */



/**
 * Get the number of elements in the collection.
 * 
 * @return number of elements in the list.
 * @retval 0 if list is NULL
 */
CPROPS_DLL 
unsigned long cp_hashlist_item_count(cp_hashlist *);

/**
 * Get the key of the entry.
 *
 * @note This function is fail-safe - works also on NULL entries.
 *       In any case you should check the result!
 *
 * @retval key of the entry.
 * @retval NULL if entry is NULL
 */
CPROPS_DLL 
void *cp_hashlist_entry_get_key(cp_hashlist_entry *entry);

/**
 * Get the value of the entry.
 *
 * @note This function is fail-safe - works also on NULL entries.
 *       In any case you should check the result!
 *
 * @retval value of the entry.
 * @retval NULL if entry is NULL
 */
CPROPS_DLL 
void *cp_hashlist_entry_get_value(cp_hashlist_entry *entry);

/**
 * Insert a new element (key, value) at the beginning of the list.
 * The operation is synchronized according to the properties of the object.
 *
 * @pre list != NULL
 * @pre key != NULL
 * @return value object
 */
CPROPS_DLL 
void *cp_hashlist_insert(cp_hashlist *list, void *key, void *value);

/**
 * Insert a new element (key, value) at the beginning of the list with mode.
 * @return value object
 */
CPROPS_DLL 
void *cp_hashlist_insert_by_option(cp_hashlist *list, void *key, void *item, int mode);

/**
 * Append a new element (key, value) at the end of the list.
 * The operation is synchronized according to the properties of the object.
 * @pre list != NULL
 * @pre key != NULL
 */
CPROPS_DLL 
void *cp_hashlist_append(cp_hashlist *list, void *key, void *value);

/**
 * Append a new element (key, value) at the end of the list with mode.
 */
CPROPS_DLL 
void *cp_hashlist_append_by_option(cp_hashlist *, void *key, void *value, int mode);

/**
 * Returns the first element with matching key.
 */
CPROPS_DLL 
void *cp_hashlist_get(cp_hashlist *, void *key);

/**
 * returns non-zero if list contains key
 */
CPROPS_DLL 
int cp_hashlist_contains(cp_hashlist *list, void *key);

/**
 * Returns the first element of the list.
 */
CPROPS_DLL 
void *cp_hashlist_get_head(cp_hashlist *);

/**
 * Returns the last element of the list.
 */
CPROPS_DLL 
void *cp_hashlist_get_tail(cp_hashlist *);

/**
 * Removes the entry with matching key and destroys it (internal locking mode). 
 *
 * @param list the object
 * @param key  Key which is searched.
 * @retval value that was stored in the entry.
 * @retval NULL if key was not found
 */
CPROPS_DLL 
void *cp_hashlist_remove(cp_hashlist *list, void *key);

CPROPS_DLL 
void *cp_hashlist_remove_deep(cp_hashlist *list, void *key);


/**
 * Removes the entry with matching key and destroys it with locking mode.
 *
 * @param list the object
 * @param key  Key which is searched.
 * @param mode locking mode
 * @retval value that was stored in the entry.
 * @retval NULL if key was not found
 */
CPROPS_DLL 
void *cp_hashlist_remove_by_option(cp_hashlist *list, void *key, int mode);

/**
 * Removes the first entry and destroys it. 
 *
 * @return Element that was stored in the first entry.
 */
CPROPS_DLL 
void *cp_hashlist_remove_head(cp_hashlist *list);

/**
 * Removes the last entry and destroys it. 
 *
 * @pre list != NULL
 * @retval Element that was stored in the first entry if not empty.
 * @retval NULL if collection is empty
 */
CPROPS_DLL 
void *cp_hashlist_remove_tail(cp_hashlist *list);

/*
 * Get the next entry. 
 *
 * @return value of the next element
 */
//CPROPS_DLL 
//void *cp_hashlist_get_next(cp_hashlist *list);

/**
 * Test if object is empty.
 *
 * @retval true if no element contained.
 * @retval false if at least one element is contained.
 */
CPROPS_DLL 
int cp_hashlist_is_empty(cp_hashlist *list); //  { return cp_hashlist_item_count(list) == 0; }

/**
 * Locks the collection with the specified mode.
 *
 * This overrides the default mode stored in the object.
 */
CPROPS_DLL 
int cp_hashlist_lock(cp_hashlist *list, int type);

/**
 * Unlock the object.
 */
CPROPS_DLL 
int cp_hashlist_unlock(cp_hashlist *list);

/**
 * Set a read lock on the object.
 */
#define cp_hashlist_rdlock(list) cp_hashlist_lock((list), COLLECTION_LOCK_READ)

/**
 * Set a write lock on the object.
 */
#define cp_hashlist_wrlock(list) cp_hashlist_lock((list), COLLECTION_LOCK_WRITE)

__END_DECLS

/** @} */

#endif

