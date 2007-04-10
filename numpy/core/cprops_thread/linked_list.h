#ifndef _CP_LINKEDLIST_H
#define _CP_LINKEDLIST_H

/** @{ */
/**
 * @file
 *
 * linked list definitions
 */

#include "common.h"

__BEGIN_DECLS

#include "cp_config.h"
#include "collection.h"
#include "mempool.h"

/**
 * Internal object that references the content and links to the neighbour
 * entries.
 */
typedef CPROPS_DLL struct _cp_list_entry
{
    void *item; /**< stored element (content) */
    struct _cp_list_entry *next; /**< link to next entry */
    struct _cp_list_entry *prev; /**< link to previous entry */
} cp_list_entry;

/**
 * doubly linked list type. 
 */
typedef CPROPS_DLL struct _cp_list
{
	cp_list_entry *head;  		/**< link to beginning of list */
	cp_list_entry *tail;  		/**< link to end of list       */

	cp_compare_fn compare_fn;  	/**< comparison method 		   */
	cp_copy_fn copy_fn; 		/**< copy method */
	cp_destructor_fn free_fn; 	/**< item destructor */

	int mode;                 	/**< operation mode (see collection.h) */
	cp_thread txowner; 			/**< current lock owner */

	long items;               	/**< number of elements in list */

	int is_view; 				/**< views don't have their own lock */
	cp_lock *lock;    			/**< lock */
	int txtype;                 /**< lock type */

	cp_mempool *mempool; 		/**< optional memory pool */
} cp_list;

/**
 * iterator helper-class of cp_list.
 */
typedef CPROPS_DLL struct _cp_list_iterator
{
	cp_list *list;           /**< link to the list */
	cp_list_entry **pos;     /**< current position */

	int lock_type;           /**< locking mode */
} cp_list_iterator;


/** Default constructor */
CPROPS_DLL 
cp_list *cp_list_create();

CPROPS_DLL 
cp_list *cp_list_create_nosync();

/**
 * Constructor
 *
 * @param mode operation mode bitmap (see collection.h)
 * @param compare_fn  compare method
 * @param copy_fn copy method
 */
CPROPS_DLL 
cp_list *cp_list_create_list(int mode, 
							 cp_compare_fn compare_fn, 
							 cp_copy_fn copy_fn, 
							 cp_destructor_fn item_destructor);

CPROPS_DLL 
cp_list *cp_list_create_view(int mode, 
							 cp_compare_fn compare_fn, 
							 cp_copy_fn copy_fn,
							 cp_destructor_fn item_destructor,
							 cp_lock *lock);

/**
 * Destroy the object with the mode stored in the list.
 */
CPROPS_DLL 
void cp_list_destroy(cp_list *);

/**
 * Destroy the object with the specified mode (override default).
 */
CPROPS_DLL 
void cp_list_destroy_by_option(cp_list *list, int option);

/**
 * Destroy the object and all contained elements.
 * For each element the method cp_destructor_fn is called. 
 */
CPROPS_DLL 
void cp_list_destroy_custom(cp_list *list, cp_destructor_fn fn);

/**
 * Insert a new element at the beginning of the list.
 * The operation is synchronized according to the properties of the object.
 */
CPROPS_DLL 
void *cp_list_insert(cp_list *list, void *item);

/**
 * Remove the element from the list.
 * The operation is synchronized according to the properties of the object.
 */
CPROPS_DLL 
void *cp_list_remove(cp_list *list, void *item);

/**
 * Insert the element after an existing one.
 */
CPROPS_DLL 
void *cp_list_insert_after(cp_list *list, void *item, void *existing);

/**
 * Insert the element before an existing one.
 */
CPROPS_DLL 
void *cp_list_insert_before(cp_list *list, void *item, void *existing);

/**
 * Get the first element that equals the parameter.
 */
CPROPS_DLL 
void *cp_list_search(cp_list *list, void *item);

/**
 * run a callback on each item. Stops if the callback function returns
 * non-zero.
 */
CPROPS_DLL 
int cp_list_callback(cp_list *l, int (*item_action)(void *, void *), void *id);

/**
 * Append the element at the end of the list.
 *
 * @retval item the appended item.
 * @retval existing_item if multiple values not allowed and an equal item already exists. 
 */
CPROPS_DLL 
void *cp_list_append(cp_list *list, void *item);

/**
 * Returns the first element of the list.
 */
CPROPS_DLL 
void *cp_list_get_head(cp_list *list);

/**
 * Returns the last element of the list.
 */
CPROPS_DLL 
void *cp_list_get_tail(cp_list *list);

/**
 * remove and release first entry
 *
 * @return previous list head
 */
CPROPS_DLL 
void *cp_list_remove_head(cp_list *list);

/**
 * remove and release last entry 
 *
 * @return Element that was stored in the last entry.
 */
CPROPS_DLL 
void *cp_list_remove_tail(cp_list *list);

/**
 * Test if object is empty.
 *
 * @retval true if no element contained.
 * @retval false if at least one element is contained.
 */
CPROPS_DLL 
int cp_list_is_empty(cp_list *list);

/**
 * Get the number of elements in the collection.
 * 
 * @return number of elements in the list.
 * @retval 0 if list is NULL
 */
CPROPS_DLL 
long cp_list_item_count(cp_list *);

/**
 * Locks the collection with the specified mode.
 *
 * This overrides the default mode stored in the object.
 */
CPROPS_DLL 
int cp_list_lock(cp_list *list, int mode);

/**
 * Set a read lock on the object.
 */
#define cp_list_rdlock(list) cp_list_lock(list, COLLECTION_LOCK_READ)

/**
 * Set a write lock on the object.
 */
#define cp_list_wrlock(list) cp_list_lock(list, COLLECTION_LOCK_WRITE)

/**
 * Unlock the object.
 */
CPROPS_DLL 
int cp_list_unlock(cp_list *list);
	
/* set list to use given mempool or allocate a new one if pool is NULL */
CPROPS_DLL
int cp_list_use_mempool(cp_list *list, cp_mempool *pool);

/* set list to use a shared memory pool */
CPROPS_DLL
int cp_list_share_mempool(cp_list *list, cp_shared_mempool *pool);


/**
 * Initialize the Iterator at the first position.
 *
 * Set the iterator at the beginning of the list and lock the list in the
 * mode specified in type.
 *
 * @param iterator the iterator object
 * @param list the list to iterate over
 * @param lock_mode locking mode to use
 * @retval return-code of the aquired lock
 * @retval 0 if no locking
 */
CPROPS_DLL 
int cp_list_iterator_init(cp_list_iterator *iterator, cp_list *list, int lock_mode);

/**
 * Initialize the Iterator at the end.
 *
 * Set the iterator at the end of the list and lock the list in the
 * mode specified in type.
 *
 * @param iterator the iterator object
 * @param list the list to iterate over
 * @param lock_mode locking mode to use
 * @retval return-code of the aquired lock
 * @retval 0 if no locking
 */
CPROPS_DLL 
int cp_list_iterator_init_tail(cp_list_iterator *iterator, cp_list *list, int lock_mode);

/**
 * create a new iterator and initialize it at the beginning of the list.
 *
 * @param list the list to iterate over
 * @param lock_mode locking mode to use
 * @return new iterator object
 */
CPROPS_DLL 
cp_list_iterator* cp_list_create_iterator(cp_list *list, int lock_mode);

/**
 * Move the iterator to the beginning of the list.
 */
CPROPS_DLL 
int cp_list_iterator_head(cp_list_iterator *iterator);

/**
 * Move the iterator to the end of the list.
 */
CPROPS_DLL 
int cp_list_iterator_tail(cp_list_iterator *iterator);

CPROPS_DLL 
int cp_list_iterator_destroy(cp_list_iterator *iterator);

/**
 * unlock the list the iterator is operating on.
 */
CPROPS_DLL 
int cp_list_iterator_release(cp_list_iterator *iterator);

/**
 * Go to the next entry in the list and return the content.
 * 
 * @return object of the next entry.
 * @retval NULL if reading beyond end or from empty list.
 */
CPROPS_DLL 
void *cp_list_iterator_next(cp_list_iterator *iterator);

/**
 * Go to the previous entry in the list and return the content.
 * 
 * @return object of the previous entry.
 * @retval NULL if reading beyond beginning or from empty list.
 */
CPROPS_DLL 
void *cp_list_iterator_prev(cp_list_iterator *iterator);

/**
 * returns the value at the current iterator position
 * 
 * @return value at the current position.
 * @retval NULL if list is empty.
 */
CPROPS_DLL 
void *cp_list_iterator_curr(cp_list_iterator *iterator);


/**
 * insert item to the list just before the current iterator position. In the 
 * special case that the iterator has been moved beyond the list end the new
 * item is added at the end of the list. 
 *
 * @return the added item or NULL if the list mode is not 
 * 		   & COLLECTION_MODE_NOSYNC and the iterator does not own a write lock
 */
CPROPS_DLL 
void *cp_list_iterator_insert(cp_list_iterator *iterator, void *item);

/**
 * append item to the list just after the current iterator position. In the 
 * special case that the iterator has been moved beyond the list head the new
 * item is added at the head of the list. 
 *
 * @return the added item or NULL if the list mode is not 
 * 		   & COLLECTION_MODE_NOSYNC and the iterator does not own a write lock
 */
CPROPS_DLL 
void *cp_list_iterator_append(cp_list_iterator *iterator, void *item);

/**
 * delete the item at the current iterator position.
 *
 * @return the deleted item or NULL the list is empty, if the iterator points
 * 		   beyond list limits or if the list mode is not 
 * 		   & COLLECTION_MODE_NOSYNC and the iterator does not own a write lock
 */
CPROPS_DLL 
void *cp_list_iterator_remove(cp_list_iterator *iterator);

__END_DECLS

/** @} */

#endif
