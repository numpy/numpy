
/**
 * @addtogroup cp_hashlist
 */
/** @{ */
/**
 * @file
 * Implementation for cp_hashlist Collection with linked elements and hash key
 * and with cp_hashlist_iterator.
 * 
 * @copydoc collection
 */
/* ----------------------------------------------------------------- */

#include <stdio.h> /* debug */
#include <stdlib.h>
#include <errno.h>

#include "collection.h"
//#include "log.h"
#include "common.h"

#include "hashlist.h"
#include "linked_list.h"

#include "cp_config.h"

/* debug */
#ifndef CP_HASHLIST_MULTIPLE_VALUES
#define CP_HASHLIST_MULTIPLE_VALUES 1
#endif


	/* internal methods */

/* internal lock function */
int cp_hashlist_txlock(cp_hashlist *list, int type);
/* internal unlock function */
int cp_hashlist_txunlock(cp_hashlist *list);

static cp_hashlist_entry *
	cp_hashlist_create_entry(cp_hashlist *list, int mode, 
							 void *key, void *value);

static void 
	cp_hashlist_entry_delete(cp_hashlist_entry *entry);

static void cp_hashlist_entry_release_by_option(cp_hashlist *list, 
												cp_hashlist_entry *entry, 
												int mode);
/**
 * insert an entry
 * 
 * @param list the collection object
 * @param entry the entry to insert
 * @return entry the inserted entry
 */
static cp_hashlist_entry *
	cp_hashlist_insert_internal(cp_hashlist *list, cp_hashlist_entry *entry);

/**
 * remove first entry matching key
 * 
 * @param list the collection object
 * @param key that is searched for
 * @retval entry if successful the removed entry
 * @retval NULL  if the entry was not found
 */
static cp_hashlist_entry *
	cp_hashlist_remove_internal(cp_hashlist *list, void *key);

/**
 * remove specified entry
 * 
 * @param list the collection object
 * @param entry the entry to remove
 * @retval entry if successful the passed entry
 * @retval NULL  if the entry was not found
 */
static cp_hashlist_entry *
	cp_hashlist_remove_entry_internal(cp_hashlist *list, 
									  cp_hashlist_entry *entry);

static void *cp_hashlist_get_internal(cp_hashlist *list, void *key);

	/* end internal methods */


cp_hashlist *
	cp_hashlist_create_by_option(int mode, 
								 unsigned long size_hint, 
		               		   	 cp_hashfunction hash_fn, 
                               	 cp_compare_fn compare_fn,
                               	 cp_copy_fn copy_key,
								 cp_destructor_fn free_key,
                               	 cp_copy_fn copy_value,
								 cp_destructor_fn free_value)
{
    cp_hashlist *list = (cp_hashlist *) calloc(1, sizeof(cp_hashlist));
    if (list == NULL) 
	{
        errno = ENOMEM;
        return NULL;
    }

    list->table_size = cp_hashtable_choose_size(size_hint);
    list->items = 0;
    list->table = (cp_hashlist_entry **) calloc(list->table_size, sizeof(cp_hashlist_entry *));
    if (list->table == NULL) 
	{
		errno = ENOMEM;
        return NULL;
    }

    list->hash_fn = hash_fn;
    list->compare_fn = compare_fn;
    list->copy_key = NULL;
    list->copy_value = NULL;

    list->head = list->tail = NULL;
	list->lock = malloc(sizeof(cp_lock));
	if (list->lock == NULL)
	{
		free(list->table);
		free(list);
		errno = ENOMEM;
		return NULL;
	}
    if (!(mode & COLLECTION_MODE_NOSYNC)) 
		if (cp_lock_init(list->lock, NULL))
		{
			free(list->lock);
			free(list->table);
			free(list);
			return NULL;
		}

    list->mode = mode;

    list->copy_key = copy_key;
    list->copy_value = copy_value;
	list->free_key = free_key;
	list->free_value = free_value;

	list->min_size = size_hint;
	list->fill_factor_min = CP_HASHTABLE_DEFAULT_MIN_FILL_FACTOR;
	list->fill_factor_max = CP_HASHTABLE_DEFAULT_MAX_FILL_FACTOR;

    return list;
}


cp_hashlist *cp_hashlist_create_by_mode(int mode, 
										unsigned long size_hint, 
										cp_hashfunction hash_fn, 
										cp_compare_fn compare_fn)
{
	return 
		cp_hashlist_create_by_option(mode, size_hint, hash_fn, compare_fn,
									 NULL, NULL, NULL, NULL);
}

static cp_hashlist_entry *
	cp_hashlist_create_entry(cp_hashlist *list, int mode,
						     void *key, void *value)
{
	cp_copy_fn ck = list->copy_key;
	cp_copy_fn cv = list->copy_value;
    cp_hashlist_entry *entry = 
		(cp_hashlist_entry *) calloc(1, sizeof(cp_hashlist_entry));
	
	if (entry == NULL) 
	{
		errno = ENOMEM;
		return NULL;
	}

    entry->next = entry->prev = entry->bucket = NULL;
    if (mode & COLLECTION_MODE_COPY) 
	{
        entry->key = ck ? (*ck)(key) : key;
        entry->value = cv ? (*cv)(value) : value;
    } 
	else 
	{
        entry->key = key;
        entry->value = value;
    }

    return entry;
}


static void 
	cp_hashlist_destroy_internal(cp_hashlist *list, 
							  	 int mode, 
								 cp_destructor_fn dk,
								 cp_destructor_fn dv)
{
	int syncbit = list->mode & COLLECTION_MODE_NOSYNC;

    if (!syncbit) cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE);
	while (list->head != NULL)
	{
        cp_hashlist_entry *entry = list->head;
		list->head = entry->next;
		if (mode & COLLECTION_MODE_DEEP)
		{
			if (dk) (*dk)(entry->key);
			if (dv) (*dv)(entry->value);
		}
		cp_hashlist_entry_delete(entry);
	}

    if (!syncbit) cp_hashlist_txunlock(list);
	if (list->lock)
	{
        cp_lock_destroy(list->lock);
		free(list->lock);
    }

    free(list->table);
    free(list);
}

void cp_hashlist_destroy(cp_hashlist *list)
{
   	cp_hashlist_destroy_internal(list, list->mode, 
								 list->free_key, list->free_value);
}

void cp_hashlist_destroy_deep(cp_hashlist *list)
{
    cp_hashlist_set_mode(list, COLLECTION_MODE_DEEP);
    cp_hashlist_destroy_custom(list, list->free_key, list->free_value);
}

void cp_hashlist_destroy_by_option(cp_hashlist *list, int mode)
{
	
	if (list) 
		cp_hashlist_destroy_internal(list, mode, 
									 list->free_key, list->free_value);
}

void cp_hashlist_destroy_custom(cp_hashlist *list, 
									   cp_destructor_fn dk, 
									   cp_destructor_fn dv)
{
	if (list)
		cp_hashlist_destroy_internal(list, list->mode, dk, dv);
}

int cp_hashlist_callback(cp_hashlist *list, 
						 int (*cb)(void *key, void *value, void *id),
						 void *id)
{
    cp_hashlist_entry *entry;

	if (list == NULL || cb == NULL) 
	{
		errno = EINVAL;
		return -1;
	}

    if (cp_hashlist_txlock(list, COLLECTION_LOCK_READ)) return -1;

	entry = list->head;
    while (entry != NULL) 
	{
		if ((*cb)(entry->key, entry->value, id) != 0)
			break;
        entry = entry->next;
    }

    cp_hashlist_txunlock(list);
	return 0;
}

int cp_hashlist_get_mode(cp_hashlist *list)
{
	return list->mode;
}

int cp_hashlist_set_mode(cp_hashlist *list, int mode)
{
	int rc = EINVAL;
	if (list)
	{
		int syncbit = list->mode & COLLECTION_MODE_NOSYNC;

		/* can't set NOSYNC in the middle of a transaction */
		if ((list->mode & COLLECTION_MODE_IN_TRANSACTION) && 
			(mode & COLLECTION_MODE_NOSYNC)) return EINVAL;

		syncbit = list->mode & COLLECTION_MODE_NOSYNC;

		if (cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE)) return -1;
		list->mode |= mode;
		if (!syncbit) cp_hashlist_txunlock(list);

		rc = 0;
	}

	return 0;
}

int cp_hashlist_unset_mode(cp_hashlist *list, int mode)
{
	int syncbit;
	if (cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE)) return -1;
	syncbit = list->mode & COLLECTION_MODE_NOSYNC;
    list->mode &= list->mode ^ mode;
   	if (!syncbit) cp_hashlist_txunlock(list);

	return 0;
}

int cp_hashlist_set_min_size(cp_hashlist *list, unsigned long min_size)
{
	if (cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE)) return -1;
	list->min_size = min_size;
	cp_hashlist_txunlock(list);
	return 0;
}


int cp_hashlist_set_max_fill_factor(cp_hashlist *list, int fill_factor)
{
	if (cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE)) return -1;
	list->fill_factor_max = fill_factor;
	cp_hashlist_txunlock(list);
	return 0;
}

int cp_hashlist_set_min_fill_factor(cp_hashlist *list, int fill_factor)
{
	if (cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE)) return -1;
	list->fill_factor_min = fill_factor;
	cp_hashlist_txunlock(list);
	return 0;
}

void *cp_hashlist_resize_nosync(cp_hashlist *list, unsigned long new_size)
{
    unsigned long old_size;
    cp_hashlist_entry **old_table;
    cp_hashlist_entry *entry, *next, **insert;
    unsigned long i, index;

    old_size = list->table_size;
    old_table = list->table;
    
	new_size = cp_hashtable_choose_size(new_size);
	if (old_size == new_size) /* not enough gap to make a change yet */
		return list;
	else
    	list->table_size = new_size;
#ifdef __TRACE__
    DEBUGMSG("resizing table (nosync): %d to %d\n", old_size, list->table_size);
#endif

    list->table = (cp_hashlist_entry **) calloc(list->table_size, sizeof(cp_hashlist_entry *));

    if (list->table == NULL) 
	{
		errno = ENOMEM;
        return NULL;
    }

    for (i = 0; i < old_size; i++) 
	{
        entry = old_table[i];
        while (entry != NULL) 
		{
            index = abs((*list->hash_fn)(entry->key)) % list->table_size;
            next = entry->bucket;
            entry->bucket = NULL;
            insert = &list->table[index];
            while (*insert != NULL) insert = &(*insert)->bucket;
            *insert = entry;
            
            entry = next;
        }
    }

    free(old_table);

    return list;
}

void *cp_hashlist_append(cp_hashlist *list, void *key, void *value)
{
    return cp_hashlist_append_by_option(list, key, value, list->mode);
}

static int cp_hashlist_contains_internal(cp_hashlist *list, void *key)
{
	int rc = 0;
	cp_hashlist_entry *curr;
	int index = abs((*list->hash_fn)(key)) % list->table_size;
    
	curr = list->table[index];

    while (curr != NULL) 
	{
		if ((*list->compare_fn)(key, curr->key) == 0)
		{
			rc = 1;
			break;
		}
		curr = curr->bucket;
	}

	return rc;
}

int cp_hashlist_contains(cp_hashlist *list, void *key)
{
	int rc = 0;

	if (cp_hashlist_txlock(list, COLLECTION_LOCK_READ)) return -1;
	rc = cp_hashlist_contains_internal(list, key);
	cp_hashlist_txunlock(list);

	return rc;
}

void *cp_hashlist_append_by_option(cp_hashlist *list, 
								   void *key, 
								   void *value, 
								   int mode)
{
    cp_hashlist_entry *entry;
    void *res = NULL;
	int rc = 0;

    if (cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;

	if ((mode & COLLECTION_MODE_MULTIPLE_VALUES) == 0
		&& cp_hashlist_contains_internal(list, key)) 
	{
		rc = EINVAL;
		goto DONE;
	}

    if ((entry = cp_hashlist_create_entry(list, mode, key, value)) == NULL)
	{
		rc = ENOMEM;
		goto DONE;
	}

    cp_hashlist_insert_internal(list, entry);

    if (list->tail) 
	{
        entry->prev = list->tail;
        list->tail->next = entry;
        list->tail = entry;
    }
    else list->tail = list->head = entry;

    res = value;

DONE:
    cp_hashlist_txunlock(list);
	if (rc) errno = rc;

    return res;
}


void *cp_hashlist_insert(cp_hashlist *list, void *key, void *value)
{
    return cp_hashlist_insert_by_option(list, key, value, list->mode);
}


void *cp_hashlist_insert_by_option(cp_hashlist *list, void *key, 
								   void *value, int mode)
{
    cp_hashlist_entry *entry;
    void *res = NULL;
	int rc = 0;

    if (cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;

	if ((mode & COLLECTION_MODE_MULTIPLE_VALUES) == 0
		&& cp_hashlist_contains_internal(list, key)) 
	{
		rc = EINVAL;
		goto DONE;
	}

    if ((entry = cp_hashlist_create_entry(list, mode, key, value)) == NULL)
	{
		rc = errno;
		goto DONE;
	}

    cp_hashlist_insert_internal(list, entry);
    if (list->head) 
	{
        entry->next = list->head;
        list->head->prev = entry;
        list->head = entry;
    }
    else list->head = list->tail = entry;

    res = value;

DONE:
    cp_hashlist_txunlock(list);
	if (rc) errno = rc;

    return res;
}
    

void *cp_hashlist_get(cp_hashlist *list, void *key)
{
    void *res = NULL;

    if (cp_hashlist_txlock(list, COLLECTION_LOCK_READ)) return NULL;

    res = cp_hashlist_get_internal(list, key);

#if CP_HASHLIST_MULTIPLE_VALUES
	if (!(list->mode & COLLECTION_MODE_MULTIPLE_VALUES))
#endif
    if (res) res = ((cp_hashlist_entry *) res)->value;

    cp_hashlist_txunlock(list);

    return res;
}

static void *cp_hashlist_unlink_internal(cp_hashlist *list, 
										 cp_hashlist_entry *holder,
										 int mode)
{
	void *res = NULL;

    if (holder) 
	{    
        if (holder->next) 
			holder->next->prev = holder->prev;
        else              
			list->tail = holder->prev;
		
        if (holder->prev) 
			holder->prev->next = holder->next;
        else              
			list->head = holder->next;
		
        res = holder->value;
        cp_hashlist_entry_release_by_option(list, holder, mode);
    }
	
	return res;
}

void *cp_hashlist_remove_by_option(cp_hashlist *list, void *key, int mode)
{
    void *res;
    cp_hashlist_entry *holder;

    if (cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;

    holder = cp_hashlist_remove_internal(list, key);
	res = cp_hashlist_unlink_internal(list, holder, mode);

    cp_hashlist_txunlock(list);

    return res;
}

void *cp_hashlist_remove(cp_hashlist *list, void *key)
{
	return cp_hashlist_remove_by_option(list, key, list->mode);
}

void *cp_hashlist_remove_deep(cp_hashlist *list, void *key)
{
	return 
		cp_hashlist_remove_by_option(list, key, 
									 list->mode | COLLECTION_MODE_DEEP);
}

void *cp_hashlist_get_head(cp_hashlist *list)
{
    void *res;

    if (cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;

    res = list->head ? list->head->value : NULL;
    
    cp_hashlist_txunlock(list);
    return res;
}

void *cp_hashlist_get_tail(cp_hashlist *list)
{
    void *res;
	
    if (cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;

    res = list->tail ? list->tail->value : NULL;
    
    cp_hashlist_txunlock(list);
    return res;
}


void *cp_hashlist_remove_head(cp_hashlist *list)
{
    return cp_hashlist_remove_head_by_option(list, list->mode);
}

void *cp_hashlist_remove_head_by_option(cp_hashlist *list, int mode)
{
    cp_hashlist_entry *entry;
    void *res = NULL;

    if (cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;

    entry = list->head;
    if (entry) 
	{
        if (!cp_hashlist_remove_entry_internal(list, entry))  /* should _not_ happen */
            printf("failed to remove item from cp_hashlist table: %s\n", (char *) entry->value);

        list->head = list->head->next;
        if (list->head) 
			list->head->prev = NULL;
        else
			list->tail = NULL;

        res = entry->value;

        cp_hashlist_entry_release_by_option(list, entry, mode);
    }

    cp_hashlist_txunlock(list);
    return res;
}

void *cp_hashlist_remove_tail(cp_hashlist *list)
{
    return cp_hashlist_remove_tail_by_option(list, list->mode);
}

void *cp_hashlist_remove_tail_by_option(cp_hashlist *list, int mode)
{
    cp_hashlist_entry *entry;
    void *res = NULL;

    if (cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;

    entry = list->tail;
    if (entry) 
	{
        if (!cp_hashlist_remove_entry_internal(list, entry)) /* should _not_ happen */
            printf("failed to remove item from cp_hashlist table: %s\n", (char *) entry->value);

        list->tail = entry->prev;
        if (list->tail) 
			list->tail->next = NULL;
        else
			list->head = NULL;

        res = entry->value;
        cp_hashlist_entry_release_by_option(list, entry, mode);
    }

    cp_hashlist_txunlock(list);
    return res;
}

unsigned long cp_hashlist_item_count(cp_hashlist *list)
{
    unsigned long count;
    if (cp_hashlist_txlock(list, COLLECTION_LOCK_READ)) return -1;

    count = list->items;

    cp_hashlist_txunlock(list);
    return count;
}

    
int cp_hashlist_lock_internal(cp_hashlist *list, int lock_mode)
{
    int rc = -1;

    switch (lock_mode) 
	{
        case COLLECTION_LOCK_READ:
            rc = cp_lock_rdlock(list->lock);
            break;

        case COLLECTION_LOCK_WRITE:
            rc = cp_lock_wrlock(list->lock);
            break;

		case COLLECTION_LOCK_NONE:
			rc = 0;
			break;

        default:
			errno = EINVAL;
            break;
    }

    return rc;
}

int cp_hashlist_unlock_internal(cp_hashlist *list)
{
    return cp_lock_unlock(list->lock);
}

int cp_hashlist_txlock(cp_hashlist *list, int type)
{
	if (list->mode & COLLECTION_MODE_NOSYNC) return 0;
	if (list->mode & COLLECTION_MODE_IN_TRANSACTION && 
		list->txtype == COLLECTION_LOCK_WRITE)
	{
		cp_thread self = cp_thread_self();
		if (cp_thread_equal(self, list->txowner)) return 0;
	}
	return cp_hashlist_lock_internal(list, type);
}

int cp_hashlist_txunlock(cp_hashlist *list)
{
	if (list->mode & COLLECTION_MODE_NOSYNC) return 0;
	if (list->mode & COLLECTION_MODE_IN_TRANSACTION && 
		list->txtype == COLLECTION_LOCK_WRITE)
	{
		cp_thread self = cp_thread_self();
		if (cp_thread_equal(self, list->txowner)) return 0;
	}
	return cp_hashlist_unlock_internal(list);
}

/* lock and set the transaction indicators */
int cp_hashlist_lock(cp_hashlist *list, int type)
{
	int rc;
	if ((list->mode & COLLECTION_MODE_NOSYNC)) return EINVAL;
	if ((rc = cp_hashlist_lock_internal(list, type))) return rc;
	list->txtype = type;
	list->txowner = cp_thread_self();
	list->mode |= COLLECTION_MODE_IN_TRANSACTION;
	return 0;
}

/* unset the transaction indicators and unlock */
int cp_hashlist_unlock(cp_hashlist *list)
{
	cp_thread self = cp_thread_self();
	if (list->txowner == self)
	{
		list->txtype = 0;
		list->txowner = 0;
		list->mode ^= COLLECTION_MODE_IN_TRANSACTION;
	}
	else if (list->txtype == COLLECTION_LOCK_WRITE)
		return -1;
	return cp_hashlist_unlock_internal(list);
}


void *cp_hashlist_entry_get_key(cp_hashlist_entry *entry)
{
    return (entry) ? entry->key : NULL;
}

void *cp_hashlist_entry_get_value(cp_hashlist_entry *entry)
{
    return (entry) ? entry->value : NULL;
}

int cp_hashlist_is_empty(cp_hashlist *list)
{
    return cp_hashlist_item_count(list) == 0;
}


/****************************************************************************
 *                                                                          *
 *                    cp_hashlist_iterator implementation                   *
 *                                                                          *
 ****************************************************************************/
 
cp_hashlist_iterator* cp_hashlist_create_iterator(cp_hashlist *list, int type)
{
    int rc = - 1;
    cp_hashlist_iterator *iterator = 
		(cp_hashlist_iterator *) malloc(sizeof(cp_hashlist_iterator));
	if (iterator == NULL) return NULL;

    iterator->list = list;
    iterator->pos = &list->head;
    iterator->lock_type = type;

    switch (type)
    {
        case COLLECTION_LOCK_READ : 
            rc = cp_hashlist_txlock(list, COLLECTION_LOCK_READ);
            break;

        case COLLECTION_LOCK_WRITE :
            rc = cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE);
            break;
    
        default :
            rc = 0;
    }

	if (rc) /* locking failed */
	{
		free(iterator);
		iterator = NULL;
	}

    return iterator;
}

int cp_hashlist_iterator_init(cp_hashlist_iterator *iterator, 
							  cp_hashlist *list, int type)
{
    int rc;
    iterator->list = list;
    iterator->pos = &list->head;
    iterator->lock_type = type;

    switch (type)
    {
        case COLLECTION_LOCK_READ : 
            rc = cp_hashlist_txlock(list, COLLECTION_LOCK_READ);
            break;

        case COLLECTION_LOCK_WRITE :
            rc = cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE);
            break;
    
        default :
            rc = 0;
    }

    return rc;
}


int cp_hashlist_iterator_head(cp_hashlist_iterator *iterator)
{
    if (iterator == NULL) return -1;
    iterator->pos = &iterator->list->head;

    return 0;
}

int cp_hashlist_iterator_tail(cp_hashlist_iterator *iterator)
{
    if (iterator == NULL) return -1;
    iterator->pos = &iterator->list->tail;

    return 0;
}

int cp_hashlist_iterator_init_tail(cp_hashlist_iterator *iterator, cp_hashlist *list, int type)
{
    int rc;
    iterator->list = list;
    iterator->pos = &list->tail;
    iterator->lock_type = type;

    switch (type)
    {
        case COLLECTION_LOCK_READ : 
            rc = cp_hashlist_txlock(list, COLLECTION_LOCK_READ);
            break;

        case COLLECTION_LOCK_WRITE :
            rc = cp_hashlist_txlock(list, COLLECTION_LOCK_WRITE);
            break;
    
        default :
            rc = 0;
    }

    return rc;
}

int cp_hashlist_iterator_to_key(cp_hashlist_iterator *iterator, void *key)
{
	cp_hashlist_entry *entry = NULL;
	
#if CP_HASHLIST_MULTIPLE_VALUES
	if ((iterator->list->mode & COLLECTION_MODE_MULTIPLE_VALUES))
	{
		cp_list *res = cp_hashlist_get_internal(iterator->list, key);
		if (res)
		{
			entry = cp_list_get_head(res);
			cp_list_destroy(res);
		}
	}
	else
#endif
	entry = cp_hashlist_get_internal(iterator->list, key);

	if (entry == NULL) return -1;

	if (entry->prev) 
		iterator->pos = &entry->prev->next;
	else 
		iterator->pos = &iterator->list->head;

	return 0;
}


int cp_hashlist_iterator_release(cp_hashlist_iterator *iterator)
{
	int rc = 0;
    if (iterator->lock_type != COLLECTION_LOCK_NONE) 
		rc = cp_hashlist_txunlock(iterator->list);

    return rc;
}

int cp_hashlist_iterator_destroy(cp_hashlist_iterator *iterator)
{
    int rc = cp_hashlist_iterator_release(iterator);
    free(iterator);

    return rc;
}

cp_hashlist_entry *cp_hashlist_iterator_next(cp_hashlist_iterator *iterator)
{
    cp_hashlist_entry *entry = NULL;

    if (*(iterator->pos)) 
	{
        entry = (*iterator->pos);
        iterator->pos = &(*(iterator->pos))->next;
    }
	else if (iterator->list->head && 
			 iterator->pos == &iterator->list->head->prev)
	{
		entry = iterator->list->head;
		iterator->pos = &iterator->list->head;
	}

    return entry;
}

void *cp_hashlist_iterator_next_key(cp_hashlist_iterator *iterator)
{
    return cp_hashlist_entry_get_key(cp_hashlist_iterator_next(iterator));
}

void *cp_hashlist_iterator_next_value(cp_hashlist_iterator *iterator)
{
    return cp_hashlist_entry_get_value(cp_hashlist_iterator_next(iterator));
}

cp_hashlist_entry *cp_hashlist_iterator_prev(cp_hashlist_iterator *iterator)
{
    cp_hashlist_entry  *entry = NULL;

    if (*iterator->pos) 
	{
        entry = (*iterator->pos);
        iterator->pos = &(*iterator->pos)->prev;
    }
	else if (iterator->list->tail && 
			 iterator->pos == &iterator->list->tail->next)
	{
		entry = iterator->list->tail;
		iterator->pos = &iterator->list->tail;
	}

    return entry;
}

void *cp_hashlist_iterator_prev_key(cp_hashlist_iterator *iterator)
{
    return cp_hashlist_entry_get_key(cp_hashlist_iterator_prev(iterator));
}

void *cp_hashlist_iterator_prev_value(cp_hashlist_iterator *iterator)
{
    return cp_hashlist_entry_get_value(cp_hashlist_iterator_prev(iterator));
}

cp_hashlist_entry *cp_hashlist_iterator_curr(cp_hashlist_iterator *iterator)
{
    cp_hashlist_entry *item = NULL;

    if (*iterator->pos)
        item = (*iterator->pos);

    return item;
}

void *cp_hashlist_iterator_curr_key(cp_hashlist_iterator *iterator)
{
    return cp_hashlist_entry_get_key(cp_hashlist_iterator_curr(iterator));
}

void *cp_hashlist_iterator_curr_value(cp_hashlist_iterator *iterator)
{
    return cp_hashlist_entry_get_value(cp_hashlist_iterator_curr(iterator));
}

cp_hashlist_entry *cp_hashlist_iterator_insert(cp_hashlist_iterator *iterator, 
								  			   void *key, 
								  			   void *value)
{
	cp_hashlist_entry *new_entry = NULL;

	if ((iterator->list->mode & COLLECTION_MODE_NOSYNC) || 
		(iterator->lock_type == COLLECTION_LOCK_WRITE))
	{
		cp_hashlist_entry *entry; 
		if ((iterator->list->mode & COLLECTION_MODE_MULTIPLE_VALUES) == 0
			&& cp_hashlist_contains_internal(iterator->list, key)) 
		{
			errno = EINVAL;
			return NULL;
		}
		
		entry = cp_hashlist_create_entry(iterator->list, 
									 	 iterator->list->mode, 
									 	 key, 
									 	 value);
		if (entry == NULL) return NULL;
		cp_hashlist_insert_internal(iterator->list, entry);
		new_entry = entry;
		
		entry->next = *iterator->pos;

		if (*iterator->pos)
		{
			entry->prev = (*iterator->pos)->prev;
			(*iterator->pos)->prev = entry;
			if (entry->prev)
				entry->prev->next = entry;
		}
		else if (iterator->list->head == NULL) /* iterator not pointing at much - list may be empty */
			iterator->list->head = iterator->list->tail = entry;
		else if (iterator->pos == &iterator->list->head->prev) /* iterator moved before head */
		{
			entry->prev = NULL;
			entry->next = iterator->list->head;
			entry->next->prev = entry;
			iterator->list->head = entry;
		}
		else /* iterator moved after tail */
		{
			entry->prev = iterator->list->tail;
			entry->prev->next = entry;
			iterator->list->tail = entry;
		}

		iterator->pos = &entry->next; /* keep iterator on same entry */
		
		iterator->list->items++;
	}
	else /* mode is not NOSYNC and no LOCK_WRITE */
		errno = EINVAL;

	return new_entry;
}

cp_hashlist_entry *cp_hashlist_iterator_append(cp_hashlist_iterator *iterator, 
											   void *key,
											   void *value)
{
	cp_hashlist_entry *new_entry = NULL;

	if ((iterator->list->mode & COLLECTION_MODE_NOSYNC) || 
		(iterator->lock_type == COLLECTION_LOCK_WRITE))
	{
		cp_hashlist_entry *entry;

		if ((iterator->list->mode & COLLECTION_MODE_MULTIPLE_VALUES) == 0
			&& cp_hashlist_contains_internal(iterator->list, key)) 
		{
			errno = EINVAL;
			return NULL;
		}

		entry = cp_hashlist_create_entry(iterator->list, 
									 	 iterator->list->mode,
									 	 key, 
									 	 value);
		if (entry == NULL) return NULL;
		cp_hashlist_insert_internal(iterator->list, entry);
		new_entry = entry;
		
		entry->prev = *iterator->pos;

		if (*iterator->pos)
		{
			entry->next = (*iterator->pos)->next;
			(*iterator->pos)->next = entry;
			if (entry->next)
				entry->next->prev = entry;
		}
		else if (iterator->list->tail == NULL) /* iterator not pointing at much - list may be empty */
			iterator->list->tail = iterator->list->head = entry;
		else if (iterator->pos == &iterator->list->tail->next) /* iterator moved after tail */
		{
			entry->next = NULL;
			entry->prev = iterator->list->tail;
			entry->prev->next = entry;
			iterator->list->tail = entry;
		}
		else /* iterator moved before head */
		{
			entry->next = iterator->list->head;
			entry->next->prev = entry;
			iterator->list->head = entry;
		}

		iterator->pos = &entry->prev; /* keep iterator on same entry */
		iterator->list->items++;
	}
	else /* mode is not NOSYNC and no LOCK_WRITE */
		errno = EINVAL;

	return new_entry;
}

void *cp_hashlist_iterator_remove(cp_hashlist_iterator *iterator)
{
	void *rm_value = NULL;

	if ((iterator->list->mode & COLLECTION_MODE_NOSYNC) || 
		(iterator->lock_type == COLLECTION_LOCK_WRITE))
	{
		if (*iterator->pos)
		{
			cp_hashlist_entry *curr = *iterator->pos;

			if (curr->next)
				iterator->pos = &curr->next->prev;
			else if (curr->prev)
				iterator->pos = &curr->prev->next;
			else /* removing last item */
				iterator->pos = &iterator->list->head;			

			curr = cp_hashlist_remove_internal(iterator->list, curr->key);
			rm_value = 
				cp_hashlist_unlink_internal(iterator->list, 
											curr, iterator->list->mode);
		}
	}

	return rm_value;
}

/* ----------------------------------------------------------------- */
/** @} */


/* ----------------------------------------------------------------- */
/* Internal methods */
/* ----------------------------------------------------------------- */

static cp_hashlist_entry *
	cp_hashlist_insert_internal(cp_hashlist *list, 
                                cp_hashlist_entry *entry)
{
    cp_hashlist_entry **insert;
    unsigned long index;

    if (!(list->mode & COLLECTION_MODE_NORESIZE) && 
		list->items * 100 > list->table_size * list->fill_factor_max)
        cp_hashlist_resize_nosync(list, list->table_size * 2);

    index = abs((*list->hash_fn)(entry->key)) % list->table_size;
    insert = &list->table[index];

    /* accept doubles */
    while (*insert != NULL) 
		insert = &(*insert)->bucket;

    *insert = entry;
    list->items++;

    return entry;
}


static void *cp_hashlist_get_internal(cp_hashlist *list, void *key)
{
    unsigned long index;
    cp_hashlist_entry *entry;

    index = abs((*list->hash_fn)(key)) % list->table_size;
    entry = list->table[index];
    while (entry && (*list->compare_fn)(entry->key, key)) 
		entry = entry->bucket;

#if CP_HASHLIST_MULTIPLE_VALUES
	if ((list->mode & COLLECTION_MODE_MULTIPLE_VALUES) && entry)
	{
		cp_list *res = 
			cp_list_create_view(list->mode, 
								NULL, 
								list->copy_value,
								list->free_value,
								list->lock);
		cp_list_insert(res, entry->value);

		if (list->mode & COLLECTION_MODE_LIST_ORDER) /* list order */
		{
			cp_hashlist_entry *i;
			for (i = entry->prev; i; i++)
				if ((*list->compare_fn)(i->key, key) == 0)
					cp_list_insert(res, i->value);

			for (i = entry->next; i; i++)
				if ((*list->compare_fn)(i->key, key) == 0)
					cp_list_append(res, i->value);
		}
		else /* insertion order */
		{
			entry = entry->bucket;
			while (entry)
				if ((*list->compare_fn)(entry->key, key) == 0)
					cp_list_append(res, entry->value);
		}

		return res;
	}
#endif
			
    return entry;
}


static cp_hashlist_entry *
	cp_hashlist_remove_entry_internal(cp_hashlist *list, 
                                      cp_hashlist_entry *entry)
{
    cp_hashlist_entry **remove;
    unsigned long index;

    index = abs((*list->hash_fn)(entry->key)) % list->table_size;
    remove = &list->table[index];
    while (*remove != NULL) 
	{
        if (*remove == entry) break;
        remove = &(*remove)->bucket;
    }

    if (*remove) 
	{
        *remove = (*remove)->bucket;
        list->items--;
    } 
	else /* should _not_ happen */
	{
        printf("may day, cannot find that entry: %s\n", (char *) entry->key);
        return NULL;
    }

    return entry;
}

static cp_hashlist_entry *
	cp_hashlist_remove_internal(cp_hashlist *list, void *key)
{
    cp_hashlist_entry **remove, *holder;
    unsigned long index;

    if (!(list->mode & COLLECTION_MODE_NORESIZE) && 
		list->items * 100 < list->table_size * list->fill_factor_min &&
		list->items > list->min_size)
        cp_hashlist_resize_nosync(list, list->table_size / 2);

    index = abs((*list->hash_fn)(key)) % list->table_size;
    remove = &list->table[index];
    while (*remove != NULL) 
	{
        if ((*list->compare_fn)((*remove)->key, key) == 0) break;
        remove = &(*remove)->bucket;
    }

    holder = NULL;
    if (*remove) 
	{
        holder = *remove;
        *remove = (*remove)->bucket;
        list->items--;
    }

    return holder;
}


static void cp_hashlist_entry_delete(cp_hashlist_entry *entry)
{
    if (entry) free(entry);
}

static void 
	cp_hashlist_entry_release_by_option(cp_hashlist *list, 
										cp_hashlist_entry *entry, 
										int mode)
{
    if (entry) 
	{
        if (mode & COLLECTION_MODE_DEEP) 
		{
            if (list->free_key) (*list->free_key)(entry->key);
            if (list->free_value) (*list->free_value)(entry->value);
        }
        free(entry);
    }
}

