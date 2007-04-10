/**
 * @addtogroup cp_list
 */
/** @{ */
/**
 * @file
 * Implementation of cp_list Collection with linked elements and
 * cp_list_iterator.
 * 
 * The elements are stored in cp_list_entry objects.
 *
 * @copydoc collection
 */
/* ----------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "linked_list.h"

/* fwd declarations of internal locking functions */
int cp_list_txlock(cp_list *list, int type);
/* fwd declarations of internal locking functions */
int cp_list_txunlock(cp_list *list);

/**
 * Creates an entry with the mode of the list.
 *
 * The new entry is not added to the list.
 * If the list has copy mode, it copies the item and puts the copy into the
 * new entry.
 *
 * @return entry object for the item.
 */
static cp_list_entry *cp_list_create_entry(cp_list *list, void *item);

//static void cp_list_destroy_entry(cp_list *list, cp_list_entry *entry);

/**
 * Unsynchronized and unchecked insert a new element at the beginning
 * of the list.
 */
static cp_list_entry *cp_list_insert_internal(cp_list *list, void *item);

/**
 * Unsynchronized and unchecked remove the entry from the list.
 */
static cp_list_entry *cp_list_remove_internal(cp_list *list, cp_list_entry *entry);

/**
 * Appends the element and returns the Entry.
 */
static cp_list_entry *cp_list_append_internal(cp_list *list, void *item);

/**
 * Removes the first entry from the list and returns it.
 *
 * Sets the references into a consistent state.
 */
static cp_list_entry *cp_list_remove_head_internal(cp_list *list);

/**
 * Removes the last entry from the list and returns it.
 *
 * Sets the references into a consistent state.
 */
static cp_list_entry *cp_list_remove_tail_internal(cp_list *list);


cp_list *cp_list_create_internal(int mode, 
								 cp_compare_fn compare_fn,
								 cp_copy_fn copy_fn,
								 cp_destructor_fn item_destructor,
								 int is_view)
{
    cp_list *list = NULL;

    list = (cp_list *) calloc(1, sizeof(cp_list));
    if (list == NULL) 
	{
        errno = ENOMEM;
        return NULL;
    }

    list->head       = NULL;
    list->tail       = NULL;
    list->compare_fn = compare_fn;
    list->copy_fn    = copy_fn;
	list->free_fn	 = item_destructor;
    list->mode       = mode;
    list->items      = 0;

	list->is_view    = is_view;

	if (!(mode & COLLECTION_MODE_NOSYNC) && !is_view)
	{
		list->lock = malloc(sizeof(cp_lock));
		if (list->lock == NULL)
		{
			free(list);
			errno = ENOMEM;
			return NULL;
		}
		if (cp_lock_init(list->lock, NULL))
		{
			free(list->lock);
			free(list);
			return NULL;
		}
	}

    return list;
}

cp_list *cp_list_create()
{
	return 
		cp_list_create_internal(COLLECTION_MODE_MULTIPLE_VALUES,
				NULL, NULL, NULL, 0);
}

cp_list *cp_list_create_nosync()
{
	return 
		cp_list_create_internal(COLLECTION_MODE_MULTIPLE_VALUES |
							    COLLECTION_MODE_NOSYNC, 
								NULL, NULL, NULL, 0);
}

cp_list *cp_list_create_list(int mode, 
							 cp_compare_fn compare_fn, 
							 cp_copy_fn copy_fn,
							 cp_destructor_fn item_destructor)
{
	return 
		cp_list_create_internal(mode, compare_fn, copy_fn, item_destructor, 0);
}


cp_list *cp_list_create_view(int mode, 
							 cp_compare_fn compare_fn, 
							 cp_copy_fn copy_fn,
							 cp_destructor_fn item_destructor,
							 cp_lock *lock)
{
	cp_list *list = 
		cp_list_create_internal(mode, compare_fn, copy_fn, item_destructor, 1);

	list->lock = lock; //~~ test

	return list;
}

/* 
 * locking provides some protection if the list is being destroyed while it is
 * still in use. However if the lock causes a different thread to block the 
 * other thread is likely to crash when releasing the lock which will possibly
 * have been deallocated in the meanwhile. It is the applications 
 * responsibility to assure the list isn't being accessed and destroyed in
 * different threads simultaneously.
 */
void cp_list_destroy_internal(cp_list *list, cp_destructor_fn fn, int mode)
{
    cp_list_entry *curr, *next;
	int shared_pool;

	cp_list_txlock(list, COLLECTION_LOCK_WRITE);
	
	shared_pool = list->mempool && list->mempool->refcount > 1;
		
    curr = list->head;
    while (curr) 
	{
        next = curr->next;
        if ((mode & COLLECTION_MODE_DEEP) && fn)
			(*fn)(curr->item);
		if (list->mempool) 
		{
			if (shared_pool)
				cp_mempool_free(list->mempool, curr);
		}
		else
			free(curr);
        curr = next;
    }
	cp_list_txunlock(list);
	
	if (list->lock && !list->is_view)
	{
		cp_lock_destroy(list->lock);
		free(list->lock);
	}

	if (list->mempool) cp_mempool_destroy(list->mempool);

    free(list);
}

void cp_list_destroy(cp_list *list)
{
    cp_list_destroy_internal(list, list->free_fn, list->mode);
}

void cp_list_destroy_by_option(cp_list *list, int option)
{
	cp_list_destroy_internal(list, list->free_fn, option);
}

void cp_list_destroy_custom(cp_list *list, cp_destructor_fn fn)
{
    cp_list_destroy_internal(list, fn, list->mode | COLLECTION_MODE_DEEP);
}

long cp_list_item_count(cp_list *list)
{
    return (list == NULL) ? 0 : list->items;
}


static cp_list_entry **cp_list_get_entry_ref(cp_list *list, void *item)
{
    cp_list_entry **here = &list->head;

    while (*here != NULL && (*list->compare_fn)(item, (*here)->item)) 
        here = &(*here)-> next;

    return here;
}

void *cp_list_insert(cp_list *list, void *item)
{
    cp_list_entry *entry, **lookup;
    void *res = NULL;

    if (cp_list_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;

    entry = NULL;
    if (!(list->mode & COLLECTION_MODE_MULTIPLE_VALUES)) 
	{
        lookup = cp_list_get_entry_ref(list, item);
        if (lookup) entry = *lookup;
    }
    if (entry == NULL) entry = cp_list_insert_internal(list, item);
    if (entry) res = entry->item;

    cp_list_txunlock(list);
    
    return res;
}

void *cp_list_remove(cp_list *list, void *item)
{
    void *res = NULL;
    cp_list_entry *here, *curr;
	int mvalbit = list->mode & COLLECTION_MODE_MULTIPLE_VALUES;
	int deepbit = list->mode & COLLECTION_MODE_DEEP;

    if (cp_list_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;

    here = list->head;
    while (here != NULL) 
	{
        curr = here;
        here = here->next;
        if ((*list->compare_fn)(item, curr->item) == 0) 
		{
            cp_list_remove_internal(list, curr);
            if (deepbit && list->free_fn) (*list->free_fn)(curr->item);
			if (list->mempool)
				cp_mempool_free(list->mempool, curr);
			else
	            free(curr);
    
            if (!mvalbit) break;
        }
    }

    cp_list_txunlock(list);

    return res;
}

void *cp_list_insert_after(cp_list *list, void *item, void *existing)
{
    cp_list_entry **here, *entry;

    if (cp_list_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;

    if (!(list->mode) & COLLECTION_MODE_MULTIPLE_VALUES) 
	{
        here = cp_list_get_entry_ref(list, item);
        if (*here != NULL) 
		{
            cp_list_txunlock(list);
            return (*here)->item;
        }
    }

    entry = cp_list_create_entry(list, item);
    if (entry == NULL) 
	{
        cp_list_txunlock(list);
        return NULL;
    }

    here = cp_list_get_entry_ref(list, existing);

    if (*here == NULL) /* no match - append to end of list */
		here = &list->tail;

	entry->prev = *here;
	entry->next = (*here)->next;
   	(*here)->next = entry;

	if (entry->next) 
		entry->next->prev = entry;
	else
		list->tail = entry;
    
    list->items++;
    
    cp_list_txunlock(list);

    return entry->item;
}

void *cp_list_insert_before(cp_list *list, void *item, void *existing)
{
    cp_list_entry **here, *entry;

    if (cp_list_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;

    if (!(list->mode) & COLLECTION_MODE_MULTIPLE_VALUES) 
	{
        here = cp_list_get_entry_ref(list, item);
        if (*here != NULL) 
		{
            cp_list_txunlock(list);
            return (*here)->item;
        }
    }

    entry = cp_list_create_entry(list, item);
    if (entry == NULL) 
	{
        cp_list_txunlock(list);
        return NULL;
    }

    here = cp_list_get_entry_ref(list, existing);

    if (*here == NULL) /* no match - insert at top of list */
		here = &list->head;
		
	entry->next = *here;
	entry->prev = (*here)->prev;
	(*here)->prev = entry;
	if (entry->prev)
		entry->prev->next = entry;
	else
		list->head = entry;

    list->items++;
        
    cp_list_txunlock(list);

    return entry->item;
}

void *cp_list_search(cp_list *list, void *item)
{
    cp_list_entry **here;
    void *res;

    if (cp_list_txlock(list, COLLECTION_LOCK_READ)) return NULL;

    here = cp_list_get_entry_ref(list, item);
    res = *here ? (*here)->item : NULL;

    cp_list_txunlock(list);
    
    return res;
}

int cp_list_callback(cp_list *l, int (*item_action)(void *, void *), void *id)
{
	int rc = 0;
	cp_list_entry *curr;

	if ((rc = cp_list_txlock(l, COLLECTION_LOCK_READ))) return rc;

	for (curr = l->head; curr; curr = curr->next)
		if ((rc = (*item_action)(curr->item, id)))
			break;

	cp_list_txunlock(l);

	return rc;
}


void *cp_list_append(cp_list *list, void *item)
{
    cp_list_entry **lookup, *entry;
    void *res = NULL;

    if (cp_list_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;

    if (!(list->mode & COLLECTION_MODE_MULTIPLE_VALUES)) 
	{
        lookup = cp_list_get_entry_ref(list, item);
        if (lookup != NULL) 
		{
            cp_list_txunlock(list);
            return (*lookup)->item;
        }
    }

    entry = cp_list_append_internal(list, item);
    if (entry) res = entry->item;

    cp_list_txunlock(list);

    return res;
}

void *cp_list_get_head(cp_list *list)
{
    void *item = NULL;
        
    if (cp_list_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;
    if (list->head) item = list->head->item;
    cp_list_txunlock(list);

    return item;
}

void *cp_list_get_tail(cp_list *list)
{
    void *item = NULL;
        
    if (cp_list_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;
    if (list->tail) item = list->tail->item;
    cp_list_txunlock(list);

    return item;
}


void *cp_list_remove_head(cp_list *list)
{
    cp_list_entry *old_head;
    void *res = NULL;
    
	if (cp_list_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;
	   
    old_head = cp_list_remove_head_internal(list);
    if (old_head) 
	{
        res = old_head->item;
		if ((list->mode & COLLECTION_MODE_DEEP) && list->free_fn)
			(*list->free_fn)(old_head->item);
		if (list->mempool)
			cp_mempool_free(list->mempool, old_head);
		else
			free(old_head);
    }

    cp_list_txunlock(list);

    return res;
}

void *cp_list_remove_tail(cp_list *list)
{
    cp_list_entry *old_tail;
    void *res = NULL;

    if (cp_list_txlock(list, COLLECTION_LOCK_WRITE)) return NULL;

    old_tail = cp_list_remove_tail_internal(list);
    if (old_tail) 
	{
        res = old_tail->item;
		if (list->mempool)
			cp_mempool_free(list->mempool, old_tail);
		else
	        free(old_tail);
    }

    cp_list_txunlock(list);

    return res;
}


int cp_list_is_empty(cp_list *list)
{
    int empty = 0;

    cp_list_txlock(list, COLLECTION_LOCK_READ);
    empty = list->head == NULL;
   cp_list_txunlock(list);

    return empty;
}

int cp_list_lock_internal(cp_list *list, int mode)
{
    int rc;

    if (mode == COLLECTION_LOCK_READ) 
		rc = cp_lock_rdlock(list->lock);
    else
		rc = cp_lock_wrlock(list->lock);

    return rc;
}

int cp_list_unlock_internal(cp_list *list)
{
    return cp_lock_unlock(list->lock);
}

int cp_list_txlock(cp_list *list, int type)
{
	if (list->mode & COLLECTION_MODE_NOSYNC) return 0;
	if (list->mode & COLLECTION_MODE_IN_TRANSACTION && 
		list->txtype == COLLECTION_LOCK_WRITE)
	{
		cp_thread self = cp_thread_self();
		if (cp_thread_equal(self, list->txowner)) return 0;
	}
	return cp_list_lock_internal(list, type);
}

int cp_list_txunlock(cp_list *list)
{
	if (list->mode & COLLECTION_MODE_NOSYNC) return 0;
	if (list->mode & COLLECTION_MODE_IN_TRANSACTION && 
		list->txtype == COLLECTION_LOCK_WRITE)
	{
		cp_thread self = cp_thread_self();
		if (cp_thread_equal(self, list->txowner)) return 0;
	}
	return cp_list_unlock_internal(list);
}

/* lock and set the transaction indicators */
int cp_list_lock(cp_list *list, int type)
{
	int rc;
	if ((list->mode & COLLECTION_MODE_NOSYNC)) return EINVAL;
	if ((rc = cp_list_lock_internal(list, type))) return rc;
	list->txtype = type;
	list->txowner = cp_thread_self();
	list->mode |= COLLECTION_MODE_IN_TRANSACTION;
	return 0;
}

/* unset the transaction indicators and unlock */
int cp_list_unlock(cp_list *list)
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

	return cp_list_unlock_internal(list);
}

/* get the current collection mode */
int cp_list_get_mode(cp_list *list)
{
    return list->mode;
}

/* set mode bits on the list mode indicator */
int cp_list_set_mode(cp_list *list, int mode)
{
	int nosync;

	/* can't set NOSYNC in the middle of a transaction */
	if ((list->mode & COLLECTION_MODE_IN_TRANSACTION) && 
		(mode & COLLECTION_MODE_NOSYNC)) return EINVAL;
	
	nosync = list->mode & COLLECTION_MODE_NOSYNC;
	if (!nosync)
		if (cp_list_txlock(list, COLLECTION_LOCK_WRITE))
			return -1;

	list->mode |= mode;

	if (!nosync)
		cp_list_txunlock(list);

	return 0;
}

/* unset mode bits on the list mode indicator. if unsetting 
 * COLLECTION_MODE_NOSYNC and the list was not previously synchronized, the 
 * internal synchronization structure is initalized.
 */
int cp_list_unset_mode(cp_list *list, int mode)
{
	int nosync = list->mode & COLLECTION_MODE_NOSYNC;

	if (!nosync)
		if (cp_list_txlock(list, COLLECTION_LOCK_WRITE))
			return -1;
	
	/* handle the special case of unsetting COLLECTION_MODE_NOSYNC */
	if ((mode & COLLECTION_MODE_NOSYNC) && list->lock == NULL)
	{
		/* list can't be locked in this case, no need to unlock on failure */
		if ((list->lock = malloc(sizeof(cp_lock))) == NULL)
			return -1; 
		if (cp_lock_init(list->lock, NULL))
			return -1;
	}
	
	/* unset specified bits */
    list->mode &= list->mode ^ mode;
	if (!nosync)
		cp_list_txunlock(list);

	return 0;
}

/* set list to use given mempool or allocate a new one if pool is NULL */
int cp_list_use_mempool(cp_list *list, cp_mempool *pool)
{
	int rc = 0;
	
	if ((rc = cp_list_txlock(list, COLLECTION_LOCK_WRITE))) return rc;
	
	if (pool)
	{
		if (pool->item_size < sizeof(cp_list_entry))
		{
			rc = EINVAL;
			goto DONE;
		}
		if (list->mempool) 
		{
			if (list->items) 
			{
				rc = ENOTEMPTY;
				goto DONE;
			}
			cp_mempool_destroy(list->mempool);
		}
		cp_mempool_inc_refcount(pool);
		list->mempool = pool;
	}
	else
	{
		list->mempool = 
			cp_mempool_create_by_option(COLLECTION_MODE_NOSYNC, 
										sizeof(cp_list_entry), 0);
		if (list->mempool == NULL) 
		{
			rc = ENOMEM;
			goto DONE;
		}
	}

DONE:
	cp_list_txunlock(list);
	return rc;
}


/* set list to use a shared memory pool */
int cp_list_share_mempool(cp_list *list, cp_shared_mempool *pool)
{
	int rc;

	if ((rc = cp_list_txlock(list, COLLECTION_LOCK_WRITE))) return rc;

	if (list->mempool)
	{
		if (list->items)
		{
			rc = ENOTEMPTY;
			goto DONE;
		}

		cp_mempool_destroy(list->mempool);
	}

	list->mempool = cp_shared_mempool_register(pool, sizeof(cp_list_entry));
	if (list->mempool == NULL) 
	{
		rc = ENOMEM;
		goto DONE;
	}
	
DONE:
	cp_list_txunlock(list);
	return rc;
}


/****************************************************************************
 *                                                                          *
 *                    cp_list_iterator implementation                       *
 *                                                                          *
 ****************************************************************************/
 
cp_list_iterator* cp_list_create_iterator(cp_list *list, int type)
{
    int rc = - 1;
    cp_list_iterator *iterator = (cp_list_iterator *) malloc(sizeof(cp_list_iterator));
    iterator->list = list;
    iterator->pos = &list->head;
    iterator->lock_type = type;

	rc = cp_list_txlock(list, type);
	if (rc) /* locking failed */
	{
		free(iterator);
		iterator = NULL;
	}

    return iterator;
}

int cp_list_iterator_init(cp_list_iterator *iterator, cp_list *list, int type)
{
    iterator->list = list;
    iterator->pos = &list->head;
    iterator->lock_type = type;
	return cp_list_txlock(list, type);
}


int cp_list_iterator_head(cp_list_iterator *iterator)
{
    if (iterator == NULL) return -1;
    iterator->pos = &iterator->list->head;

    return 0;
}

int cp_list_iterator_tail(cp_list_iterator *iterator)
{
    if (iterator == NULL) return -1;
    iterator->pos = &iterator->list->tail;

    return 0;
}

int cp_list_iterator_init_tail(cp_list_iterator *iterator, 
							   cp_list *list, 
							   int type)
{
    iterator->list = list;
    iterator->pos = &list->tail;
    iterator->lock_type = type;
	return cp_list_txlock(list, type);
}

int cp_list_iterator_release(cp_list_iterator *iterator)
{
	int rc = 0;
    if (iterator->lock_type != COLLECTION_LOCK_NONE) 
		rc = cp_list_txunlock(iterator->list);

    return rc;
}

int cp_list_iterator_destroy(cp_list_iterator *iterator)
{
    int rc = cp_list_iterator_release(iterator);
    free(iterator);

    return rc;
}

void *cp_list_iterator_next(cp_list_iterator *iterator)
{
    void *item = NULL;

    if (*(iterator->pos)) 
	{
        item = (*iterator->pos)->item;
        iterator->pos = &(*(iterator->pos))->next;
    }
	else if (iterator->list->head && 
			 iterator->pos == &iterator->list->head->prev)
	{
		item = iterator->list->head;
		iterator->pos = &iterator->list->head;
	}

    return item;
}

void *cp_list_iterator_prev(cp_list_iterator *iterator)
{
    void *item = NULL;

    if (*iterator->pos) 
	{
        item = (*iterator->pos)->item;
        iterator->pos = &(*iterator->pos)->prev;
    }
	else if (iterator->list->tail && 
			 iterator->pos == &iterator->list->tail->next)
	{
		item = iterator->list->tail->item;
		iterator->pos = &iterator->list->tail->prev;
	}

    return item;
}

void *cp_list_iterator_curr(cp_list_iterator *iterator)
{
    void *item = NULL;

    if (*iterator->pos)
        item = (*iterator->pos)->item;

    return item;
}

void *cp_list_iterator_insert(cp_list_iterator *iterator, void *item)
{
	void *new_item = NULL;

	if ((iterator->list->mode & COLLECTION_MODE_NOSYNC) || 
		(iterator->lock_type == COLLECTION_LOCK_WRITE))
	{
		cp_list_entry *entry = cp_list_create_entry(iterator->list, item);
		if (entry == NULL) return NULL;
		new_item = entry->item;
		
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

		iterator->pos =  &entry->next; /* keep iterator at same position */
		iterator->list->items++;
	}
	else /* mode is not NOSYNC and no LOCK_WRITE */
		errno = EINVAL;

	return new_item;
}

void *cp_list_iterator_append(cp_list_iterator *iterator, void *item)
{
	void *new_item = NULL;

	if ((iterator->list->mode & COLLECTION_MODE_NOSYNC) || 
		(iterator->lock_type == COLLECTION_LOCK_WRITE))
	{
		cp_list_entry *entry = cp_list_create_entry(iterator->list, item);
		if (entry == NULL) return NULL;
		new_item = entry->item;
		
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

		iterator->pos = &entry->prev; /* keep iterator at same position */
		iterator->list->items++;
	}
	else /* mode is not NOSYNC and no LOCK_WRITE */
		errno = EINVAL;

	return new_item;
}

void *cp_list_iterator_remove(cp_list_iterator *iterator)
{
	void *rm_item = NULL;

	if ((iterator->list->mode & COLLECTION_MODE_NOSYNC) || 
		(iterator->lock_type == COLLECTION_LOCK_WRITE))
	{
		if (*iterator->pos)
		{
			cp_list_entry *curr = *iterator->pos;
			if (curr->prev)
				iterator->pos = &curr->prev->next;
			else if (curr->prev)
				iterator->pos = &curr->next->prev;
			else /* removing last item */
				iterator->pos = &iterator->list->head;

			cp_list_remove_internal(iterator->list, curr);
            if (iterator->list->mode & COLLECTION_MODE_DEEP && 
				iterator->list->free_fn) (*iterator->list->free_fn)(curr->item);
			if (iterator->list->mempool)
				cp_mempool_free(iterator->list->mempool, curr);
			else
	            free(curr);
		}
	}

	return rm_item;
}




/* ----------------------------------------------------------------- */
/** @} */

static cp_list_entry *cp_list_create_entry(cp_list *list, void *item)
{
    cp_list_entry *entry;
	
	if (list->mempool)
		entry = (cp_list_entry *) cp_mempool_calloc(list->mempool);
	else
		entry = (cp_list_entry *) calloc(1, sizeof(cp_list_entry));

    if (entry == NULL) 
	{
        errno = ENOMEM;
        return NULL;
    }
    entry->item = (list->mode & COLLECTION_MODE_COPY) ? (*list->copy_fn)(item) : item;

    return entry;
}

static cp_list_entry *cp_list_insert_internal(cp_list *list, void *item)
{
    cp_list_entry *entry;

    entry = cp_list_create_entry(list, item);
    if (entry == NULL) return NULL;

    entry->next = list->head;
    list->head = entry;
    if (entry->next) entry->next->prev = entry;
    if (list->tail == NULL) list->tail = entry;

    list->items++;

    return entry;
}

static cp_list_entry *
	cp_list_remove_internal(cp_list *list, cp_list_entry *entry)
{
    if (entry->prev) 
		entry->prev->next = entry->next;
    else
		list->head = entry->next;

    if (entry->next)
		entry->next->prev = entry->prev;
    else
		list->tail = entry->prev;

    list->items--;

    return entry;
}

static cp_list_entry *cp_list_append_internal(cp_list *list, void *item)
{
    cp_list_entry *entry;

    entry = cp_list_create_entry(list, item);
    if (entry == NULL) return NULL;

    entry->prev = list->tail;
    list->tail = entry;

    if (entry->prev) entry->prev->next = entry;

    if (list->head == NULL) list->head = entry;

    list->items++;

    return entry;
}

static cp_list_entry *cp_list_remove_head_internal(cp_list *list)
{
    cp_list_entry *old_head;

    old_head = list->head;
    if (old_head) 
	{
        list->head = list->head->next;

        if (list->head == NULL) 
			list->tail = NULL;
        else
			list->head->prev = NULL;

        list->items--;
    }

    return old_head;
}

static cp_list_entry *cp_list_remove_tail_internal(cp_list *list)
{
    cp_list_entry *old_tail;

    old_tail = list->tail;
    if (old_tail) 
	{
        list->tail = list->tail->prev;

        if (list->tail == NULL) 
			list->head = NULL;
        else
			list->tail->next = NULL;

        list->items--;
    }

    return old_tail;
}

