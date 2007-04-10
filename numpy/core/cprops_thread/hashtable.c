
/**
 * @addtogroup cp_hashtable
 */
/** @{ */

/**
 * @file
 * Implementation of generic synchronized cp_hashtable. 
 *
 * The elements are stored in cp_list_entry objects.
 *
 * @copydoc collection
 */
/* ----------------------------------------------------------------- */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "hashtable.h"
#include "linked_list.h"
#include "log.h"
#include "common.h"
#include "collection.h"
#include "thread.h"
/* #include "util.h" */

#include "cp_config.h"
#ifdef CP_HAS_STRINGS_H
#include <strings.h>
#endif

#ifdef __OpenBSD__
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#endif /* __OpenBSD__ */

#ifndef CP_HASHTABLE_MULTIPLE_VALUES
#define CP_HASHTABLE_MULTIPLE_VALUES 1
#endif

static int table_sizes[] = 
    {
           5,        7,       11,       23,       47,
          79,       97,      157,      197,      299,
         397,      599,      797,     1297,     1597,
        2499,     3199,     4799,     6397,     9599,
       12799,    19199,    25597,    38399,    51199,
	   76799,   102397,   153599,   204797,   306797,
	  409597,   614399,   819199,  1288799,  1638397, 
     2457599,  3276799,  4915217,  6553577,  9830393 
    };

static int table_sizes_len = 40;


unsigned long cp_hashtable_choose_size(unsigned long size_request)
{
    unsigned long new_size;

    if (table_sizes[table_sizes_len - 1] < size_request)
    {
        for (new_size = table_sizes[table_sizes_len - 1]; 
             new_size < size_request; 
             new_size = new_size * 2 + 1);
    }
    else
    {
        int min = -1;
        int max = table_sizes_len - 1;
        int pos;

        while (max > min + 1)
        {
            pos = (max + min + 1) / 2;
            if (table_sizes[pos] < size_request)
                min = pos;
            else
                max = pos;
        }

        new_size = table_sizes[max];
    }
         
    return new_size;
}

cp_hashtable *cp_hashtable_create(unsigned long size_hint, 
                                  cp_hashfunction hash_fn, 
                                  cp_compare_fn compare_fn)
{
    return cp_hashtable_create_by_option(0, size_hint, hash_fn, 
                                         compare_fn, NULL, NULL, NULL, NULL);
}


cp_hashtable *
    cp_hashtable_create_copy_mode(unsigned long size_hint, 
                                  cp_hashfunction hash_fn, 
                                  cp_compare_fn compare_fn, 
                                  cp_copy_fn copy_key, 
                                  cp_destructor_fn free_key,
                                  cp_copy_fn copy_value,
                                  cp_destructor_fn free_value)
{
    return cp_hashtable_create_by_option(COLLECTION_MODE_DEEP | 
                                         COLLECTION_MODE_COPY, 
                                         size_hint, 
                                         hash_fn, 
                                         compare_fn, 
                                         copy_key, 
                                         free_key,
                                         copy_value,
                                         free_value);
}

cp_hashtable *
    cp_hashtable_create_by_option(int mode, unsigned long size_hint, 
                                  cp_hashfunction hash_fn, 
                                  cp_compare_fn compare_fn, 
                                  cp_copy_fn copy_key, 
                                  cp_destructor_fn free_key,
                                  cp_copy_fn copy_value,
                                  cp_destructor_fn free_value)
{
    cp_hashtable *table;
    
    table = (cp_hashtable *) calloc(1, sizeof(cp_hashtable));
    if (table == NULL)
    {
        errno = ENOMEM;
        return NULL;
    }

    table->table_size = cp_hashtable_choose_size(size_hint);
    table->items = 0;
    table->table = (cp_hashtable_entry **) 
		calloc(table->table_size, sizeof(cp_hashtable_entry *));
    if (table->table == NULL) 
    {
        errno = ENOMEM;
        return NULL;
    }

    table->hash_fn = hash_fn;
    table->compare_fn = compare_fn;
    table->copy_key = copy_key;
    table->copy_value = copy_value;
    table->free_key = free_key;
    table->free_value = free_value;

    table->mode = mode;

    table->min_size = size_hint;
    table->fill_factor_min = CP_HASHTABLE_DEFAULT_MIN_FILL_FACTOR;
    table->fill_factor_max = CP_HASHTABLE_DEFAULT_MAX_FILL_FACTOR;

    table->resizing = 0;

    table->lock = malloc(sizeof(cp_lock));
    if (table->lock == NULL)
    {
        free(table->table);
        free(table);
        errno = ENOMEM;
        return NULL;
    }
  
	if (cp_lock_init(table->lock, NULL))
	{
		free(table->lock);
        free(table->table);
        free(table);
        return NULL;
	}

    return table;
}

cp_hashtable_entry *
	cp_hashtable_create_entry(cp_hashtable *table, 
							  int mode, 
							  void *key, 
							  void *value, 
							  long hashcode)
{
    cp_hashtable_entry *entry;

    entry = (cp_hashtable_entry *) malloc(sizeof(cp_hashtable_entry));
    if (entry == NULL) 
    {
        errno = ENOMEM;
        return NULL;
    }

    if (mode & COLLECTION_MODE_COPY) 
    {
        entry->key = table->copy_key ? (*table->copy_key)(key) : key;
        entry->value = table->copy_value ? (*table->copy_value)(value) :value;
    } 
    else 
    {
        entry->key = key;
        entry->value = value;    
    }

    entry->hashcode = hashcode;
    entry->next = NULL;

    return entry;
}

    
void cp_hashtable_destroy(cp_hashtable *table)
{
    long i;
    cp_hashtable_entry *entry, *next;
    
	table->mode |= COLLECTION_MODE_NORESIZE;

    if (table->resizing) 
    {
		struct timeval timeout;
		timeout.tv_sec = 0;
		timeout.tv_usec = 10;
		while (table->resizing)
			select(0, NULL, NULL, NULL, &timeout);
    }
        
    for (i = 0; i < table->table_size; i++) 
    {
        entry = table->table[i];
        while (entry != NULL) 
        {
            next = entry->next;
            if (table->mode & COLLECTION_MODE_DEEP) 
            {
                if (table->free_key)
                    (*table->free_key)(entry->key);
                if (table->free_value)
                    (*table->free_value)(entry->value);
            }
            free(entry);
            entry = next;
        }
    }
        
    free(table->table);
    cp_lock_destroy(table->lock);
    free(table->lock);
    free(table);
}

void cp_hashtable_destroy_deep(cp_hashtable *table)
{
    cp_hashtable_set_mode(table, COLLECTION_MODE_DEEP);
    cp_hashtable_destroy_custom(table, table->free_key, table->free_value);
}

void cp_hashtable_destroy_custom(cp_hashtable *table, cp_destructor_fn dk, cp_destructor_fn dv)
{
    long i;
    cp_hashtable_entry *entry, *next;

    if (table->resizing) 
    {
		struct timeval timeout;
		timeout.tv_sec = 0;
		timeout.tv_usec = 10;
		while (table->resizing)
			select(0, NULL, NULL, NULL, &timeout);
    }

    for (i = 0; i < table->table_size; i++) 
    {
        entry = table->table[i];
        while (entry != NULL) 
        {
            next = entry->next;
            if (dk) (*dk)(entry->key);
            if (dv) (*dv)(entry->value);
            free(entry);
            entry = next;
        }
    }
        
    free(table->table);
    cp_lock_destroy(table->lock);
    free(table->lock);
    free(table);
}


void cp_hashtable_destroy_shallow(cp_hashtable *table)
{
    cp_hashtable_unset_mode(table, COLLECTION_MODE_DEEP);
    cp_hashtable_destroy(table);
}

int cp_hashtable_lock_internal(cp_hashtable *table, int type)
{
    int rc;

    switch (type)
    {
        case COLLECTION_LOCK_READ:
            rc = cp_lock_rdlock(table->lock);
            break;

        case COLLECTION_LOCK_WRITE:
            rc = cp_lock_wrlock(table->lock);
            break;

        case COLLECTION_LOCK_NONE:
            rc = 0;
            break;

        default:
            rc = EINVAL;
            break;
    }

    return rc;
}
    
int cp_hashtable_unlock_internal(cp_hashtable *table)
{
    return cp_lock_unlock(table->lock);
}

int cp_hashtable_txlock(cp_hashtable *table, int type)
{
	if (table->mode & COLLECTION_MODE_NOSYNC) return 0;
	if (table->mode & COLLECTION_MODE_IN_TRANSACTION && 
		table->txtype == COLLECTION_LOCK_WRITE)
	{
		cp_thread self = cp_thread_self();
		if (cp_thread_equal(self, table->txowner)) return 0;
	}
	return cp_hashtable_lock_internal(table, type);
}

int cp_hashtable_txunlock(cp_hashtable *table)
{
	if (table->mode & COLLECTION_MODE_NOSYNC) return 0;
	if (table->mode & COLLECTION_MODE_IN_TRANSACTION && 
		table->txtype == COLLECTION_LOCK_WRITE)
	{
		cp_thread self = cp_thread_self();
		if (cp_thread_equal(self, table->txowner)) return 0;
	}
	return cp_hashtable_unlock_internal(table);
}

/* lock and set the transaction indicators */
int cp_hashtable_lock(cp_hashtable *table, int type)
{
	int rc;
	if ((table->mode & COLLECTION_MODE_NOSYNC)) return EINVAL;
	if ((rc = cp_hashtable_lock_internal(table, type))) return rc;
	table->txtype = type;
	table->txowner = cp_thread_self();
	table->mode |= COLLECTION_MODE_IN_TRANSACTION;
	return 0;
}

/* unset the transaction indicators and unlock */
int cp_hashtable_unlock(cp_hashtable *table)
{
	cp_thread self = cp_thread_self();
	if (table->txowner == self)
	{
		table->txtype = 0;
		table->txowner = 0;
		table->mode ^= COLLECTION_MODE_IN_TRANSACTION;
	}
	else if (table->txtype == COLLECTION_LOCK_WRITE)
		return -1;
	return cp_hashtable_unlock_internal(table);
}



int cp_hashtable_set_mode(cp_hashtable *table, int mode)
{
    int syncbit; 
	/* can't set NOSYNC in the middle of a transaction */
	if ((table->mode & COLLECTION_MODE_IN_TRANSACTION) && 
		(mode & COLLECTION_MODE_NOSYNC)) return EINVAL;

	syncbit = table->mode & COLLECTION_MODE_NOSYNC;
    if (!syncbit)
        cp_hashtable_txlock(table, COLLECTION_LOCK_WRITE);
    table->mode |= mode;
    if (!syncbit)
        cp_hashtable_txunlock(table);

	return 0;
}

int cp_hashtable_get_mode(cp_hashtable *table)
{
	return table->mode;
}

int cp_hashtable_unset_mode(cp_hashtable *table, int mode)
{
    int syncbit = table->mode & COLLECTION_MODE_NOSYNC;
    if (!syncbit)
        cp_hashtable_txlock(table, COLLECTION_LOCK_WRITE);
    table->mode &= table->mode ^ mode;
    if (!syncbit)
        cp_hashtable_txunlock(table);
	return 0;
}

/**
 * Retrieves the value by key from normal or resizing table.
 *
 * @param table the object
 * @param key    Key to search for.
 * @param code   Hash code of the Key (saves recalculating it)
 * @param option operation mode
 * @param resize 0: search in the normal table, 1: search in the resize table
 * @retval value of the entry with key.
 * @retval NULL otherwise (no entry with given key)
 */
void *lookup_internal(cp_hashtable *table, void *key, long code, int option, int resize)
{
    cp_hashtable_entry *entry;
    void *ret = NULL;

    entry = resize ? table->resize_table[code % table->resize_len]
                   : table->table[code % table->table_size];

    while (entry != NULL && (*table->compare_fn)(key, entry->key)) 
        entry = entry->next;
    
    if (entry)
    {
#if CP_HASHTABLE_MULTIPLE_VALUES
        if (option & COLLECTION_MODE_MULTIPLE_VALUES)
        {
            cp_list *l = 
                cp_list_create_view(table->mode, 
                                    NULL, 
                                    table->copy_value,
                                    table->free_value,
                                    table->lock);
            cp_list_insert(l, entry->value);
            entry = entry->next;
            while (entry != NULL) 
            {
                if ((*table->compare_fn)(key, entry->key) == 0) 
                    cp_list_append(l, entry->value);

                entry = entry->next;
            }

            ret = l;
        }
        else 
#endif
        ret = entry->value;
    }

    return ret;
}

/**
 * Retrieves the value by key.
 *
 * @param table the object
 * @param key Key to search for.
 * @param option operation mode
 * @retval value of the entry with key.
 * @retval NULL otherwise (no entry with given key or key == NULL)
 */
void *cp_hashtable_get_by_option(cp_hashtable *table, void *key, int option)
{
    long code;
    void *ret;

    if (table == NULL || key == NULL)
    {
        errno = EINVAL;
        return NULL;
    }

    if (cp_hashtable_txlock(table, COLLECTION_LOCK_READ)) return NULL;

    code = abs((*table->hash_fn)(key));
    ret = lookup_internal(table, key, code, option, 0);
    /* when resizing, search also there */
    if (ret == NULL && table->resizing) 
        ret = lookup_internal(table, key, code, option, 1);

    cp_hashtable_txunlock(table);

    return ret;
}

/**
 * \<\<Thread\>\> resize a cp_hashtable.
 *
 * This cp_thread does a background resize of the table.
 * It creates a new table and moves the entries to the new table.
 * @post All items moved to the new table which replaces the old table
 *       The old table is destroyed.
 * @note The cp_thread locks and unlocks the table for each item.
 * This creates some overhead, but ensures that the table can still be used
 * during resizing.
 */
void *cp_hashtable_resize_thread(void *tbl)
{
    long i, old_size, index;
    long new_size;
    cp_hashtable_entry **new_table, **old_table;
    cp_hashtable_entry **insert, *entry;
    cp_hashtable *table = (cp_hashtable *) tbl;

    new_size = table->table_size;
    new_table = table->table;

    old_size = table->resize_len;
    old_table = table->resize_table;

#ifdef __TRACE__
    DEBUGMSG("resize %d to %d - resize thread starting\n", old_size, new_size);
#endif

    /* copy old table into new table, bucket by bucket */
    for (i = 0; i < old_size; i++) 
    {
//        if (!(table->mode & COLLECTION_MODE_NOSYNC))
        cp_hashtable_txlock(table, COLLECTION_LOCK_WRITE);
        entry = old_table[i];
        while (entry != NULL) 
        {
            index = entry->hashcode % new_size;
            insert = &new_table[index];
            while (*insert != NULL)
                insert = &(*insert)->next;

            *insert = entry;
            entry = entry->next;
            (*insert)->next = NULL;
        }
        old_table[i] = NULL;

//        if (!(table->mode & COLLECTION_MODE_NOSYNC))
           cp_hashtable_txunlock(table);
    }

    /* cleanup */
//    if (!(table->mode & COLLECTION_MODE_NOSYNC)) 
    cp_hashtable_txlock(table, COLLECTION_LOCK_WRITE);
    free(table->resize_table); table->resize_table = NULL;
    table->resizing = 0;
//    if (!(table->mode & COLLECTION_MODE_NOSYNC)) 

#ifdef __TRACE__
    DEBUGMSG("resize %d to %d - done\n", old_size, new_size);
#endif
    cp_hashtable_txunlock(table);

    return NULL;
}

int cp_hashtable_set_min_size(cp_hashtable *table, int min_size)
{
    if (cp_hashtable_txlock(table, COLLECTION_LOCK_WRITE)) return -1;
    table->min_size = min_size;
    cp_hashtable_txunlock(table);
	return 0;
}


int cp_hashtable_set_max_fill_factor(cp_hashtable *table, int fill_factor)
{
    if (cp_hashtable_txlock(table, COLLECTION_LOCK_WRITE)) return -1;
    table->fill_factor_max = fill_factor;
    cp_hashtable_txunlock(table);
	return 0;
}

int cp_hashtable_set_min_fill_factor(cp_hashtable *table, int fill_factor)
{
    if (cp_hashtable_txlock(table, COLLECTION_LOCK_WRITE)) return -1;
    table->fill_factor_min = fill_factor;
    cp_hashtable_txunlock(table);
	return 0;
}


/**
 * Initiates a resize of the cp_hashtable which is executed in the background.
 *
 * @return NULL
 */
void *cp_hashtable_resize(cp_hashtable *table, long new_size)
{
    cp_hashtable_entry **new_table;

    if (table->resizing) return NULL;

    new_table = (cp_hashtable_entry **) malloc(new_size * sizeof(cp_hashtable_entry *));
    if (new_table == NULL) 
    {
        errno = ENOMEM;
        return NULL;
    }
    memset(new_table, 0, new_size * sizeof(cp_hashtable_entry *));

    table->resize_table = table->table;
    table->resize_len = table->table_size;
    table->table = new_table;
    table->table_size = new_size;
    table->resizing = 1;

    cp_thread_create(table->resize_thread, NULL, cp_hashtable_resize_thread, table);
    cp_thread_detach(table->resize_thread);

    return NULL;
}

/**
 * Resizes the table to a new size.
 * 
 * This is invoked by the insertion code if the load * factor goes over the
 * fill factor limits.
 * @param table the object
 * @param new_size desired size.
 *        The system trys to optmize the actual size for good distribution.
 */
void *cp_hashtable_resize_nosync(cp_hashtable *table, unsigned long new_size)
{
    unsigned long old_size;
    cp_hashtable_entry **old_table;
    cp_hashtable_entry *entry, *next, **insert;
    unsigned long i, index;

    old_size = table->table_size;
    old_table = table->table;
    
    table->table_size = cp_hashtable_choose_size(new_size);

#ifdef __TRACE__
    DEBUGMSG("resizing table (nosync): %d to %d\n", old_size, table->table_size);
#endif

    table->table = (cp_hashtable_entry **) malloc(table->table_size * sizeof(cp_hashtable_entry *));
    memset(table->table, 0, table->table_size * sizeof(cp_hashtable_entry *));

    if (table->table == NULL) 
    {
        errno = ENOMEM;
        return NULL;
    }

    for (i = 0; i < old_size; i++) 
    {
        entry = old_table[i];
        while (entry != NULL) 
        {
            index = entry->hashcode % table->table_size;
            next = entry->next;
            entry->next = NULL;
            insert = &table->table[index];
            while (*insert != NULL) 
                insert = &(*insert)->next;

            *insert = entry;
            
            entry = next;
        }
    }

    free(old_table);

    return table;
}
 
/**
 * Internal replace an existing entry with a new key, value pair.
 *
 * @param table the object
 * @param key    Key to search for.
 * @param value  new value
 * @param code   Hash code of the Key (saves recalculating it)
 * @param option operation mode
 * @param resize 0: search in the normal table, 1: search in the resize table
 * @return pointer to table of entries 
 */
static cp_hashtable_entry **cp_hashtable_replace_internal(cp_hashtable *table, 
                                                          void *key, 
                                                          void *value, 
                                                          unsigned long code, 
                                                          int option,
                                                          int resize)
{
    cp_hashtable_entry **entry;
    unsigned long index;

    index = resize ? code % table->resize_len : code % table->table_size;
    
    entry = resize ? &table->resize_table[index] : &table->table[index];
    while (*entry != NULL) 
    {
#if CP_HASHTABLE_MULTIPLE_VALUES
        if (!(option & COLLECTION_MODE_MULTIPLE_VALUES) && 
            (*table->compare_fn)(key, (*entry)->key) == 0) 
#else
        if ((*table->compare_fn)(key, (*entry)->key) == 0) 
#endif
        {
            if (option & COLLECTION_MODE_DEEP) 
            {
                if (table->free_key)
                    (*table->free_key)((*entry)->key);
                if (table->free_value)
                    (*table->free_value)((*entry)->value);
                (*entry)->key = key;
            }
        
            if (option & COLLECTION_MODE_COPY) 
            {
                (*entry)->key = table->copy_key ? (*table->copy_key)(key) : key;
                (*entry)->value = table->copy_value ? (*table->copy_value)(value) : value;
            }
            else
                (*entry)->value = value;

            (*entry)->hashcode = code;
            break;
        }

        entry = &(*entry)->next;
    }

    return entry;
}

void *cp_hashtable_put(cp_hashtable *table, void *key, void *value)
{
    return cp_hashtable_put_by_option(table, key, value, table->mode);
}

void *cp_hashtable_put_safe(cp_hashtable *table, void *key, void *value)
{
    return cp_hashtable_put_by_option(table, key, value, table->mode | COLLECTION_MODE_DEEP);
}

void *cp_hashtable_put_copy(cp_hashtable *table, void *key, void *value)
{
        return cp_hashtable_put_by_option(table, key, value, table->mode | COLLECTION_MODE_COPY);
}


/* actual insertion code */
void *cp_hashtable_put_by_option(cp_hashtable *table, void *key, void *value, int option)
{
    unsigned long code;
    cp_hashtable_entry **entry; /* defined ** for when initializing a bucket */ 
    int syncbit;
    void *ret = NULL;

    syncbit = table->mode & COLLECTION_MODE_NOSYNC;
    if (cp_hashtable_txlock(table, COLLECTION_LOCK_WRITE)) return NULL;

    if ((option & COLLECTION_MODE_NORESIZE) == 0)
    {
        if ((table->items * 100) > 
            (table->table_size * table->fill_factor_max))
        {
            int new_size = cp_hashtable_choose_size(table->table_size * 2);
            if (syncbit)
                cp_hashtable_resize_nosync(table, new_size);
            else
                cp_hashtable_resize(table, new_size);
        }
    }
    
    code = abs((*table->hash_fn)(key));

    entry = cp_hashtable_replace_internal(table, key, value, code, option, 0);

    if (*entry == NULL && table->resizing) 
    {
        cp_hashtable_entry **resize_entry;

        resize_entry = cp_hashtable_replace_internal(table, key, value, code, option, 1);
        if (*resize_entry != NULL)  /* prevent write */
            entry = resize_entry;

    }
        
    if (*entry == NULL) 
    {
        /* no entry found, entry points at lookup location */
        *entry = cp_hashtable_create_entry(table, option, key, value, code);
        if (*entry) 
            table->items++;
    }
    
    if (*entry)
        ret = (*entry)->value;

    if (!syncbit) cp_hashtable_txunlock(table);

    return ret;
}

/**
 * Internal remove an entry from the table by key.
 *
 * Get the value by key and destroy the entry.
 *
 * @param table the object
 * @param key    Key to search for.
 * @param code   Hash code of the Key (saves recalculating it)
 * @param mode   operation mode
 * @param resize 0: search in the normal table, 1: search in the resize table
 * @retval value of the entry with key.
 * @retval NULL otherwise (no entry with given key)
 */
void *cp_hashtable_remove_internal(cp_hashtable *table, void *key, long code, int mode, int resize)
{
    cp_hashtable_entry **entry, *item;
    void *value = NULL;

    entry = resize ? &table->resize_table[code % table->resize_len]
                   : &table->table[code % table->table_size];

    while (*entry != NULL) 
    {
        if ((*table->compare_fn)(key, (*entry)->key) == 0) 
        {
            /* entry now points either to the table->table element or to the  */
            /* next pointer in the parent entry                               */
            table->items--;
            item = *entry;
            *entry = item->next;

            value = item->value;
            if (mode & COLLECTION_MODE_DEEP) 
            {
                if (table->free_key)
                    (*table->free_key)(item->key);
                if (table->free_value)
                    (*table->free_value)(item->value);
            }

            free(item);

#if CP_HASHTABLE_MULTIPLE_VALUES
            if (!(mode & COLLECTION_MODE_MULTIPLE_VALUES)) 
#endif
            break; 
        }

        entry = &(*entry)->next;
    }

    return value;
}

/**
 * Remove an entry from the table by key with locking mode.
 *
 * Get the value by key and destroy the entry.
 *
 * @param table the object
 * @param key    Key to search for.
 * @param mode   operation/locking mode
 * @retval value of the entry with key.
 * @retval NULL otherwise (no entry with given key)
 */
void *cp_hashtable_remove_by_mode(cp_hashtable *table, void *key, int mode)
{
    long code;
    void *value = NULL;
    int syncbit = table->mode & COLLECTION_MODE_NOSYNC;
    if (cp_hashtable_txlock(table, COLLECTION_LOCK_WRITE)) return NULL;

    code = abs((*table->hash_fn)(key));

    value = cp_hashtable_remove_internal(table, key, code, mode, 0);

    if (table->resizing) 
    {
        void *resize_value;
        resize_value = cp_hashtable_remove_internal(table, key, code, mode, 1);
        if (value == NULL) value = resize_value;
    }

    if ((table->mode & COLLECTION_MODE_NORESIZE) == 0)
    {
        if (table->table_size > table->min_size &&
            ((table->items * 100) < 
            (table->table_size * table->fill_factor_min)))
        {
            int new_size = cp_hashtable_choose_size(table->table_size / 2);
            if (syncbit)
                cp_hashtable_resize_nosync(table, new_size);
            else
                cp_hashtable_resize(table, new_size);
        }
    }

    if (!syncbit) cp_hashtable_txunlock(table);

    return value;
}

int cp_hashtable_remove_all(cp_hashtable *table)
{
    long i;
    cp_hashtable_entry *entry, *next;
    int deepbit = table->mode & COLLECTION_MODE_DEEP;
    cp_destructor_fn dk = table->free_key;
    cp_destructor_fn dv = table->free_value;

    if (cp_hashtable_txlock(table, COLLECTION_LOCK_WRITE)) return -1;
    
    for (i = 0; i < table->table_size; i++) 
    {
        if ((entry = table->table[i]) != NULL) 
        {
            while (entry != NULL) 
            {
                next = entry->next;
                if (deepbit)
                {
                    if (dk) (*dk)(entry->key);
                    if (dv) (*dv)(entry->value);
                }
                free(entry);
                entry = next;
            }
        }
    }

    if (table->resizing) 
    {
        for (i = 0; i < table->resize_len; i++) 
            if ((entry = table->resize_table[i]) != NULL) 
                while (entry != NULL) 
                {
                    next = entry->next;
                    if (deepbit)
                    {
                        if (dk) (*dk)(entry->key);
                        if (dv) (*dv)(entry->value);
                    }
                    free(entry);
                    entry = next;
                }
    }

    cp_hashtable_txunlock(table);
	return 0;
}

/* remove using current mode settings */
void *cp_hashtable_remove(cp_hashtable *table, void *key)
{
    return cp_hashtable_remove_by_mode(table, key, table->mode);
}


/**
 * Remove with COLLECTION_MODE_DEEP set
 * @note here cp_hashtable_remove_by_mode returns an invalid pointer on 
 * success, since the item has just been released. We can not use it but we 
 * can tell it isn't null.
 */
int cp_hashtable_remove_deep(cp_hashtable *table, void *key)
{
    return cp_hashtable_remove_by_mode(table, key
              , table->mode | COLLECTION_MODE_DEEP) != NULL;
}

int cp_hashtable_contains(cp_hashtable *table, void *key)
{
    int rc = 0;

    if (table != NULL && key != NULL)
    {
        long code;
        void *val;
        int option;
        if (cp_hashtable_txlock(table, COLLECTION_LOCK_READ)) return -1;

        option = table->mode & COLLECTION_MODE_MULTIPLE_VALUES;
        code = abs((*table->hash_fn)(key));
        val = lookup_internal(table, key, code, table->mode, 0);
        /* when resizing, search also there */
        if (val == NULL && table->resizing) 
            val = lookup_internal(table, key, code, table->mode, 1);

        if (val != NULL)
		{
			rc = 1;
			if (option) cp_list_destroy(val);
		}

        cp_hashtable_txunlock(table);
    }
    else
        errno = EINVAL; //~~ and set rc = -1?

    return rc;
}


void *cp_hashtable_get(cp_hashtable *table, void *key)
{
    return cp_hashtable_get_by_option(table, key, table->mode);
}

void **cp_hashtable_get_keys(cp_hashtable *table)
{
    long i, j;
    void **keys;
    cp_hashtable_entry *entry = NULL;
    int rc = 0;

    if (table == NULL)
    {
        errno = EINVAL;
        return NULL;
    }

    if (cp_hashtable_txlock(table, COLLECTION_LOCK_READ)) return NULL;

    keys = (void **) calloc(table->items, sizeof(void *));
    if (keys == NULL) 
    {
        rc = ENOMEM;
        goto DONE;
    }
    
    for (i = 0, j = 0; i < table->table_size; i++) \
    {
        entry = table->table[i];
        while (entry != NULL) 
        {
            keys[j++] = entry->key;
            entry = entry->next;
        }
    }

    if (table->resizing) 
        for (i = 0; i < table->resize_len; i++) 
            entry = table->resize_table[i];
            while (entry != NULL) 
            {
                keys[j++] = entry->key;
                entry = entry->next;
            }
            
DONE:
    cp_hashtable_txunlock(table);
    if (rc) errno = rc;

    return keys;
}
    

unsigned long cp_hashtable_count(cp_hashtable *table)
{
    return (table) ? table->items : 0L; 
}


void **cp_hashtable_get_values(cp_hashtable *table)
{
    long i, j;
    void **values;
    cp_hashtable_entry *entry;
    int rc = 0;

    if (cp_hashtable_txlock(table, COLLECTION_LOCK_READ)) return NULL;

    values = (void **) malloc(table->items * sizeof(void *));
    if (values == NULL) 
    {
        rc = ENOMEM;
        goto DONE;
    }
    
    for (i = 0, j = 0; i < table->table_size; i++) 
    {
        entry = table->table[i];
        while (entry != NULL) 
        {
            values[j++] = entry->value;
            entry = entry->next;
        }
    }        

    if (table->resizing) 
    {
        for (i = 0; i < table->resize_len; i++) 
        {
            entry = table->resize_table[i];
            while (entry != NULL) 
            {
                values[j++] = entry->value;
                entry = entry->next;
            }
        }
    }

DONE:
    cp_hashtable_txunlock(table);
    if (rc) errno = rc;

    return values;
}


/* ------------------------------------------------------------------------ */
/* ---   hash function implementations for primitive types & strings    --- */
/* ------------------------------------------------------------------------ */

unsigned long cp_hash_int(void *i)
{
    return (long) *((int *) i);
}


int cp_hash_compare_int(void *i, void *j)
{
    return *((int *) i) - *((int *) j);
}

unsigned long cp_hash_long(void *l)
{
    return *((long *) l);
}


int cp_hash_compare_long(void *i, void *j)
{
    long diff = *((long *) i) - *((long *) j);
    return diff > 0 ? 1 : diff < 0 ? -1 : 0;
}

unsigned long cp_hash_addr(void *addr)
{
	return (unsigned long) addr;
}

int cp_hash_compare_addr(void *a1, void *a2)
{
	return (long) a1 - (long) a2;
}

unsigned long cp_hash_string(void *str)
{
    int i;
    long res;
    char *_str;
    
    if (str == NULL) return 0; 

    _str = (char *) str;
    
    for (i = 0, res = 1; *_str != '\0'; _str++)
        res = res * HASH_SEED + *_str;

    return res;
}
    

int cp_hash_compare_string(void *s1, void *s2)
{
    if (s1 == NULL && s2 == NULL) return 0;
    if (s1 == NULL || s2 == NULL) return -1;
    return strcmp((char *) s1, (char *) s2);
}

#define CP_CHAR_UC(x) ((x) >= 'a' && (x) <= 'z' ? ((x) - 'a' + 'A') : (x))
        
unsigned long cp_hash_istring(void *str)
{
    int i;
    long res;
    char *_str;
    
    if (str == NULL) return 0; 

    _str = (char *) str;
    
    for (i = 0, res = 1; *_str != '\0'; _str++)
        res = res * HASH_SEED + CP_CHAR_UC(*_str);

    return res;
}

int cp_hash_compare_istring(void *s1, void *s2)
{
    if (s1 == NULL && s2 == NULL) return 0;
    if (s1 == NULL || s2 == NULL) return -1;
    return strcasecmp((char *) s1, (char *) s2);
}

void *cp_hash_copy_string(void *element)
{
    char *src;
    char *dst;
    size_t len;
    
    src = (char *) element;
    len = strlen(src) + 1;
    dst = (char *) malloc(len * sizeof(char));

    if (dst == NULL) return NULL;

#ifdef CP_HAS_STRLCPY
    strlcpy(dst, src, len);
#else
    strcpy(dst, src);
#endif /* CP_HAS_STRLCPY */
    return dst;
}

unsigned long cp_hash_float(void *addr)
{
	unsigned long *p = (unsigned long *) addr;
	return *p;
}

int cp_hash_compare_float(void *a1, void *a2)
{
	float f1 = *(float *) a1;
	float f2 = *(float *) a2;
	if (f1 > f2) return 1;
	if (f1 < f2) return -1;
	return 0;
}

unsigned long cp_hash_double(void *d)
{
	unsigned long *p = (unsigned long *) d;
	if (sizeof(unsigned long) < sizeof(double))
		return p[0] ^ p[1];
	return *p;
}

int cp_hash_compare_double(void *a1, void *a2)
{
	double d1 = *(double *) a1;
	double d2 = *(double *) a2;
	if (d1 > d2) return 1;
	if (d1 < d2) return -1;
	return 0;
}

/** @} */

