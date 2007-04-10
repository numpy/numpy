#include "mempool.h"
/* #include "util.h" */

#include <stdlib.h>		       /* for calloc() and malloc() */
#include <string.h>		       /* for memset() */
#include <errno.h>		       /* for errno and EINVAL */

#ifdef CP_HAS_GETPAGESIZE
#include <unistd.h>		       /* for getpagesize() */
#else
int getpagesize() { return 0x2000; }
#endif /* CP_HAS_GETPAGESIZE */

#ifndef WORD_SIZE
#define WORD_SIZE (sizeof(void *))
#endif /* WORD_SIZE */

#if defined(CP_HAS_PTHREAD_MUTEX_RECURSIVE) || defined(CP_HAS_PTHREAD_MUTEX_RECURSIVE_NP)
#define CP_MEMPOOL_TXLOCK(pool, err_ret) { \
    if (!((pool)->mode & COLLECTION_MODE_NOSYNC)) \
	if (cp_mutex_lock((pool)->lock)) \
	    return err_ret; \
}
#define CP_MEMPOOL_TXUNLOCK(pool, err_ret) { \
    if (!((pool)->mode & COLLECTION_MODE_NOSYNC)) \
	if (cp_mutex_unlock((pool)->lock)) \
	    return err_ret; \
}
#else
#define CP_MEMPOOL_TXLOCK(pool, err_ret) { \
    if (!((pool)->mode & COLLECTION_MODE_NOSYNC)) \
    { \
        cp_thread self = cp_thread_self(); \
        if (!cp_thread_equal(self, (pool)->txowner) && \
	        cp_mutex_lock((pool)->lock)) \
	        return err_ret; \
        (pool)->txowner = self; \
    } \
}
#define CP_MEMPOOL_TXUNLOCK(pool, err_ret) { \
    if (!((pool)->mode & COLLECTION_MODE_NOSYNC)) \
    { \
        cp_thread self = cp_thread_self(); \
        if (!cp_thread_equal(self, (pool)->txowner) && \
	        cp_mutex_unlock((pool)->lock)) \
	        return err_ret; \
        (pool)->txowner = 0; \
    } \
}
#endif /* CP_HAS_PTHREAD_MUTEX_RECURSIVE */
static size_t pagesize = 0;

cp_mempool *cp_mempool_create_by_option(const int mode, 
                                    	size_t item_size, 
                                    	size_t alloc_size)
{
	cp_mempool *pool = (cp_mempool *) calloc(1, sizeof(cp_mempool));
	if (pool == NULL) return NULL;

	pool->mode = mode;

	if (!(mode & COLLECTION_MODE_NOSYNC))
	{
#if defined(PTHREAD_MUTEX_RECURSIVE) || defined(PTHREAD_MUTEX_RECURSIVE_NP)
		pthread_mutexattr_t attr;
#endif /* PTHREAD_MUTEX_RECURSIVE */
		pool->lock = (cp_mutex *) malloc(sizeof(cp_mutex));
		if (pool->lock == NULL)
		{
			cp_mempool_destroy(pool);
			return NULL;
		}
#ifdef PTHREAD_MUTEX_RECURSIVE
		pthread_mutexattr_init(&attr);
		pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
		cp_mutex_init(pool->lock, &attr);
#else
		cp_mutex_init(pool->lock, NULL);
#endif /* PTHREAD_MUTEX_RECURSIVE */
	}

	if (pagesize == 0) pagesize = getpagesize();

	/* first, we ensure that item_size is a multiple of WORD_SIZE,
	 * and also that it is at least sizeof(void*). The first
	 * condition may imply the second on *most* platforms, but it
	 * costs us very little to make sure. */
	if (item_size < sizeof(void*)) item_size = sizeof(void*);
	if (item_size % WORD_SIZE)
		item_size += (WORD_SIZE) - (item_size % WORD_SIZE);
	pool->item_size = item_size;
	/* next, we pump up the alloc_size until it is at least big enough
	 * to hold ten chunks plus a void pointer, or ten pages, whichever
	 * is bigger. The reason for doing it this way rather than simply
	 * adding sizeof(void*) to alloc_size is that we want alloc_size to
	 * be a multiple of pagesize (this makes it faster!). */
	if (alloc_size < item_size * 10 + sizeof(void *))
		alloc_size = item_size * 10 + sizeof(void *);
	if (alloc_size < pagesize * 10) alloc_size = pagesize * 10;
	if (alloc_size % pagesize)
		alloc_size += pagesize - (alloc_size % pagesize);
	pool->alloc_size = alloc_size;

	pool->items_per_alloc = (alloc_size - sizeof(void *)) / item_size;

	pool->reuse_pool = NULL;
	pool->alloc_pool = (char *) malloc(alloc_size);
	if (pool->alloc_pool == NULL)
	{
		free(pool);
		return NULL;
	}
	*(void **) pool->alloc_pool = NULL;

	return pool;
}


cp_mempool *cp_mempool_create(const size_t item_size)
{
	return cp_mempool_create_by_option(COLLECTION_MODE_NOSYNC, item_size, 0);
}


void *cp_mempool_alloc(cp_mempool * const pool)
{
	void *p;

	CP_MEMPOOL_TXLOCK(pool, NULL);

	if (pool->reuse_pool)
	{
		p = pool->reuse_pool;
		pool->reuse_pool = *(void **)p;
	}
	else
	{
		if (pool->alloc_pool_pos == pool->items_per_alloc)
		{
			p = malloc(pool->alloc_size);
			if (p == NULL) return NULL;
			*(void **) p = pool->alloc_pool;
			pool->alloc_pool = p;
			pool->alloc_pool_pos = 0;
			/* if this pool is owned by a shared_mempool, report allocations */
			if (pool->alloc_callback) 
				(*pool->alloc_callback)(pool->callback_prm, pool, p);
		}
		p = pool->alloc_pool + sizeof(void *) + 
			pool->item_size * pool->alloc_pool_pos++;
	}

	CP_MEMPOOL_TXUNLOCK(pool, NULL);

	return p;
}

void *cp_mempool_calloc(cp_mempool * const pool)
{
	void *p = cp_mempool_alloc(pool);
	if (p)
		memset(p, 0, pool->item_size);
	return p;
}

int cp_mempool_free(cp_mempool * const pool, void *data)
{
	CP_MEMPOOL_TXLOCK(pool, -1);
	*(void **) data = pool->reuse_pool;
	pool->reuse_pool = data;
	CP_MEMPOOL_TXUNLOCK(pool, -1);
	return 0;
}

/* increment refcount */
int cp_mempool_inc_refcount(cp_mempool *pool)
{
	CP_MEMPOOL_TXLOCK(pool, -1);
	pool->refcount++;
	CP_MEMPOOL_TXUNLOCK(pool, -1);
	return 0;
}

void cp_mempool_destroy(cp_mempool *pool)
{
	if (pool)
	{
		if (pool->refcount-- <= 0)
		{
			void *p;

			while ((p = pool->alloc_pool))
			{
				pool->alloc_pool = *(void **) pool->alloc_pool;
				free(p);
			}
		}
	}
}

void cp_mempool_set_callback(cp_mempool *pool, void *prm, cp_mempool_callback_fn cb)

{
	pool->alloc_callback = cb;
	pool->callback_prm = prm;
}


/****************************************************************************
 *                                                                          *
 *                         cp_shared_mempool functions                      *
 *                                                                          *
 ****************************************************************************/

typedef struct _chunk_track
{
	void *mem;
	size_t size;
} chunk_track;

chunk_track *get_chunk_track(void *mem, size_t size)
{
	chunk_track *t = (chunk_track *) malloc(sizeof(chunk_track));
	if (t)
	{
		t->mem = mem;
		t->size = size;
	}
	return t;
}

int compare_chunk_track(void *c1, void *c2)
{
	chunk_track *t1 = c1;
	chunk_track *t2 = c2;
	return (t2->size == 0 && 
            t2->mem >= t1->mem && 
			((char *) t2->mem - (char *) t1->mem) < t1->size) ||
		   (t1->size == 0 && 
            t1->mem >= t2->mem && 
			((char *) t1->mem - (char *) t2->mem) < t2->size) ? 0 : 
		((char *) t1->mem - (char *) t2->mem);
}

cp_mempool *shared_mempool_entry_get(cp_shared_mempool *pool, size_t size)
{
	shared_mempool_entry *entry = pool->reg_tbl[size % pool->reg_tbl_size];

	while (entry && entry->item_size != size) entry = entry->next;
	if (entry) return entry->pool;

	return NULL;
}

cp_mempool *shared_mempool_entry_put(cp_shared_mempool *pool, 
									 size_t size, cp_mempool *sub)
{
	shared_mempool_entry **entry = &pool->reg_tbl[size % pool->reg_tbl_size];

	while ((*entry) && (*entry)->item_size != size) 
		entry = &(*entry)->next;

	if (*entry == NULL)
	{
		*entry = calloc(1, sizeof(shared_mempool_entry));
		(*entry)->item_size = size;
	}

	(*entry)->pool = sub;
	return sub;
}

void shared_mempool_entry_destroy(cp_shared_mempool *pool)
{
	int i;

	for (i = 0; i < pool->reg_tbl_size; i++)
	{
		shared_mempool_entry *curr, *tmp;
		curr = pool->reg_tbl[i];
		while (curr)
		{
			tmp = curr;
			curr = curr->next;
			cp_mempool_destroy(tmp->pool);
			free(tmp);
		}
	}

	free(pool->reg_tbl);
}

/* cp_shared_mempool_create */
cp_shared_mempool *cp_shared_mempool_create()
{
	return 
		cp_shared_mempool_create_by_option(0, CP_SHARED_MEMPOOL_TYPE_2, 0, 0);
}

/* cp_shared_mempool_create_by_option */
CPROPS_DLL
cp_shared_mempool *
	cp_shared_mempool_create_by_option(int mode, 
									   int arbitrary_allocation_strategy,
									   int size_hint, 
									   int page_count)
{
	cp_shared_mempool *pool = 
		(cp_shared_mempool *) calloc(1, sizeof(cp_shared_mempool));
	if (pool == NULL) return NULL;

	if (size_hint)
		size_hint = size_hint * 2 + 1; /* choose an odd number */
	else 
		size_hint = 211; /* 211 is a prime */

	pool->reg_tbl = calloc(size_hint, sizeof(shared_mempool_entry *));
	if (pool->reg_tbl == NULL) goto CREATE_ERROR;
	pool->reg_tbl_size = size_hint;

	pool->mode = mode;

	if ((mode & COLLECTION_MODE_NOSYNC))
	{
		pool->lock = (cp_mutex *) malloc(sizeof(cp_mutex));
		if (pool->lock == NULL) goto CREATE_ERROR;
		if ((cp_mutex_init(pool->lock, NULL))) goto CREATE_ERROR;
	}

	if (arbitrary_allocation_strategy == 0)
		pool->gm_mode = CP_SHARED_MEMPOOL_TYPE_1;
	else
		pool->gm_mode = arbitrary_allocation_strategy;

	pool->multiple = page_count;

	pool->chunk_tracker = 
		cp_rbtree_create_by_option(mode | COLLECTION_MODE_DEEP, 
								   compare_chunk_track, NULL, free, NULL, NULL);
	if (pool->chunk_tracker == NULL) goto CREATE_ERROR;

	return pool;

CREATE_ERROR:
	if (pool->lock)
	{
		free(pool->lock);
		pool->lock = NULL;
	}
	cp_shared_mempool_destroy(pool);
	return NULL;
}

/* cp_shared_mempool destroy */
CPROPS_DLL
void cp_shared_mempool_destroy(cp_shared_mempool *pool)
{
	if (pool)
	{
		cp_rbtree_destroy(pool->chunk_tracker);
		shared_mempool_entry_destroy(pool);
		if (pool->lock)
		{
			cp_mutex_destroy(pool->lock);
			free(pool->lock);
		}
		free(pool);
	}
}

void cp_shared_mempool_track_alloc(cp_shared_mempool *pool, 
								   cp_mempool *sub, void *mem)
{
	cp_rbtree_insert(pool->chunk_tracker, 
					 get_chunk_track(mem, sub->alloc_size), sub);
}

/* cp_shared_mempool_register */
cp_mempool *cp_shared_mempool_register(cp_shared_mempool *pool, size_t size)
{
	cp_mempool *sub;
	if (size % WORD_SIZE) size += WORD_SIZE - (size % WORD_SIZE);
	sub = shared_mempool_entry_get(pool, size);
	if (sub)
		cp_mempool_inc_refcount(sub);
	else
	{
		sub = cp_mempool_create_by_option(pool->mode, size, pool->multiple);
		cp_mempool_set_callback(sub, pool, 
			(cp_mempool_callback_fn) cp_shared_mempool_track_alloc);
		shared_mempool_entry_put(pool, size, sub);
	}

	return sub;
}

#if 0
/* unregister a mempool */
void cp_shared_mempool_unregister(cp_shared_mempool *pool, size_t size)
{
	cp_mempool *sub;
	if (size % WORD_SIZE) size += WORD_SIZE - (size % WORD_SIZE);
	sub = shared_mempool_entry_get(pool, size);
	if (sub)
		cp_mempool_destroy(sub);
}
#endif

/* cp_shared_mempool_alloc */
CPROPS_DLL
void *cp_shared_mempool_alloc(cp_shared_mempool *pool, size_t size)
{
	size_t actual;
	cp_mempool *mempool = NULL;

	if (size % WORD_SIZE) size += WORD_SIZE - (size % WORD_SIZE);
	
	if ((mempool = shared_mempool_entry_get(pool, size)))
		return cp_mempool_alloc(mempool);

	if ((pool->gm_mode & CP_SHARED_MEMPOOL_TYPE_2))
		actual = size;
	else
	{
		actual = WORD_SIZE;
		while (actual < size) actual <<= 1;
	}
	if ((mempool = cp_shared_mempool_register(pool, actual)))
		return cp_mempool_alloc(mempool);

	return NULL;
}

/* cp_shared_mempool_calloc */
CPROPS_DLL
void *cp_shared_mempool_calloc(cp_shared_mempool *pool, size_t size)
{
	size_t actual;
	cp_mempool *mempool = NULL;

	if (size % WORD_SIZE) size += WORD_SIZE - (size % WORD_SIZE);
	
	if ((mempool = shared_mempool_entry_get(pool, size)))
		return cp_mempool_calloc(mempool);

	if ((pool->gm_mode & CP_SHARED_MEMPOOL_TYPE_2))
		actual = size;
	else
	{
		actual = WORD_SIZE;
		while (actual < size) actual <<= 1;
	}
	if ((mempool = cp_shared_mempool_register(pool, actual)))
		return cp_mempool_calloc(mempool);

	return NULL;
}


/* cp_shared_mempool_free */
CPROPS_DLL
void cp_shared_mempool_free(cp_shared_mempool *pool, void *p)
{
	cp_mempool *mempool;
	chunk_track ct;
	memset(&ct, 0, sizeof(chunk_track));
	ct.mem = p;

	if ((mempool = cp_rbtree_get(pool->chunk_tracker, &ct)))
		cp_mempool_free(mempool, p);
}

