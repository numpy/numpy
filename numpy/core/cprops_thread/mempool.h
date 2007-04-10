/* mempool.h - a memory pool implementation
 *
 * cp_mempool is a memory pool for fixed allocation sizes. 
 *
 * cp_shared_mempool is a collection of cp_mempool objects. cp_shared_mempool
 * allows sharing cp_mempool instances that serve the same allocation size. 
 * Call cp_shared_mempool_register to request an allocation size and use the 
 * returned cp_mempool. 
 *
 * cp_shared_mempool may also be used for aribitrary sizes allocations, but
 * this does not necessarily improve performance. Tests on Open BSD show 
 * significant gains, whereas tests on Linux show a performance degradation for
 * generic allocation operations. Using cp_shared_mempool to share cp_mempool
 * objects between cp_* data structures does not reduce performance. The test
 * results are not conclusive and performance characteristics may also be 
 * application specific. For best results, benchmark performance for your 
 * application in realistic deployment scenarios before deciding whether to use
 * cp_shared_mempool.
 *
 * After instantiating a cp_shared_mempool, you may set one of 
 * CP_SHARED_MEMPOOL_TYPE_2 or CP_SHARED_MEMPOOL_TYPE_1. If TYPE_2 is set, 
 * requests for unregistered allocation sizes will return the requested size
 * rounded up to the machine word size, after instantiating a cp_mempool 
 * serving the requested size if none exists. This could potentially use up
 * large amounts of memory. If TYPE_1 is set, unregistered allocation sizes
 * are rounded up to the next bit. E.g. a request for 513 bytes will return a
 * chunk of 1024 bytes. This might also use up large amounts of memory. 
 *
 * Both cp_mempool and cp_shared_mempool represent a trade off of memory for
 * speed and should not be used if memory is tight or if the total allocation
 * may exceed the amount of physical memory available to the program, so as to
 * prevent swapping. Note that even if using the provided cp_mempool_free or 
 * cp_shared_mempool_free functions, the memory returned to the pool is kept 
 * for future requests and is ultimately released back to the general "malloc"
 * library only when the memory pool is destroyed. 
 * 
 * AUTHOR: Kyle Wheeler, Ilan Aelion
 */
#ifndef _CP_MEMPOOL_H
#define _CP_MEMPOOL_H

#include "common.h"
#include "collection.h"

__BEGIN_DECLS

struct _cp_mempool;

typedef void (*cp_mempool_callback_fn)(void *prm, 
									   struct _cp_mempool *pool, 
									   void *mem);

typedef struct _cp_mempool
{
	size_t item_size;
	size_t alloc_size;
	size_t items_per_alloc;

	char *reuse_pool;
	char *alloc_pool;
	size_t alloc_pool_pos;

	int refcount;

	cp_mempool_callback_fn alloc_callback;
	void *callback_prm;

	int mode;
	cp_mutex *lock;
#if !defined(CP_HAS_PTHREAD_MUTEX_RECURSIVE) && !defined(CP_HAS_PTHREAD_MUTEX_RECURSIVE_NP)
    cp_thread txowner;
#endif /* CP_HAS_PTHREAD_MUTEX_RECURSIVE */
} cp_mempool;

#define cp_mempool_item_size(p) ((p)->item_size)

/* cp_mempool_create_by_option */
CPROPS_DLL
cp_mempool *cp_mempool_create_by_option(const int mode, 
                       		            size_t chunksize, 
                                    	size_t multiple);

/* cp_mempool_create_by_option */
CPROPS_DLL
cp_mempool *cp_mempool_create(const size_t chunksize);

/* increment refcount */
CPROPS_DLL
int cp_mempool_inc_refcount(cp_mempool *pool);

/* cp_mempool_alloc */
CPROPS_DLL
void *cp_mempool_alloc(cp_mempool * const pool);

/* cp_mempool_calloc */
CPROPS_DLL
void *cp_mempool_calloc(cp_mempool * const pool);

/* cp_mempool_free */
CPROPS_DLL
int cp_mempool_free(cp_mempool * const pool, void *data);

/* cp_mempool_destroy */
CPROPS_DLL
void cp_mempool_destroy(cp_mempool *pool);

#include "rb.h"
#include "hashtable.h"

typedef struct _shared_mempool_entry
{
	size_t item_size;
	cp_mempool *pool;
	struct _shared_mempool_entry *next;
} shared_mempool_entry;

/* cp_shared_mempool is a generalized memory pool. It allows requesting variable
 * block sizes. For best results, register the required block sizes in advance.
 * requests for unregistered block sizes will return memory from a default
 * internal list, which rounds up the block size to the next bit. For example
 * allocating an unregisterd block of size 12 will return a 16 byte block. 
 * In partcular large allocations could return a lot of extra memory.
 */
typedef CPROPS_DLL struct _cp_shared_mempool
{
	unsigned int reg_tbl_size;
	unsigned int reg_tbl_count;

	shared_mempool_entry **reg_tbl;
	struct _cp_rbtree *chunk_tracker;

	int mode;
	int gm_mode;

	/* lock for mempool lists */
	cp_mutex *lock;

	int multiple; /* number of pages to allocate in sub pools */
} cp_shared_mempool;

/* ``smaller'': arbitrary size allocations are rounded up to the next bit. The
 * pool is ``smaller'' in that up to about WORD_SIZE internal memory pools are
 * allocated to serve unregistered allocation size requests.
 */
#define CP_SHARED_MEMPOOL_TYPE_1 1
/* ``faster'': arbitrary size allocations are rounded up to the word size. The 
 * pool is ``faster'' in that typically the allocation overhead is smaller, and
 * the number of operations required to determine which pool to use internally
 * is smaller. On the other hand, since a large number of memory pool could be
 * allocated internally, this may not be usable in some circumstances. 
 */
#define CP_SHARED_MEMPOOL_TYPE_2 2

/* cp_shared_mempool_create */
CPROPS_DLL
cp_shared_mempool *cp_shared_mempool_create();

/* cp_shared_mempool_create_by_option */
CPROPS_DLL
cp_shared_mempool *
	cp_shared_mempool_create_by_option(int mode, 
									   int arbitrary_allocation_strategy,
									   int size_hint, 
									   int page_count);

/* cp_shared_mempool destroy */
CPROPS_DLL
void cp_shared_mempool_destroy(cp_shared_mempool *pool);

/* cp_shared_mempool_register */
CPROPS_DLL
cp_mempool *cp_shared_mempool_register(cp_shared_mempool *pool, size_t size);

/* cp_shared_mempool_alloc */
CPROPS_DLL
void *cp_shared_mempool_alloc(cp_shared_mempool *pool, size_t size);

/* cp_shared_mempool_calloc */
CPROPS_DLL
void *cp_shared_mempool_calloc(cp_shared_mempool *pool, size_t size);

/* cp_shared_mempool_free */
CPROPS_DLL
void cp_shared_mempool_free(cp_shared_mempool *pool, void *p);

__END_DECLS

#endif /* _CP_MEMPOOL_H */

