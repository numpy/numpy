#ifndef _CP_COLLECTION_H
#define _CP_COLLECTION_H

#include "cp_config.h"
#include "common.h"

__BEGIN_DECLS

#ifdef CP_HAS_PTHREAD_H
#include <pthread.h>
#endif

#ifdef _WINDOWS
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0400 /* for SignalObjectAndWait */
#include <Windows.h>
#endif

/** @{ */
/**
 * @file
 *
 * The c collection classes provide plain c implementations of famous data
 * structures and algorithms such as linked lists, hash tables, hash lists and
 * an extensible tree implementation (see graph.h). Behavior in terms of
 * synchronization and member uniqueness may be set on initialization using the
 * appropriate constructor function with the required flags. The default 
 * implementations are synchronized, and mostly allow mutliple values with the
 * notable exception of the cp_hashtable and cp_hashlist collections which do 
 * not by default. Other subtle differences deriving from data structure 
 * characteristics or implentation inconsistencies would suggest reading the 
 * inline documentation in the header files for the specific collection you 
 * intend to use. 
 *
 * This header file defines macros and function types used commonly throughout 
 * the package.
 */

/** use collection defaults */
#define COLLECTION_MODE_PLAIN             0
/** collection copies and deletes elements (keys, values) */
#define COLLECTION_MODE_DEEP              1
/** collection allows non-unique keys */
#define COLLECTION_MODE_MULTIPLE_VALUES   2
/** collection stores copies of elements (keys, values) */
#define COLLECTION_MODE_COPY              4
/** no synchronization - suitable for the single threaded situation or if you 
  * want to do the synchronization yourself. */
#define COLLECTION_MODE_NOSYNC            8
/** 
 * The collection does not resize underlying hashtables. It might make sense 
 * to set this temporarily in code sections that shouldn't be unexpectedly 
 * slowed down by a resize operation, but resize should be allowed if the 
 * table fill factor is expected to go over ~70%, which is the point at which
 * hashtable performace is rumored to start degrading. 
 */
#define COLLECTION_MODE_NORESIZE         16
/**
 * hashlist multiple values are returned in list order (O(N)) rather than 
 * insertion order (O(1))
 */
#define COLLECTION_MODE_LIST_ORDER		 32
/**
 * indicates a transaction is in progress
 */
#define COLLECTION_MODE_IN_TRANSACTION   64

/** no lock */
#define COLLECTION_LOCK_NONE   0
/** lock for reading */
#define COLLECTION_LOCK_READ   1
/** lock for writing */
#define COLLECTION_LOCK_WRITE  2

/**
 * copy function.
 *
 * In cases where the collection holds copies rather than references to the
 * original objects. To do this you need to provide a copy function for
 * the items.
 */
typedef void *(*cp_copy_fn)(void *);

/**
 * destructor function.
 */
typedef void (*cp_destructor_fn)(void *);

/**
 * comparator functions implement strcmp semantics - 0 for identical keys, 
 * non-zero otherwise.
 */
typedef int (*cp_compare_fn)(void *, void *);

/**
 * callback function for iterator callback etc
 */
typedef int (*cp_callback_fn)(void *entry, void *client_prm);

/**
 * lock for collection types - current implementation uses pthread_rwlock_t
 *
 * _WINDOWS implementation for cp_cond is based on "Strategies for Implementing 
 * POSIX Condition Variables on _WINDOWS" by Douglas C. Schmidt and Irfan Pyarali
 * see http://www.cs.wustl.edu/~schmidt/_WINDOWS-cv-1.html
 */
#ifdef CP_HAS_PTHREAD_H
typedef pthread_t cp_thread;
typedef pthread_rwlock_t cp_lock;
typedef pthread_mutex_t cp_mutex;
typedef pthread_cond_t cp_cond;
#define cp_thread_create(thread, attr, fn, prm) pthread_create(&(thread), attr, fn, prm)
#define cp_thread_join pthread_join
#define cp_thread_detach pthread_detach
#define cp_thread_self pthread_self
#define cp_mutex_init pthread_mutex_init
#define cp_mutex_lock pthread_mutex_lock
#define cp_mutex_unlock pthread_mutex_unlock
#define cp_mutex_destroy pthread_mutex_destroy
#define cp_cond_init pthread_cond_init
#define cp_cond_wait pthread_cond_wait
#define cp_cond_signal pthread_cond_signal
#define cp_cond_broadcast pthread_cond_broadcast
#define cp_cond_destroy pthread_cond_destroy
#define cp_lock_init pthread_rwlock_init
#define cp_lock_rdlock pthread_rwlock_rdlock
#define cp_lock_wrlock pthread_rwlock_wrlock
#define cp_lock_unlock pthread_rwlock_unlock
#define cp_lock_destroy pthread_rwlock_destroy
#define cp_thread_equal pthread_equal
#ifndef CP_HAS_PTHREAD_MUTEX_RECURSIVE
#ifdef CP_HAS_PTHREAD_MUTEX_RECURSIVE_NP
#define CP_HAS_PTHREAD_MUTEX_RECURSIVE CP_HAS_PTHREAD_MUTEX_RECURSIVE_NP
#endif /* CP_HAS_PTHREAD_MUTEX_RECURSIVE_NP */
#endif /* CP_HAS_PTHREAD_MUTEX_RECURSIVE */
#else 
#ifdef _WINDOWS
typedef HANDLE cp_thread;
typedef HANDLE *cp_mutex;
#define cp_thread_create(thread, attr, fn, prm) \
	(((thread) = CreateThread(attr, 0, (LPTHREAD_START_ROUTINE) fn, prm, 0, NULL)) == NULL)
#define cp_thread_join(thread, exp) \
	{ \
		cp_thread p = thread; \
		WaitForSingleObject(p, INFINITE); \
	}
#define cp_thread_detach 
#define cp_thread_self GetCurrentThread
CPROPS_DLL
int cp_mutex_init(cp_mutex *mutex, void *attr);
#define cp_mutex_lock(mutex) (WaitForSingleObject((*(mutex)), INFINITE))
#define cp_mutex_unlock(mutex) (ReleaseMutex(*(mutex)))
#define cp_mutex_destroy(mutex) (CloseHandle(*(mutex)))

/* WIN32 implementation of a basic POSIX-condition-variable-like API
 * 
 * based on "Strategies for Implementing POSIX Condition Variables on WIN32"
 * by Douglas C. Schmidt and Irfan Pyarali - 
 * see http://www.cs.wustl.edu/~schmidt/WIN32-cv-1.html
 */
typedef CPROPS_DLL struct
{
  int waiters_count_;
  // Number of waiting threads.

  CRITICAL_SECTION waiters_count_lock_;
  // Serialize access to <waiters_count_>.

  HANDLE sema_;
  // Semaphore used to queue up threads waiting for the condition to
  // become signaled. 

  HANDLE waiters_done_;
  // An auto-reset event used by the broadcast/signal thread to wait
  // for all the waiting thread(s) to wake up and be released from the
  // semaphore. 

  size_t was_broadcast_;
  // Keeps track of whether we were broadcasting or signaling.  This
  // allows us to optimize the code if we're just signaling.
} cp_cond;

CPROPS_DLL int cp_cond_init(cp_cond *cv, const void *attr); // pthread_condattr_t *)
CPROPS_DLL int cp_cond_wait(cp_cond *cv, cp_mutex *mutex);
CPROPS_DLL int cp_cond_signal(cp_cond *cv);
CPROPS_DLL int cp_cond_broadcast(cp_cond *cv);
CPROPS_DLL int cp_cond_destroy(cp_cond *cv);

/* WIN32 implementation of a basic POSIX-read-write-lock-like API. cp_lock
 * is not upgradeable, ie attempting to obtain the lock if the current
 * thread already owns it causes deadlock.
 */
typedef CPROPS_DLL struct _cp_lock
{
	HANDLE access_mutex;
//	HANDLE write_mutex;

	DWORD writer;

	int readers;
	int writer_waiting;

} cp_lock;

CPROPS_DLL int cp_lock_init(cp_lock *lock, void *attr);
CPROPS_DLL int cp_lock_rdlock(cp_lock *lock);
CPROPS_DLL int cp_lock_wrlock(cp_lock *lock);
CPROPS_DLL int cp_lock_unlock(cp_lock *lock);
CPROPS_DLL int cp_lock_destroy(cp_lock *lock);

#define cp_thread_equal(p, q) ((p) == (q))
#endif
#endif

typedef CPROPS_DLL struct _cp_wrap
{
	void *item;
	cp_destructor_fn dtr;
} cp_wrap;

CPROPS_DLL cp_wrap *cp_wrap_new(void *item, cp_destructor_fn dtr);
CPROPS_DLL void cp_wrap_delete(cp_wrap *wrap);
	
typedef CPROPS_DLL struct _cp_mapping
{
	void *key;
	void *value;
} cp_mapping;

#define cp_mapping_key(m) 	((m)->key)
#define cp_mapping_value(m)	((m)->value)

typedef int (*cp_mapping_cmp_fn)(cp_mapping *a, cp_mapping *b);

/* free an allocation made by a cprops api function. On Windows you can't just
 * call free on memory allocated in a call to a DLL function.
 */
#ifdef _WINDOWS
CPROPS_DLL
void *cp_malloc(size_t size);
CPROPS_DLL
void *cp_calloc(size_t count, size_t size);
CPROPS_DLL 
void *cp_realloc(void *p, size_t size);
CPROPS_DLL
void cp_free(void *p);
#else
#define cp_malloc malloc
#define cp_calloc calloc
#define cp_realloc realloc
#define cp_free free
#endif /* _WINDOWS */

struct _cp_mempool;
struct _cp_shared_mempool;

__END_DECLS

/** @} */

#endif

