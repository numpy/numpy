#ifndef _CP_THREAD_H
#define _CP_THREAD_H

/**
 * @addtogroup cp_thread
 */
/** @{ */
/** 
 * @file
 * definitions for the thread management framework. the
 * current implementation is based on the POSIX thread api. 
 */

#include "common.h"

__BEGIN_DECLS

#include "cp_config.h"
/* synchronization api abstraction (a bunch of pthread-like macros basically)
 * is in collection.h
 */
#include "collection.h"
#include "linked_list.h"
#include "hashlist.h"
#include "vector.h"

/** a thread function */
typedef void *(*cp_thread_action) (void *);


/** a stop function for client threads */
typedef int (*cp_thread_stop_fn)(void *);

/**
 * cp_pooled_thread is a thread that lives in a thread_pool. The struct holds 
 * synchronization elements used to control the thread and actuation settings, 
 * i.e. a thread function and a parameter to be passed for the next starting 
 * thread. The pool initializes a bunch of threads and sets them in 'wait'. 
 *
 * When a client requests threads from the pool, the next available thread is
 * signalled out of 'wait', and runs the thread function requested by the 
 * client (see cp_thread_pool_get_impl). When the client thread function is 
 * done, the cp_pooled_thread returns to the thread pool and becomes available 
 * to pool clients. The cp_pooled_thread only exits when the pool exits, unless
 * explicitly stopped (eg pthread_exit) by client code. 
 */ 
typedef CPROPS_DLL struct _cp_pooled_thread
{
	long id;						///< integral (implementation specific) thread id
	struct _cp_thread_pool *owner;	///< the pool this thread belongs to
	cp_thread *worker;				///< the actual thread
	cp_thread_action action;		///< thread function for the next assignment
	void *action_prm;				///< parameter for the next assignment
	cp_thread_stop_fn stop_fn;		///< called on cp_pooled_thread_stop
	void *stop_prm;					///< if not set, stop_fn invoked w/ action_prm

	cp_mutex *suspend_lock;			///< lock for framework scheduling
	cp_cond *suspend_cond;			///< condition variable for framework scheduling

	int done;						///< done flag
	int wait;						///< wait flag
} cp_pooled_thread;

/** return a thread id for this thread */
CPROPS_DLL
long cp_pooled_thread_get_id(cp_pooled_thread *thread);

/** thread constructor function */
CPROPS_DLL
cp_pooled_thread *cp_pooled_thread_create(struct _cp_thread_pool *owner);

/** thread destructor */
CPROPS_DLL
void cp_pooled_thread_destroy(cp_pooled_thread *t);

/** signal a thread to stop */
CPROPS_DLL
int cp_pooled_thread_stop(cp_pooled_thread *t);

/** retrieve an integer type thread id for this thread */
CPROPS_DLL
long cp_pooled_thread_get_id(cp_pooled_thread *t);

/** sets the action and prm for this thread, then invokes 'action' */
CPROPS_DLL
int cp_pooled_thread_run_task(cp_pooled_thread *pt, 
							  cp_thread_action action, 
							  void *prm);

/** perform action with stop function */
CPROPS_DLL
int cp_pooled_thread_run_stoppable_task(cp_pooled_thread *pt, 
									  	cp_thread_action action,
									  	void *action_prm,
									  	cp_thread_stop_fn stop_fn,
									  	void *stop_prm);

/** framework thread function */
CPROPS_DLL
void *cp_pooled_thread_run(void *prm);
	

/** 
 * cp_thread_pool holds a list of free threads (in wait mode). The list grows 
 * up to max_size, after which subsequent calls to cp_thread_pool_get will 
 * block, and calls to cp_thread_pool_get_nb will return NULL - until clients 
 * return their threads to the pool. 
 */
typedef CPROPS_DLL struct _cp_thread_pool
{
	int size;				///< current size

	int min_size;			///< initial size
	int max_size;			///< size limit

	int running;

	cp_mutex *pool_lock;	///< to sync thread assignment and release
	cp_cond *pool_cond;		///< to sync thread assignment and release

	cp_list *free_pool;		///< holder for unused threads
	cp_hashlist *in_use;	///< holder for running threads
} cp_thread_pool;

/** cp_thread_pool constructor */
CPROPS_DLL
cp_thread_pool *cp_thread_pool_create(int min_size, int max_size);

/** cp_thread_pool destructor */
CPROPS_DLL
void cp_thread_pool_destroy(cp_thread_pool *pool);

/** wait for threads to finish processing client requests */
CPROPS_DLL
int cp_thread_pool_wait(cp_thread_pool *pool);

/** signal all threads in this pool to stop */
CPROPS_DLL
int cp_thread_pool_stop(cp_thread_pool *pool);

/** 
 * request a thread from the pool. If no threads are available, this function
 * will block until a thread becomes available.
 */
CPROPS_DLL 
cp_thread *cp_thread_pool_get(cp_thread_pool *pool, 
						   	  cp_thread_action action, 
						   	  void *prm);

/** 
 * request a thread from the pool. If no threads are available, this function
 * will block until a thread becomes available.
 */
CPROPS_DLL 
cp_thread *cp_thread_pool_get_stoppable(cp_thread_pool *pool, 
						   		  		cp_thread_action action, 
						   		  		void *action_prm, 
										cp_thread_stop_fn stop_fn,
										void *stop_prm);

/** 
 * request a thread from the pool - non-blocking version. Returns a pointer 
 * to the requested thread if one is available or NULL if the pool is empty. 
 */
CPROPS_DLL 
cp_thread *cp_thread_pool_get_nb(cp_thread_pool *pool, 
							  	 cp_thread_action action, 
							  	 void *prm);
/** 
 * request a thread from the pool - non-blocking version. Returns a pointer 
 * to the requested thread if one is available or NULL if the pool is empty. 
 */
CPROPS_DLL 
cp_thread *cp_thread_pool_get_stoppable_nb(cp_thread_pool *pool, 
							  	  cp_thread_action action, 
							  	  void *action_prm,
								  cp_thread_stop_fn stop_fn,
								  void *stop_prm);

/** returns the number of available threads in the pool. */
CPROPS_DLL 
int cp_thread_pool_count_available(cp_thread_pool *pool);

/* **************************************************************************
 *                                                                          *
 *                      thread management framework                         *
 *                                                                          *
 ************************************************************************** */


/**
 * Definitions for thread management framework follow. The framework is based
 * on the cp_thread_pool and cp_pooled_thread types. 
 * <br>
 * The pooled thread scheduler interface is meant for use by clients who 
 * require a variable number of threads. Each such component should create 
 * an instance of cp_pooled_thread_client_interface and use the api functions
 * to get threads from the underlying cp_thread_pool. Here is some example
 * code. 
 * <p><tt><code><pre>
 * cp_pooled_thread_scheduler *main_scheduler; 
 * 
 * ...
 *
 * component_a_start(component_a *a, ...)
 * {
 *     a->scheduler_interface = 
 *         cp_pooled_thread_client_interface_create(main_scheduler, a, 2, 10, 
 *             component_a_report_load, component_a_stop_thread, 
 *             component_a_thread_run, a);
 * 
 *     ...
 * 
 *     for (i = 0; i < a->scheduler_interface->min; i++)
 *         cp_pooled_thread_client_get(a->scheduler_interface);
 * }
 *
 * component_b_start(component_b *b, ...)
 * {
 *     b->scheduler_interface = 
 *         cp_pooled_thread_client_interface_create(main_scheduler, b, 2, 10, 
 *             component_a_report_load, component_a_stop_thread, 
 *             component_a_thread_run, b);
 * 
 *     ...
 * 
 *     for (i = 0; i < b->scheduler_interface->min; i++)
 *         cp_pooled_thread_client_get(b->scheduler_interface);
 * }
 * 
 * </tt></code></pre><br>
 * In this example, the threads for component_a and component_b will be 
 * managed jointly, since their cp_pooled_thread_client_interface *'s have the 
 * same cp_pooled_thread_scheduler *. <p>
 * 
 * 
 * See cp_pooled_thread_client_negociate for details. 
 */
typedef CPROPS_DLL struct cp_pooled_thread_scheduler
{
	cp_thread_pool *pool; ///< pool to operate on
	cp_vector *client_list; ///< list of clients
	int bypass; ///< when bypass flag set 'negociate' function becomes nop
} cp_pooled_thread_scheduler;

CPROPS_DLL struct _cp_pooled_thread_client_interface;

typedef int (*cp_pooled_thread_report_load) //~~ change prm to void *client
	(struct _cp_pooled_thread_client_interface *s);
typedef void (*cp_pooled_thread_shrink) //~~ change prm to void *client
	(struct _cp_pooled_thread_client_interface *);

/**
 * cp_pooled_thread_client_interface acts as the link to the 
 * cp_pooled_thread_scheduler for clients that require a variable number of 
 * threads. This interface holds 3 functions pointers that must be supplied
 * by a client: <br>
 * <li> report_load - should return the number of open requests the client has
 * to handle
 * <li> shrink - will be called by the framework to stop one client thread 
 * <li> action - the thread function for this client
 */
typedef CPROPS_DLL struct _cp_pooled_thread_client_interface
{
	int max;	///< thread count upper limit 
	int min;	///< thread count bottom limit

	int count;	///< actual number of threads serving this client

	void *client;	///< pointer to client
	cp_pooled_thread_scheduler *owner;	///< pointer to thread_pool_scheduler
	cp_pooled_thread_report_load report_load; ///< client function to report load
	cp_pooled_thread_shrink shrink; ///< client function to stop 1 thread
	cp_thread_action action; ///< client thread function 
	void *action_prm; ///< parameter to client thread function (usually == client)
	cp_thread_stop_fn stop_fn; ///< stop callback 
	void *stop_prm; ///< parameter to stop callback - if NULL action_prm will be used
} cp_pooled_thread_client_interface;

/** cp_pooled_thread_client_interface constructor */
CPROPS_DLL
cp_pooled_thread_client_interface *
	cp_pooled_thread_client_interface_create
		(cp_pooled_thread_scheduler *owner, 
		 void *client, 
		 int min_threads, 
		 int max_threads,
		 cp_pooled_thread_report_load report_load,
		 cp_pooled_thread_shrink shrink,
		 cp_thread_action action,
		 void *action_prm,
		 cp_thread_stop_fn stop_fn,
		 void *stop_prm);

/** cp_pooled_thread_client_interface destructor */
CPROPS_DLL
void cp_pooled_thread_client_interface_destroy
	(cp_pooled_thread_client_interface *client);

/**
 * threads should call negociate when a change in the number of threads a 
 * client owns is required. 
 * Two possible scheduling approaches are -
 * 
 * (1) centralized:
 *     Clients report their load factor to the thread manager. The thread 
 *     manager grants requests for new threads to clients with higher loads
 *     first. 
 * 
 * (2) distributed:
 *     clients negociate with other clients requesting a thread based on their
 *     load factors. The client with the lower load factor releases a thread 
 *     to the client with the higher load factor. 
 * 
 * The distributed approach saves some bookkeeping over head in the thread 
 * manager, reduces the number steps involved in acquiring a new thread or
 * releasing an unused one, and makes a dedicated synchronization thread 
 * unnecessary. <p>
 * 
 * In the current implementation, the scheduler will randomly choose one 
 * other client to negociate with. If the load factors are different enough, 
 * one thread will be switched to the busier client.
 */
CPROPS_DLL
void cp_pooled_thread_client_negociate(cp_pooled_thread_client_interface *c);

/** cp_pooled_thread_scheduler constructor */
CPROPS_DLL
cp_pooled_thread_scheduler *cp_pooled_thread_scheduler_create(cp_thread_pool *pool);

/** cp_pooled_thread_scheduler destructor */
CPROPS_DLL
void cp_pooled_thread_scheduler_destroy(cp_pooled_thread_scheduler *scheduler);

/** register client as a client of this scheduler */
CPROPS_DLL
void cp_pooled_thread_scheduler_register_client
		(cp_pooled_thread_scheduler *scheduler, 
		 cp_pooled_thread_client_interface *client);

/** 
 * convenience to abstract cp_thread_pool based implementation, see 
 * cp_pooled_thread_get and cp_pooled_thread_get_nb
 */
CPROPS_DLL 
void cp_pooled_thread_client_get(cp_pooled_thread_client_interface *c);

/** 
 * convenience to abstract cp_thread_pool based implementation, see 
 * cp_pooled_thread_get and cp_pooled_thread_get_nb
 */
CPROPS_DLL 
void cp_pooled_thread_client_get_stoppable(cp_pooled_thread_client_interface *c);

/** 
 * convenience to abstract cp_thread_pool based implementation, see 
 * cp_pooled_thread_get and cp_pooled_thread_get_nb
 */
CPROPS_DLL 
void cp_pooled_thread_client_get_nb(cp_pooled_thread_client_interface *c);

/** 
 * convenience to abstract cp_thread_pool based implementation, see 
 * cp_pooled_thread_get and cp_pooled_thread_get_nb
 */
CPROPS_DLL 
void cp_pooled_thread_client_get_stoppable_nb(cp_pooled_thread_client_interface *c);

__END_DECLS

/** @} */

#endif /* _CP_THREAD */

