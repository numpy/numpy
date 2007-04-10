#include <stdio.h>
#include "thread.h"
#include "common.h"
#include "log.h"

/**
 * @addtogroup cp_thread
 */
/** @{ */

#include <stdlib.h>
#include <time.h>
#include <errno.h>

#ifdef _WINDOWS
#define __WINDOWS_WINNT 0x0400
#include <Windows.h>
#endif

/** internal function to return a cp_pooled_thread to the available pool */
static int cp_thread_pool_set_available(cp_thread_pool *owner, cp_pooled_thread *pt);

long cp_pooled_thread_get_id(cp_pooled_thread *pt)
{
	long id = 0;
#ifdef CP_HAS_PTHREAD_GETUNIQUE_NP
	pthread_getunique_np(*pt->worker, &id);
#else
	id = (long) *pt->worker;
#endif

	return id;
}

cp_pooled_thread *cp_pooled_thread_create(cp_thread_pool *owner)
{
	int rc;
	cp_pooled_thread *pt = calloc(1, sizeof(cp_pooled_thread));

	if (pt == NULL) 
	{
		cp_error(CP_MEMORY_ALLOCATION_FAILURE, "can\'t allocate pooled thread");
		errno = ENOMEM;
		return NULL;
	}
	pt->worker = calloc(1, sizeof(cp_thread));
	if (pt->worker == NULL)
	{
		cp_error(CP_MEMORY_ALLOCATION_FAILURE, "can\'t allocate thread");
		errno = ENOMEM;
		return NULL;
	}
		
	pt->owner = owner;

	pt->suspend_lock = (cp_mutex *) malloc(sizeof(cp_mutex));
	if (pt->suspend_lock == NULL) 
		goto THREAD_CREATE_CANCEL;
	if ((rc = cp_mutex_init(pt->suspend_lock, NULL)))
	{
		cp_error(rc, "starting up pooled thread");
		goto THREAD_CREATE_CANCEL;
	}

	pt->suspend_cond = (cp_cond *) malloc(sizeof(cp_cond));
	if ((rc = cp_cond_init(pt->suspend_cond, NULL)))
	{
		cp_error(rc, "starting up pooled thread");
		cp_mutex_destroy(pt->suspend_lock);
		free(pt->suspend_lock);
		goto THREAD_CREATE_CANCEL;
	}

	pt->done = 0;
	pt->wait = 1;

	cp_thread_create(*pt->worker, NULL, cp_pooled_thread_run, pt);

	pt->id = cp_pooled_thread_get_id(pt);
	cp_thread_detach(*pt->worker); //~~ check

	return pt;

THREAD_CREATE_CANCEL:
	free(pt->worker);
	free(pt);
	
	return NULL;
}
		
int cp_pooled_thread_stop(cp_pooled_thread *pt)
{
	int rc = 0;

	cp_mutex_lock(pt->suspend_lock);
	pt->action = NULL;
	if (pt->stop_fn)
	{
		rc = pt->stop_prm ? (*pt->stop_fn)(pt->stop_prm) : 
							  (*pt->stop_fn)(pt->action_prm);
	}

	pt->done = 1;
	pt->wait = 0;

	cp_cond_signal(pt->suspend_cond);
	cp_mutex_unlock(pt->suspend_lock);

//	cp_thread_join(*pt->worker, NULL); //~~ rc

	return rc;
}

void cp_pooled_thread_destroy(cp_pooled_thread *t)
{
#ifdef __TRACE__
	DEBUGMSG("destroying cp_pooled_thread %lX", t);
#endif
	cp_mutex_destroy(t->suspend_lock);
	free(t->suspend_lock);
	cp_cond_destroy(t->suspend_cond);
	free(t->suspend_cond);
	free(t->worker);
	free(t);
}

int cp_pooled_thread_release(cp_pooled_thread *t)
{
	return 0;
}

int cp_pooled_thread_run_task(cp_pooled_thread *pt, 
						      cp_thread_action action, 
							  void *prm)
{
#ifdef __TRACE__
	DEBUGMSG("cp_pooled_thread_run_task: action %lx, prm %lx\n", 
			 (long) action, (long) prm);
#endif

	pt->action = action;
	pt->action_prm = prm;

	if (action == NULL)
	{
		cp_error(CP_INVALID_FUNCTION_POINTER, "missing thread function");
		return CP_INVALID_FUNCTION_POINTER;
	}

	/* signal thread to run */
	cp_mutex_lock(pt->suspend_lock);
	pt->wait = 0;
	cp_cond_signal(pt->suspend_cond);
	cp_mutex_unlock(pt->suspend_lock);

	return 0;
}

int cp_pooled_thread_run_stoppable_task(cp_pooled_thread *pt, 
                                        cp_thread_action action, 
                                        void *action_prm, 
                                        cp_thread_stop_fn stop_fn,
                                        void *stop_prm)
{
	pt->stop_fn = stop_fn;
	pt->stop_prm = stop_prm;
	return cp_pooled_thread_run_task(pt, action, action_prm);
}

void *cp_pooled_thread_run(void *prm)
{
	cp_pooled_thread *pt = (cp_pooled_thread *) prm;

#ifdef __TRACE__
	DEBUGMSG("cp_pooled_thread (%lx) starts", (long) pt);
#endif

	while (!pt->done && pt->owner->running)
	{
		cp_mutex_lock(pt->suspend_lock);
		while (pt->wait && (!pt->done) && pt->owner->running)
			cp_cond_wait(pt->suspend_cond, pt->suspend_lock);
		cp_mutex_unlock(pt->suspend_lock);

		if (pt->done || !pt->owner->running) break;
#ifdef __TRACE__
		DEBUGMSG("cp_pooled_thread_run: action is %lX, action_prm is %lX", pt->action, pt->action_prm);
#endif
		if (pt->action) /* run user defined function if set */
		{
#ifdef __TRACE__
			DEBUGMSG("pooled thread (%lX) handles action (%lX)", (long) pt, (long) pt->action);
#endif
			(*pt->action)(pt->action_prm);
		}
		if (pt->done || !pt->owner->running) break;

		pt->wait = 1;
		/* performed work, notify pool */
		cp_thread_pool_set_available(pt->owner, pt);
	}

#ifdef __TRACE__
	DEBUGMSG("cp_pooled_thread (%lx) exits", (long) pt);
#endif

	cp_pooled_thread_destroy(pt);

	return NULL;
}

static int cp_thread_pool_set_available(cp_thread_pool *pool, 
		                                cp_pooled_thread *thread)
{
	cp_mutex_lock(pool->pool_lock);
	cp_hashlist_remove(pool->in_use, &thread->id);
	cp_list_append(pool->free_pool, thread);
//	pool->size--;
 	cp_cond_signal(pool->pool_cond);
	cp_mutex_unlock(pool->pool_lock);

	return 0;
}

	
int cp_thread_pool_wait(cp_thread_pool *pool)
{
	while (cp_hashlist_item_count(pool->in_use) && pool->running)
	{
		cp_mutex_lock(pool->pool_lock);
		cp_cond_wait(pool->pool_cond, pool->pool_lock);

		if (pool->running && 
			cp_hashlist_item_count(pool->in_use)) /* wake up someone else */
			cp_cond_signal(pool->pool_cond);
		cp_mutex_unlock(pool->pool_lock);
	}

	return 0;
}

int cp_thread_pool_stop(cp_thread_pool *pool)
{
	cp_hashlist_iterator *i;
	cp_list_iterator *j;
	cp_pooled_thread *pt;

	pool->running = 0;

	i = cp_hashlist_create_iterator(pool->in_use, COLLECTION_LOCK_READ);
	while ((pt = (cp_pooled_thread *) cp_hashlist_iterator_next_value(i)))
		cp_pooled_thread_stop(pt);
	cp_hashlist_iterator_destroy(i);

	j = cp_list_create_iterator(pool->free_pool, COLLECTION_LOCK_READ);
	while ((pt = (cp_pooled_thread *) cp_list_iterator_next(j)))
		cp_pooled_thread_stop(pt);
	cp_list_iterator_destroy(j);

	return 0;
}

cp_thread_pool *cp_thread_pool_create(int min_size, int max_size)
{
	int rc;
	cp_thread_pool *pool = calloc(1, sizeof(cp_thread_pool));
	if (pool == NULL)
		cp_fatal(CP_MEMORY_ALLOCATION_FAILURE, "can\'t allocate thread pool structure");

	pool->min_size = min_size;
	pool->max_size = max_size;

	pool->running = 1;

	pool->free_pool = cp_list_create();
	if (pool->free_pool == NULL)
		cp_fatal(CP_MEMORY_ALLOCATION_FAILURE, "can\'t allocate thread pool list");

	pool->in_use = cp_hashlist_create(10, cp_hash_long, cp_hash_compare_long);
	if (pool->in_use == NULL)
		cp_fatal(CP_MEMORY_ALLOCATION_FAILURE, "can\'t allocate thread pool running list");

	pool->pool_lock = (cp_mutex *) malloc(sizeof(cp_mutex));
	if (pool->pool_lock == NULL)
	{
		cp_error(CP_MEMORY_ALLOCATION_FAILURE, "can\'t create mutex");
		goto THREAD_POOL_CREATE_CANCEL;
	}
	if ((rc = cp_mutex_init(pool->pool_lock, NULL))) 
	{
		cp_error(rc, "can\'t create mutex");
		goto THREAD_POOL_CREATE_CANCEL;
	}

	pool->pool_cond = (cp_cond *) malloc(sizeof(cp_cond));
	if (pool->pool_cond == NULL)
	{
		cp_error(rc, "can\'t create condition variable");
		cp_mutex_destroy(pool->pool_lock);
		free(pool->pool_lock);
		goto THREAD_POOL_CREATE_CANCEL;
	}
	if ((rc = cp_cond_init(pool->pool_cond, NULL)))
	{
		cp_error(rc, "can\'t create condition variable");
		free(pool->pool_cond);
		cp_mutex_destroy(pool->pool_lock);
		free(pool->pool_lock);
		goto THREAD_POOL_CREATE_CANCEL;
	}

	for ( ; pool->size < pool->min_size; pool->size++)
	{
		cp_pooled_thread *pt = cp_pooled_thread_create(pool);
		if (pt == NULL)
		{    
            char msg[1000];		    
		    sprintf(msg, "can\'t create thread pool (created %d threads, minimum pool size is %d", pool->size, pool->min_size);
			cp_fatal(CP_THREAD_CREATION_FAILURE, msg);
		}
		cp_list_append(pool->free_pool, pt);
	}

	return pool;

THREAD_POOL_CREATE_CANCEL:
	cp_list_destroy_custom(pool->free_pool, 
			(cp_destructor_fn) cp_pooled_thread_destroy);
	cp_hashlist_destroy_custom(pool->in_use, NULL, 
			(cp_destructor_fn) cp_pooled_thread_destroy);
	free(pool);
	return NULL;
}

cp_thread *cp_thread_pool_get_impl(cp_thread_pool *pool, 
                                   cp_thread_action action, 
                                   void *action_prm, 
                                   cp_thread_stop_fn stop_fn,
                                   void *stop_prm,
                                   int block)
{
	cp_pooled_thread *pt = NULL;
	cp_mutex_lock(pool->pool_lock);
		
#ifdef __TRACE__
	DEBUGMSG("cp_thread_pool_get_impl (%d) pool size = %d max size = %d\n", block, pool->size, pool->max_size);
#endif

	pt = cp_list_remove_head(pool->free_pool);
	if (pt == NULL)
	{
		if (pool->size < pool->max_size)
		{
			pt = cp_pooled_thread_create(pool);
			if (pt)
				pool->size++;
		}

		if (pt == NULL) /* no thread available and poolsize == max */
		{
			if (!block)  /* asked not to block, return NULL */
			{
				cp_mutex_unlock(pool->pool_lock);
				return NULL;
			}

			/* wait for a thread to be released to the pool */
#ifdef _WINDOWS
			cp_mutex_unlock(pool->pool_lock);
#endif
			while (pool->running && cp_list_is_empty(pool->free_pool))
				cp_cond_wait(pool->pool_cond, pool->pool_lock);

			if (pool->running)
				pt = cp_list_remove_head(pool->free_pool);

			if (pt == NULL) /* shouldn't be happening except for shutdown */
			{
				cp_mutex_unlock(pool->pool_lock);
				return NULL;
			}
		}
	}

	cp_hashlist_append(pool->in_use, &pt->id, pt);
	cp_mutex_unlock(pool->pool_lock);

	cp_pooled_thread_run_stoppable_task(pt, action, action_prm, stop_fn, stop_prm);

	return pt->worker;
}

cp_thread *cp_thread_pool_get(cp_thread_pool *pool, 
						   	  cp_thread_action action, 
					   		  void *prm)
{
	return cp_thread_pool_get_impl(pool, action, prm, NULL, NULL, 1);
}

cp_thread *cp_thread_pool_get_stoppable(cp_thread_pool *pool, 
										cp_thread_action action, 
										void *action_prm, 
										cp_thread_stop_fn stop_fn, 
										void *stop_prm)
{
	return cp_thread_pool_get_impl(pool, action, action_prm, stop_fn, stop_prm, 1);
}

cp_thread *cp_thread_pool_get_nb(cp_thread_pool *pool, 
						  		 cp_thread_action action, 
						  		 void *prm)
{
	return cp_thread_pool_get_impl(pool, action, prm, NULL, NULL, 0);
}
	
cp_thread *cp_thread_pool_get_stoppable_nb(cp_thread_pool *pool, 
						  		 		   cp_thread_action action, 
						  		 		   void *action_prm,
								 		   cp_thread_stop_fn stop_fn, 
								 		   void *stop_prm)
{
	return cp_thread_pool_get_impl(pool, action, action_prm, stop_fn, stop_prm, 0);
}

void cp_thread_pool_destroy(cp_thread_pool *pool)
{
#ifdef __TRACE__
	DEBUGMSG("stopping cp_thread_pool %lX", pool);
#endif

	if (pool->running) cp_thread_pool_stop(pool);

	cp_list_destroy(pool->free_pool);
//	cp_list_destroy_custom(pool->free_pool, 
//				(cp_destructor_fn) cp_pooled_thread_destroy);
	cp_hashlist_destroy(pool->in_use);
//	cp_hashlist_destroy_custom(pool->in_use, NULL, 
//				(cp_destructor_fn) cp_pooled_thread_destroy);
	cp_mutex_destroy(pool->pool_lock);
	free(pool->pool_lock);
	cp_cond_destroy(pool->pool_cond);
	free(pool->pool_cond);

	free(pool);
}

int cp_thread_pool_count_available(cp_thread_pool *pool)
{
	return (pool->max_size - pool->size) + 
			cp_list_item_count(pool->free_pool);
}


/* **************************************************************************
 *                                                                          *
 *                      thread management framework                         *
 *                                                                          *
 ************************************************************************** */

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
		 void *stop_prm)
{
	cp_pooled_thread_client_interface *ci = 
		calloc(1, sizeof(cp_pooled_thread_client_interface));
	if (client == NULL)
		cp_fatal(CP_MEMORY_ALLOCATION_FAILURE, "can\'t allocate thread pool client interface");

	ci->owner = owner;
	ci->client = client;
	ci->min = min_threads;
	ci->max = max_threads;
	ci->report_load = report_load;
	ci->shrink = shrink;
	ci->action = action;
	ci->action_prm = action_prm;
	ci->stop_fn = stop_fn;
	ci->stop_prm = stop_prm;

	cp_pooled_thread_scheduler_register_client(owner, ci);

	return ci;
}


void cp_pooled_thread_client_interface_destroy
	(cp_pooled_thread_client_interface *client)
{
	free(client);
}

cp_pooled_thread_scheduler *cp_pooled_thread_scheduler_create(cp_thread_pool *pool)
{
	cp_pooled_thread_scheduler *scheduler = 
		calloc(1, sizeof(cp_pooled_thread_scheduler));
	if (scheduler == NULL)
		cp_fatal(CP_MEMORY_ALLOCATION_FAILURE, "can\'t allocate thread manager");

	scheduler->pool = pool;
	scheduler->client_list = cp_vector_create(20);

#ifdef CP_HAS_SRANDOM
	srandom(time(NULL));
#else
	srand(time(NULL));
#endif

	return scheduler;
}

void cp_pooled_thread_scheduler_destroy(cp_pooled_thread_scheduler *scheduler)
{
	cp_vector_destroy(scheduler->client_list);
	free(scheduler);
}

void cp_pooled_thread_scheduler_register_client
		(cp_pooled_thread_scheduler *scheduler, 
		 cp_pooled_thread_client_interface *client)
{
	cp_vector_add_element(scheduler->client_list, client);
}

cp_pooled_thread_client_interface *
	choose_random_client(cp_pooled_thread_scheduler *scheduler)
{
	int index = 
#ifdef CP_HAS_RANDOM
		random() 
#else
		rand()
#endif
		% (cp_vector_size(scheduler->client_list));
	return (cp_pooled_thread_client_interface *) 
		cp_vector_element_at(scheduler->client_list, index);
}

#define SCHEDULER_THRESHOLD 1

void cp_pooled_thread_client_get_nb(cp_pooled_thread_client_interface *c)
{
	if (cp_thread_pool_get_nb(c->owner->pool, c->action, c->action_prm))
		c->count++; 
}

void cp_pooled_thread_client_get(cp_pooled_thread_client_interface *c)
{
	c->count++;
	cp_thread_pool_get(c->owner->pool, c->action, c->action_prm);
}

void cp_pooled_thread_client_get_stoppable_nb(cp_pooled_thread_client_interface *c)
{
	if (cp_thread_pool_get_stoppable_nb(c->owner->pool, c->action, c->action_prm, c->stop_fn, c->stop_prm))
		c->count++; 
}

void cp_pooled_thread_client_get_stoppable(cp_pooled_thread_client_interface *c)
{
	c->count++;
	cp_thread_pool_get_stoppable(c->owner->pool, c->action, c->action_prm, c->stop_fn, c->stop_prm);
}

#if 0 //~~ for future optimization
void cp_pooled_thread_client_negociate(cp_pooled_thread_client_interface *c)
{
	int curr_load = (*c->report_load)(c);
	cp_pooled_thread_scheduler *scheduler = c->owner;
	Vector *clients = scheduler->client_list;
	int i, min, imin, max, imax, cval, max_count;
	cp_pooled_thread_client_interface *other;

	int clen = cp_vector_size(clients);

	if (clen == 0) return; //~~ warning for bad usage

	min = INT_MAX;
	max = -1;
	max_count = -1;

	for (i = 0; i < clen; i++)
	{
		other = (cp_pooled_thread_client_interface *) cp_vector_element_at(clients, i);
		cval = (*other->report_load)(other);

		if (other->count < other->min) //~~ what's with the switching
			cp_thread_pool_get_nb(other->owner->pool, other->action, other->prm);

		DEBUGMSG("negociate: pool %d: load = %d, %d <= %d <= %d", i, cval, other->min, other->count, other->max);
		if (cval > max) { max = cval; imax = i; }
		if (cval < min || (cval == min && other->count > max_count)) { min = cval; imin = i; max_count = other->count; }
	}
	DEBUGMSG("negociate: min = %d, max = %d", min, max);
	if (abs(max - min) > SCHEDULER_THRESHOLD)
	{
		cp_pooled_thread_client_interface *maxc = cp_vector_element_at(clients, imax);
		cp_pooled_thread_client_interface *minc = cp_vector_element_at(clients, imin);

		if (cp_thread_pool_count_available(scheduler->pool) == 0 &&
			minc->count > minc->min)
		{
			DEBUGMSG("negociate: shrinking min pool (%d)", imin);
			(*minc->shrink)(minc);
		}
			
		DEBUGMSG("negociate: get_nb for max pool (%d)", imax);
		if (cp_thread_pool_get_nb(maxc->owner->pool, maxc->action, maxc->prm))
			maxc->count++;
	}
}
		
#endif

void cp_pooled_thread_client_negociate(cp_pooled_thread_client_interface *c)
{
	int curr_load;
	if (c->owner->bypass) return;

	curr_load = (*c->report_load)(c);

#ifdef __TRACE__
	DEBUGMSG("negociate for client %lx - load == %d\n", (long) c, curr_load);
#endif
	/* if needed, try get idle thread from pool */
	if (curr_load > SCHEDULER_THRESHOLD && c->count < c->max) 
	{
		cp_thread *t = 
			cp_thread_pool_get_nb(c->owner->pool, c->action, c->action_prm);
		if (t) /* got a thread, return */
		{
#ifdef __TRACE__
	DEBUGMSG("negociate: got thread from pool");
#endif
			c->count++;
			return;
		}
	}

	/* 
	 * following code runs if 
	 * 
	 * (1) no threads needed - check if someone else (other client) needs one
	 * 
 	 * or
	 * 
	 * (2) thread needed but no idle threads in pool, ask someone else
	 */
	{
		int other_load;
		int *inc; /* the thread count to be incremented on switch */
		cp_thread *t;
		cp_pooled_thread_client_interface *load, *unload;
		cp_pooled_thread_client_interface *other = 
			choose_random_client(c->owner);

		other_load = (*other->report_load)(other);
		if (abs(curr_load - other_load) < SCHEDULER_THRESHOLD) return; 
		if (curr_load > other_load) /* other releases thread */
		{
			if (c->count >= c->max || 
				other->count <= other->min) return; /* no switch */
#ifdef __TRACE__
			DEBUGMSG("negociate: switching a thread to this pool");
#endif
			inc = &c->count;
			other->count--;
			load = c;
			unload = other;
		}
		else /* client releases thread */
		{
			if (c->count <= c->min ||
				other->count >= other->max) return; /* no switch */
#ifdef __TRACE__
			DEBUGMSG("negociate: switching a thread to other pool");
#endif
			inc = &other->count;
			c->count--;
			load = other;
			unload = c;
		}
#ifdef __TRACE__
		DEBUGMSG("negociate: shrinking unload pool");
#endif
		(*unload->shrink)(unload);
		t = cp_thread_pool_get_nb(load->owner->pool, //~~ get_nb?
							   load->action, load->action_prm); 

		if (t) (*inc)++; //~~ maybe better to get blocking and be sure?
	}
}

/** @} */

