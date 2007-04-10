#include <stdlib.h>
#include "collection.h"

cp_wrap *cp_wrap_new(void *item, cp_destructor_fn dtr)
{
/*
	cp_wrap *wrap = calloc(1, sizeof(cp_wrap));
	if (wrap)
	{
		wrap->item = item;
		wrap->dtr = dtr;
	}

	return wrap;
*/
}

void cp_wrap_delete(cp_wrap *wrap)
{
	if (wrap)
	{
		if (wrap->dtr)
			(*wrap->dtr)(wrap->item);

		free(wrap);
	}
}

#ifdef _WINDOWS
BOOL APIENTRY DllMain( HANDLE hModule, 
                       DWORD  ul_reason_for_call, 
                       LPVOID lpReserved
					 )
{
    switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
		case DLL_THREAD_ATTACH:
		case DLL_THREAD_DETACH:
		case DLL_PROCESS_DETACH:
			break;
    }
    return TRUE;
}

/* WIN32 implementation of cp_mutex_init */
int cp_mutex_init(cp_mutex *mutex, void *attr)
{ 
	*mutex = CreateMutex((attr), FALSE, NULL);
	return *mutex == NULL;
}

/* WIN32 implementation of read-write locks. cp_lock is not upgradeable. */

int cp_lock_init(cp_lock *lock, void *attr)
{
	SECURITY_ATTRIBUTES sec_attr;
	sec_attr.nLength = sizeof(SECURITY_ATTRIBUTES);
	sec_attr.lpSecurityDescriptor = NULL;
	sec_attr.bInheritHandle = FALSE;

	lock->access_mutex = CreateMutex(&sec_attr, FALSE, NULL);
//	lock->write_mutex = CreateMutex(&sec_attr, FALSE, NULL);
	lock->readers = 0;
	lock->writer = 0;
	lock->writer_waiting = 0;

	return 0;
}

int cp_lock_rdlock(cp_lock *lock)
{
	while (1)
	{
		WaitForSingleObject(lock->access_mutex, INFINITE);
		if (lock->writer_waiting)
			ReleaseMutex(lock->access_mutex);
		else
			break;
	}
	lock->readers++;
	ReleaseMutex(lock->access_mutex);

	return 0;
}

int cp_lock_wrlock(cp_lock *lock)
{
	if (lock->writer == GetCurrentThreadId()) return 0;

	while (1)
	{
		WaitForSingleObject(lock->access_mutex, INFINITE);
		lock->writer_waiting = 1;
		if (lock->readers)
			ReleaseMutex(lock->access_mutex);
		else
			break;
	}

	lock->writer = GetCurrentThreadId();
	lock->writer_waiting = 0;

	return 0;
}

int cp_lock_unlock(cp_lock *lock)
{
	if (lock->writer == GetCurrentThreadId())
		lock->writer = 0;
	else
	{
		WaitForSingleObject(lock->access_mutex, INFINITE);
		lock->readers--;
	}

	ReleaseMutex(lock->access_mutex);
	return 0;
}

int cp_lock_destroy(cp_lock *lock)
{
	CloseHandle(lock->access_mutex);
	return 0;
}

/* WIN32 implementation of a basic POSIX-condition-variable-like API
 * 
 * based on "Strategies for Implementing POSIX Condition Variables on WIN32"
 * by Douglas C. Schmidt and Irfan Pyarali - 
 * see http://www.cs.wustl.edu/~schmidt/WIN32-cv-1.html
 */
int cp_cond_init(cp_cond *cv, const void *attr) // pthread_condattr_t *)
{
  cv->waiters_count_ = 0;
  cv->was_broadcast_ = 0;
  cv->sema_ = CreateSemaphore (NULL,       // no security
                               0,          // initially 0
                               0x7fffffff, // max count
                               NULL);      // unnamed 
  if (cv->sema_ == NULL) return -1;
  InitializeCriticalSection (&cv->waiters_count_lock_);
  cv->waiters_done_ = CreateEvent (NULL,  // no security
                                   FALSE, // auto-reset
                                   FALSE, // non-signaled initially
                                   NULL); // unnamed
  if (cv->waiters_done_ == NULL) return -1;

  return 0;
}

int cp_cond_destroy(cp_cond *cv)
{
	if (cv)
	{
		CloseHandle(cv->sema_);
		DeleteCriticalSection(&cv->waiters_count_lock_);
		CloseHandle(cv->waiters_done_);
		return 0;
	}

	return -1;
}

int cp_cond_wait(cp_cond *cv, cp_mutex *external_mutex)
{
	int last_waiter;
//printf(" <<< cond_wait: starting\n");
	// Avoid race conditions.
	EnterCriticalSection (&cv->waiters_count_lock_);
	cv->waiters_count_++;
	LeaveCriticalSection (&cv->waiters_count_lock_);

//printf("cond: calling SignalObjectAndWait\n");
	// This call atomically releases the mutex and waits on the
	// semaphore until <pthread_cond_signal> or <pthread_cond_broadcast>
	// are called by another thread.
	SignalObjectAndWait (*external_mutex, cv->sema_, INFINITE, FALSE);
//printf("cond: popped wait\n");

	// Reacquire lock to avoid race conditions.
	EnterCriticalSection (&cv->waiters_count_lock_);

	// We're no longer waiting...
	cv->waiters_count_--;

	// Check to see if we're the last waiter after <pthread_cond_broadcast>.
	last_waiter = cv->was_broadcast_ && cv->waiters_count_ == 0;

	LeaveCriticalSection (&cv->waiters_count_lock_);

	// If we're the last waiter thread during this particular broadcast
	// then let all the other threads proceed.
	if (last_waiter)
	{
//		printf("cond_wait: signaling waiters_done_\n");
		// This call atomically signals the <waiters_done_> event and waits until
		// it can acquire the <external_mutex>.  This is required to ensure fairness. 
		SignalObjectAndWait (cv->waiters_done_, *external_mutex, INFINITE, FALSE);
	}
	else
	{
//		printf("cond_wait: grab external_mutex\n");
		// Always regain the external mutex since that's the guarantee we
		// give to our callers. 
		WaitForSingleObject(*external_mutex, INFINITE);
	}
//printf(" >>> cond_wait: done\n");
	return 0;
}

int cp_cond_signal(cp_cond *cv)
{
	int have_waiters;

	EnterCriticalSection (&cv->waiters_count_lock_);
//printf("cp_cond_signal: %d waiters\n", cv->waiters_count_);
	have_waiters = cv->waiters_count_ > 0;
	LeaveCriticalSection (&cv->waiters_count_lock_);

	// If there aren't any waiters, then this is a no-op.  
	if (have_waiters)
		ReleaseSemaphore (cv->sema_, 1, 0);

	return 0;
}

int cp_cond_broadcast(cp_cond *cv)
{
	int have_waiters;
	// This is needed to ensure that <waiters_count_> and <was_broadcast_> are
	// consistent relative to each other.
	EnterCriticalSection (&cv->waiters_count_lock_);
	have_waiters = 0;
//printf("cp_cond_broadcast: %d waiters\n", cv->waiters_count_);
	if (cv->waiters_count_ > 0) {
		// We are broadcasting, even if there is just one waiter...
		// Record that we are broadcasting, which helps optimize
		// <pthread_cond_wait> for the non-broadcast case.
		cv->was_broadcast_ = 1;
		have_waiters = 1;
	}

	if (have_waiters) {
		// Wake up all the waiters atomically.
		ReleaseSemaphore (cv->sema_, cv->waiters_count_, 0);

		LeaveCriticalSection (&cv->waiters_count_lock_);

		// Wait for all the awakened threads to acquire the counting
		// semaphore. 
		WaitForSingleObject (cv->waiters_done_, INFINITE);
		// This assignment is okay, even without the <waiters_count_lock_> held 
		// because no other waiter threads can wake up to access it.
		cv->was_broadcast_ = 0;
	}
	else
		LeaveCriticalSection (&cv->waiters_count_lock_);

	return 0;
}

void *cp_calloc(size_t count, size_t size)
{
	return calloc(count, size);
}

void *cp_realloc(void *p, size_t size)
{
	return realloc(p, size);
}

void *cp_malloc(size_t size)
{
	return malloc(size);
}

void cp_free(void *p)
{
	free(p);
}
#endif /* _WINDOWS */
