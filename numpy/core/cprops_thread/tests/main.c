#include <time.h>
#include <unistd.h>

pid_t getpid(void);

#include "thread.h"


void* worker_task(void* blah)
{
    pid_t pid;
    pid = getpid();
// fixme: what is up with sleep?  How do we get the unix style sleep?    
#ifdef _WINDOWS
    _sleep(1000);
#else
    sleep(1);
#endif
    printf("hello from: %d\n", pid);
}

int main()
{
    time_t t1, t2;
    float seconds;
    pid_t pid;
    int i;
    cp_thread_pool* thread_pool;
    cp_thread* thread;
        
    thread_pool = cp_thread_pool_create(4, 4);
    
    pid = getpid();
    printf("main thread id: %d\n", pid);
    
    t1 = clock();
    
    for (i=0;i<8;i++)
    {
        thread = cp_thread_pool_get(thread_pool, worker_task, NULL);
        printf("task %d started on thread %d.\n", i+1, thread);
    }
    
    /* wait for all threads to finish executing */
    cp_thread_pool_wait(thread_pool);
    
    t2 = clock();
    
    seconds = ((float)(t2-t1))/CLOCKS_PER_SEC;
    printf("runtime: %d %d %d\n", t1, t2, CLOCKS_PER_SEC);
	return 1;
}
