#ifndef ufuncthreadapi_h
#define ufuncthreadapi_h

/* Data structure to hold information about thread settings. */

typedef struct {
    
    /* Should ufuncs divide their work between multiple threads */
    int use_threads;        
        
    /* Number of threads to use in threaded operations */
    int thread_count;  
    
    /* Minimum number of elements in an array before threading is used */
    int element_threshold; 

} UFuncThreadSettings;


static PyObject *
ufunc_setthreading(PyObject *dummy, PyObject *args);

#endif
