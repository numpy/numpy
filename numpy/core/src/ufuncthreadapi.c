#include <Python.h>
#include "numpy/ufuncthreadapi.h"

#ifdef WIN32
    /* fixme: npy_byte conflicts with a definition in the windows headers.
    */
    #undef npy_byte
    #include <windows.h>
    
#endif
    
/* Global object that stores how ufunc operations are threaded.
   fixme: This is potentially a bad idea.  At the minimum, we should
          store this on a per-thread basis.
*/

UFuncThreadSettings thread_settings;

/* Determine the number of processors on this machine. 
   Return 1 if it is not possible to determine the value.
*/

int number_of_processors()
{
   /* default the number of processors to 1.*/
   int number_of_processors=1;
   
#ifdef WIN32
    /* determine the number of processors */   
    SYSTEM_INFO siSysInfo;
    GetSystemInfo(&siSysInfo); 
    number_of_processors = (int) siSysInfo.dwNumberOfProcessors; 
#endif

    return number_of_processors;
}

void init_thread_settings(UFuncThreadSettings* settings)
{
   /* default the number of processors to 1.*/
   int processor_count=number_of_processors();

    /* default to using one thread per processor.
       fixme: We probably want to throttle this back on machines with
              more than 4 processors unless we figure out better 
              scaling.
    */
    settings->thread_count = processor_count; 
    settings->use_threads = processor_count > 1;
    settings->element_threshold=20000;
}

/* Python UFunc Threading API */

/* Set the parameters that control how ufuncs operations are threaded. */
static char
doc_setthreading[] = "_setthreading(use_threads, thread_count, element_threshold) determines how numpy threads vector operations.\nIt returns a list of the old settings.\nSee setthreading() for more information on these parameters.";

static PyObject *
ufunc_setthreading(PyObject *dummy, PyObject *args)
{

	int use_threads, thread_count, element_threshold;

	if (!PyArg_ParseTuple(args, "iii", &use_threads, &thread_count, &element_threshold)) 
	    return NULL;

	thread_settings.use_threads = use_threads;
	thread_settings.thread_count = thread_count;
	thread_settings.element_threshold = element_threshold;

	Py_INCREF(Py_None);
	return Py_None;
}

static char
doc_getthreading[] = "_getthreading() returns a list of the form [use_threads, thread_count, element_threshold].\nSee setthreading() for more information on these parameters.";

static PyObject *
ufunc_getthreading(PyObject *dummy, PyObject *args)
{
	PyObject *res;

	/* Construct list of defaults */
	res = PyList_New(3);
	if (res == NULL) 
	    return NULL;
	    
	PyList_SET_ITEM(res, 0, PyInt_FromLong(thread_settings.use_threads));
	PyList_SET_ITEM(res, 1, PyInt_FromLong(thread_settings.thread_count));
	PyList_SET_ITEM(res, 2, PyInt_FromLong(thread_settings.element_threshold));

	return res;

}
