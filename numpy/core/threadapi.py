""" API for setting/getting the parameters that determine how numpy threads
    vector operations.
"""

__all__ = ['setthreading', 'getthreading']

import umath

def setthreading(use_threads=None, thread_count=None, element_threshold=None):
    """ Settings that determine how numpy uses threads on vector operations.          
    
        use_threads -- None, leaves value unaltered.  
                       0 tells numpy not to use threads on vector operations.
                       1 numpy uses threads on vector operations if appropriate.
        thread_count -- Number of threads to use for vector operations.        
        element_threshold -- numpy will only thread vector operations that
                             have more than this number of elements in the
                             arrays.
                             
        fixme: In many cases, the thread_count and element_threshold values
               might be better as "suggestions" instead of exact values. 
               We may want to add another thread state value that allows
               numpy to pick the best thread settings.               
    """
    old_settings = umath._getthreading()
    
    settings = old_settings[:]
    if use_threads is not None:
        if use_threads:
            settings[0] = 1
        else: 
            settings[0] = 0

    if thread_count is not None:
        settings[1] = int(thread_count)

    if element_threshold is not None:
        settings[2] = int(element_threshold)

    umath._setthreading(*settings)
   
    return old_settings         
            
getthreading = umath._getthreading