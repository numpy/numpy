/*
 * NumPy C-Level Tracing Hooks for Instrumented Code
 */

#include "npy_config.h"

#ifdef NUMPY_DEBUG_CTRACE

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static void (*real_enter)(void *, void *) = NULL;
static void (*real_exit)(void *, void *) = NULL;
static int hooks_resolved = 0;
static int hooks_failed = 0;
static int resolve_attempted = 0;

__attribute__((no_instrument_function))
static void resolve_hooks(void) {
    if (hooks_resolved || hooks_failed) return;
    if (resolve_attempted) return;
    resolve_attempted = 1;
    
    /* Try RTLD_DEFAULT first */
    real_enter = (void (*)(void *, void *))dlsym(RTLD_DEFAULT, "npy_ctrace_hook_enter");
    real_exit = (void (*)(void *, void *))dlsym(RTLD_DEFAULT, "npy_ctrace_hook_exit");
    
    if (real_enter && real_exit) {
        hooks_resolved = 1;
        return;
    }
    
    /* On macOS, try to find _ctrace_impl library and load symbols from it */
    Dl_info info;
    if (dladdr((void*)resolve_hooks, &info) && info.dli_fname) {
        /* Get the directory of this library */
        char path[1024];
        strncpy(path, info.dli_fname, sizeof(path) - 1);
        path[sizeof(path) - 1] = '\0';
        
        /* Find last slash and replace filename */
        char *last_slash = strrchr(path, '/');
        if (last_slash) {
            /* Try to find _ctrace_impl in the same directory */
            strcpy(last_slash + 1, "_ctrace_impl.cpython-313-darwin.so");
            
            void *handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
            if (handle) {
                real_enter = (void (*)(void *, void *))dlsym(handle, "npy_ctrace_hook_enter");
                real_exit = (void (*)(void *, void *))dlsym(handle, "npy_ctrace_hook_exit");
                
                if (real_enter && real_exit) {
                    hooks_resolved = 1;
                    return;
                }
            }
        }
    }
    
    hooks_failed = 1;
}

__attribute__((no_instrument_function))
void __cyg_profile_func_enter(void *func, void *caller) {
    resolve_hooks();
    if (real_enter) real_enter(func, caller);
}

__attribute__((no_instrument_function))
void __cyg_profile_func_exit(void *func, void *caller) {
    resolve_hooks();
    if (real_exit) real_exit(func, caller);
}

#endif
