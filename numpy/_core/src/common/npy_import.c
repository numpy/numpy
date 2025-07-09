#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "numpy/ndarraytypes.h"
#include "npy_import.h"
#include "npy_atomic.h"


NPY_VISIBILITY_HIDDEN npy_runtime_imports_struct npy_runtime_imports;

NPY_NO_EXPORT int
init_import_mutex(void) {
#if PY_VERSION_HEX < 0x30d00b3
    npy_runtime_imports.import_mutex = PyThread_allocate_lock();
    if (npy_runtime_imports.import_mutex == NULL) {
        PyErr_NoMemory();
        return -1;
    }
#endif
    return 0;
}
