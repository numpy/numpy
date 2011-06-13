/* -*- c -*- */

/*
 * The purpose of this module is to add faster sort functions
 * that are type-specific.  This is done by altering the
 * function table for the builtin descriptors.
 *
 * These sorting functions are copied almost directly from numarray
 * with a few modifications (complex comparisons compare the imaginary
 * part if the real parts are equal, for example), and the names
 * are changed.
 *
 * The original sorting code is due to Charles R. Harris who wrote
 * it for numarray.
 */

/*
 * Quick sort is usually the fastest, but the worst case scenario can
 * be slower than the merge and heap sorts.  The merge sort requires
 * extra memory and so for large arrays may not be useful.
 *
 * The merge sort is *stable*, meaning that equal components
 * are unmoved from their entry versions, so it can be used to
 * implement lexigraphic sorting on multiple keys.
 *
 * The heap sort is included for completeness.
 */

/*#include "sort.c"*/
#include "numpy/npy_sort.h"

static struct PyMethodDef methods[] = {
    {NULL, NULL, 0, NULL}
};


#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_sort",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

/* Initialization function for the module */
#if defined(NPY_PY3K)
PyObject *PyInit__sort(void) {
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    add_sortfuncs();
    return m;
}
#else
PyMODINIT_FUNC
init_sort(void) {
    Py_InitModule("_sort", methods);

    import_array();
    add_sortfuncs();
}
#endif
