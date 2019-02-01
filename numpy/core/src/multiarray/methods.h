#ifndef _NPY_ARRAY_METHODS_H_
#define _NPY_ARRAY_METHODS_H_

extern NPY_NO_EXPORT PyMethodDef array_methods[];

NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting);

/*
 * Pathlib support.
 * For Python >= 3.6, use the os.Pathlike interface.
 * Else, for Python >= 3.4, use file = str(file) if file is a PurePath
 * For older Python, do nothing.
 */
static inline PyObject *
NpyPath_PathlikeToFspath(PyObject *file)
{
#if PY_VERSION_HEX >= 0x03060000  /* os.pathlike arrived in 3.6 */
    if (PyObject_HasAttrString(file, "__fspath__")) {
        file = PyOS_FSPath(file);
    }
    return file;
#elif PY_VERSION_HEX >= 0x03040000 /* pathlib arrived in 3.4 */
    PyObject *pathlib, *pathlib_PurePath;
    int fileIsPurePath;

    pathlib = PyImport_ImportModule("pathlib");
    if (!pathlib) {
        return NULL;
    }
    pathlib_PurePath = PyObject_GetAttrString(pathlib, "PurePath");
    fileIsPurePath = PyObject_IsInstance(file,  pathlib_PurePath);
    Py_XDECREF(pathlib);
    Py_XDECREF(pathlib_PurePath);
    if (fileIsPurePath)  {
        file = PyObject_Str(file);
    }
#endif
    return file;
}

#endif
