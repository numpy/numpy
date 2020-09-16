#ifndef _NPY_DTYPEMETA_H
#define _NPY_DTYPEMETA_H

#define NPY_DTYPE(descr) ((PyArray_DTypeMeta *)Py_TYPE(descr))

NPY_NO_EXPORT int
dtypemeta_wrap_legacy_descriptor(PyArray_Descr *dtypem);

#endif  /*_NPY_DTYPEMETA_H */
