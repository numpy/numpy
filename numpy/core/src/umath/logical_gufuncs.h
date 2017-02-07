typedef PyObject* (*PyUFunc_FromFuncAndDataAndSignature_t)(PyUFuncGenericFunction*,
                                                           void**,
                                                           char*,
                                                           int,
                                                           int,
                                                           int,
                                                           int,
                                                           const char*,
                                                           const char*,
                                                           int,
                                                           const char*);

void InitLogicalGufuncs(PyObject *dictionary, PyUFunc_FromFuncAndDataAndSignature_t createPyUFunc);
