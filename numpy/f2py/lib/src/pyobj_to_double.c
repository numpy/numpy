int pyobj_to_double(PyObject* obj, double* value) {
  PyObject* tmp = NULL;
  if (PyFloat_Check(obj)) {
#ifdef __sgi
    *value = PyFloat_AsDouble(obj);
    return (!PyErr_Occurred());
#else
    *value = PyFloat_AS_DOUBLE(obj);
    return 1;
#endif
  }
  tmp = PyNumber_Float(obj);
  if (tmp) {
#ifdef __sgi
    *value = PyFloat_AsDouble(tmp);
    Py_DECREF(tmp);
    return (!PyErr_Occurred());
#else
    *value = PyFloat_AS_DOUBLE(tmp);
    Py_DECREF(tmp);
    return 1;
#endif
  }
  if (PyComplex_Check(obj))
    tmp = PyObject_GetAttrString(obj,"real");
  else if (PyString_Check(obj))
    /*pass*/;
  else if (PySequence_Check(obj))
    tmp = PySequence_GetItem(obj,0);
  if (tmp) {
    PyErr_Clear();
    if (pyobj_to_double(tmp, value)) {Py_DECREF(tmp); return 1;}
    Py_DECREF(tmp);
  }
  if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "Failed to convert python object to C double.");
  }
  return 0;
}
