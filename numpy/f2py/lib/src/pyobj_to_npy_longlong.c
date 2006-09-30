int pyobj_to_npy_longlong(PyObject *obj, npy_longlong* value) {
  PyObject* tmp = NULL;
  if (PyLong_Check(obj)) {
    *value = PyLong_AsLongLong(obj);
    return (!PyErr_Occurred());
  }
  if (PyInt_Check(obj)) {
    *value = (npy_longlong)PyInt_AS_LONG(obj);
    return 1;
  }
  tmp = PyNumber_Long(obj);
  if (tmp) {
    *value = PyLong_AsLongLong(tmp);
    Py_DECREF(tmp);
    return (!PyErr_Occurred());
  }
  if (PyComplex_Check(obj))
    tmp = PyObject_GetAttrString(obj,"real");
  else if (PyString_Check(obj))
    /*pass*/;
  else if (PySequence_Check(obj))
    tmp = PySequence_GetItem(obj,0);
  if (tmp) {
    PyErr_Clear();
    if (pyobj_to_npy_longlong(tmp, value)) {
      Py_DECREF(tmp);
      return 1;
    }
    Py_DECREF(tmp);
  }
  if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError,
		    "Failed to convert python object to C npy_longlong.");
  }
  return 0;
}

