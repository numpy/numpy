int pyobj_to_long(PyObject *obj, long* value) {
  PyObject* tmp = NULL;
  if (PyInt_Check(obj)) {
    *value = PyInt_AS_LONG(obj);
    return 1;
  }
  tmp = PyNumber_Int(obj);
  if (tmp) {
    *value = PyInt_AS_LONG(tmp);
    Py_DECREF(tmp);
    return 1;
  }
  if (PyComplex_Check(obj))
    tmp = PyObject_GetAttrString(obj,"real");
  else if (PyString_Check(obj))
    /*pass*/;
  else if (PySequence_Check(obj))
    tmp = PySequence_GetItem(obj,0);
  if (tmp) {
    PyErr_Clear();
    if (pyobj_to_long(tmp, value)) {
      Py_DECREF(tmp);
      return 1;
    }
    Py_DECREF(tmp);
  }
  if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError,
                    "Failed to convert python object to C long.");
  }
  return 0;
}
