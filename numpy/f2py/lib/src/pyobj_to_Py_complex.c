int pyobj_to_Py_complex(PyObject* obj, Py_complex* value) {
  if (PyComplex_Check(obj)) {
    *value =PyComplex_AsCComplex(obj);
    return 1;
  }
  /* Python does not provide PyNumber_Complex function :-( */
  (*value).imag=0.0;
  if (PyFloat_Check(obj)) {
#ifdef __sgi
    (*value).real = PyFloat_AsDouble(obj);
    return (!PyErr_Occurred());
#else
    (*value).real = PyFloat_AS_DOUBLE(obj);
    return 1;
#endif
  }
  if (PyInt_Check(obj)) {
    (*value).real = (double)PyInt_AS_LONG(obj);
    return 1;
  }
  if (PyLong_Check(obj)) {
    (*value).real = PyLong_AsDouble(obj);
    return (!PyErr_Occurred());
  }
  if (PySequence_Check(obj) && (!PyString_Check(obj))) {
    PyObject *tmp = PySequence_GetItem(obj,0);
    if (tmp) {
      if (pyobj_to_Py_complex(tmp,value)) {
        Py_DECREF(tmp);
        return 1;
      }
      Py_DECREF(tmp);
    }
  }
  if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, "Failed to convert python object to C Py_complex.");
  }
  return 0;
}
