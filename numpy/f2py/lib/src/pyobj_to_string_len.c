int pyobj_to_string_len(PyObject* obj, f2py_string* value, size_t length) {
  if (PyString_Check(obj)) {
    if (strncpy((char*)value,PyString_AS_STRING(obj), length))
      return 1;
  }
  if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError,
		    "Failed to convert python object to C f2py_string.");
  }
  return 0;
}
