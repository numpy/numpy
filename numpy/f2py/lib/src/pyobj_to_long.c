int pyobj_to_long(PyObject *obj, long* value) {
  int return_value = 0;
  PyObject* tmp = NULL;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_long(type=%s)\n",PyString_AS_STRING(PyObject_Repr(PyObject_Type(obj))));
#endif
  if (PyInt_Check(obj)) {
    *value = PyInt_AS_LONG(obj);
    return_value = 1;
    goto capi_done;
  }
  tmp = PyNumber_Int(obj);
  if (tmp) {
    *value = PyInt_AS_LONG(tmp);
    Py_DECREF(tmp);
    return_value = 1;
    goto capi_done;
  }
  /*
  if (PyComplex_Check(obj))
    tmp = PyObject_GetAttrString(obj,"real");
  else
  */ 
  if (PyString_Check(obj))
    /*pass*/;
  else if (PySequence_Check(obj) && PySequence_Size(obj)==1)
    tmp = PySequence_GetItem(obj,0);
  if (tmp) {
    PyErr_Clear();
    if (pyobj_to_long(tmp, value)) {
      Py_DECREF(tmp);
      return_value = 1;
      goto capi_done;
    }
    Py_DECREF(tmp);
  }
  if (!PyErr_Occurred()) {
    PyObject* r = PyString_FromString("Failed to convert ");
    PyString_ConcatAndDel(&r, PyObject_Repr(PyObject_Type(obj)));
    PyString_ConcatAndDel(&r, PyString_FromString(" to C long"));
    PyErr_SetObject(PyExc_TypeError,r);
  }
 capi_done:
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_long: return_value=%d, PyErr_Occurred()=%p\n", return_value, PyErr_Occurred());
#endif
  return return_value;
}
