int pyobj_to_double(PyObject* obj, double* value) {
  int return_value = 0;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_double(type=%s)\n",PyString_AS_STRING(PyObject_Repr(PyObject_Type(obj))));
#endif
  PyObject* tmp = NULL;
  if (PyFloat_Check(obj)) {
#ifdef __sgi
    *value = PyFloat_AsDouble(obj);
    return_value = !PyErr_Occurred();
    goto capi_done;
#else
    *value = PyFloat_AS_DOUBLE(obj);
    return_value = 1;
    goto capi_done;
#endif
  }
  tmp = PyNumber_Float(obj);
  if (tmp) {
#ifdef __sgi
    *value = PyFloat_AsDouble(tmp);
    Py_DECREF(tmp);
    return_value = !PyErr_Occurred();
    goto capi_done;
#else
    *value = PyFloat_AS_DOUBLE(tmp);
    Py_DECREF(tmp);
    return_value = 1;
    goto capi_done;
#endif
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
    if (pyobj_to_double(tmp, value)) {
      Py_DECREF(tmp); 
      return_value = 1;
      goto capi_done;
    }
    Py_DECREF(tmp);
  }
  if (!PyErr_Occurred()) {
    PyObject* r = PyString_FromString("Failed to convert ");
    PyString_ConcatAndDel(&r, PyObject_Repr(PyObject_Type(obj)));
    PyString_ConcatAndDel(&r, PyString_FromString(" to C double"));
    PyErr_SetObject(PyExc_TypeError,r);
  }
 capi_done:
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_double: return_value=%d, PyErr_Occurred()=%p\n", return_value, PyErr_Occurred());
#endif
  return return_value;
}
