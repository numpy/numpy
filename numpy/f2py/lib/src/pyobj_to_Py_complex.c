int pyobj_to_Py_complex(PyObject* obj, Py_complex* value) {
  int return_value = 0;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_Py_complex(type=%s)\n",PyString_AS_STRING(PyObject_Repr(PyObject_Type(obj))));
#endif
  if (PyArray_CheckScalar(obj)) {
    if (PyArray_IsScalar(obj, CDouble)) {
      npy_cdouble obval = PyArrayScalar_VAL(obj, CDouble);
      value->real = obval.real;
      value->imag = obval.imag;
      return_value = 1;
      goto capi_done;
    }
    else if (PyArray_IsScalar(obj, CFloat)) {
      npy_cfloat obval = PyArrayScalar_VAL(obj, CFloat);
      value->real = obval.real;
      value->imag = obval.imag;
      return_value = 1;
      goto capi_done;
    }
  }
  if (PyComplex_Check(obj)) {
    *value = PyComplex_AsCComplex(obj);
    return_value = 1;
    goto capi_done;
  }
  /* Python does not provide PyNumber_Complex function :-( */
  value->imag=0.0;
  if (PyFloat_Check(obj)) {
#ifdef __sgi
    value->real = PyFloat_AsDouble(obj);
    return_value = !PyErr_Occurred()
#else
    value->real = PyFloat_AS_DOUBLE(obj);
    return_value = 1;
#endif
    goto capi_done;
  }
  if (PyInt_Check(obj)) {
    value->real = (double)PyInt_AS_LONG(obj);
    return_value = 1;
    goto capi_done;
  }
  if (PyLong_Check(obj)) {
    value->real = PyLong_AsDouble(obj);
    return_value = !PyErr_Occurred();;
    goto capi_done;
  }
  if (PyString_Check(obj))
    /*pass*/;
  else if (PySequence_Check(obj) && PySequence_Size(obj)==1) {
    PyObject *tmp = PySequence_GetItem(obj,0);
    if (tmp) {
      if (pyobj_to_Py_complex(tmp,value)) {
        Py_DECREF(tmp);
	return_value = 1;
	goto capi_done;
      }
      Py_DECREF(tmp);
    }
  }
  if (!PyErr_Occurred()) {
    PyObject* r = PyString_FromString("Failed to convert ");
    PyString_ConcatAndDel(&r, PyObject_Repr(PyObject_Type(obj)));
    PyString_ConcatAndDel(&r, PyString_FromString(" to C Py_complex"));
    PyErr_SetObject(PyExc_TypeError,r);
  }
 capi_done:
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_Py_complex: return_value=%d, PyErr_Occurred()=%p\n", return_value, PyErr_Occurred());
#endif
  return return_value;
}
