/*-*- C -*-*/

/**********************************************************************/

/* For numpy versions prior to 1.0, the names of certain data types
 * are different than in later versions.  This fragment provides macro
 * substitutions that allow us to support old and new versions of
 * numpy.
 */

/**********************************************************************/

/* Override the SWIG_AsVal_frag(long) fragment so that it also checks
 * for numpy scalar array types.  The code through the %#endif is
 * essentially cut-and-paste from pyprimtype.swg
 */

%fragment(SWIG_AsVal_frag(long), "header",
	  fragment="SWIG_CanCastAsInteger",
          fragment="NumPy_Backward_Compatibility")
{
  SWIGINTERN int
  SWIG_AsVal_dec(long)(PyObject * obj, long * val)
  {
    if (PyInt_Check(obj)) {
      if (val) *val = PyInt_AsLong(obj);
      return SWIG_OK;
    } else if (PyLong_Check(obj)) {
      long v = PyLong_AsLong(obj);
      if (!PyErr_Occurred()) {
	if (val) *val = v;
	return SWIG_OK;
      } else {
	PyErr_Clear();
      }
    }
%#ifdef SWIG_PYTHON_CAST_MODE
    {
      int dispatch = 0;
      long v = PyInt_AsLong(obj);
      if (!PyErr_Occurred()) {
	if (val) *val = v;
	return SWIG_AddCast(SWIG_OK);
      } else {
	PyErr_Clear();
      }
      if (!dispatch) {
	double d;
	int res = SWIG_AddCast(SWIG_AsVal(double)(obj,&d));
	if (SWIG_IsOK(res) && SWIG_CanCastAsInteger(&d, LONG_MIN, LONG_MAX)) {
	  if (val) *val = (long)(d);
	  return res;
	}
      }
    }
%#endif
    if (!PyArray_IsScalar(obj,Integer)) return SWIG_TypeError;
    PyArray_Descr * longDescr = PyArray_DescrNewFromType(NPY_LONG);
    PyArray_CastScalarToCtype(obj, (void*)val, longDescr);
    Py_DECREF(longDescr);
    return SWIG_OK;
  }
}


/* Override the SWIG_AsVal_frag(unsigned long) fragment so that it
 * also checks for numpy scalar array types.  The code through the
 * %#endif is essentially cut-and-paste from pyprimtype.swg
 */

%fragment(SWIG_AsVal_frag(unsigned long),"header",
	  fragment="SWIG_CanCastAsInteger",
          fragment="NumPy_Backward_Compatibility")
{
  SWIGINTERN int
  SWIG_AsVal_dec(unsigned long)(PyObject *obj, unsigned long *val)
  {
    %#if PY_VERSION_HEX < 0x03000000
    if (PyInt_Check(obj)) 
    {
      long v = PyInt_AsLong(obj);
      if (v >= 0) 
      {
        if (val) *val = v;
	    return SWIG_OK;
      } 
      else 
      {
	    return SWIG_OverflowError;
      }
    } else 
    %#endif
    if (PyLong_Check(obj)) {
      unsigned long v = PyLong_AsUnsignedLong(obj);
      if (!PyErr_Occurred()) {
	if (val) *val = v;
	return SWIG_OK;
      } else {
	PyErr_Clear();
      }
    }
%#ifdef SWIG_PYTHON_CAST_MODE
    {
      int dispatch = 0;
      unsigned long v = PyLong_AsUnsignedLong(obj);
      if (!PyErr_Occurred()) {
	if (val) *val = v;
	return SWIG_AddCast(SWIG_OK);
      } else {
	PyErr_Clear();
      }
      if (!dispatch) {
	double d;
	int res = SWIG_AddCast(SWIG_AsVal(double)(obj,&d));
	if (SWIG_IsOK(res) && SWIG_CanCastAsInteger(&d, 0, ULONG_MAX)) {
	  if (val) *val = (unsigned long)(d);
	  return res;
	}
      }
    }
%#endif
    if (!PyArray_IsScalar(obj,Integer)) return SWIG_TypeError;
    PyArray_Descr * ulongDescr = PyArray_DescrNewFromType(NPY_ULONG);
    PyArray_CastScalarToCtype(obj, (void*)val, ulongDescr);
    Py_DECREF(ulongDescr);
    return SWIG_OK;
  }
}
