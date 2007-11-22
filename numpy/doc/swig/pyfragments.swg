// Override the SWIG_AsVal_frag(long) fragment so that it also checks
// for numpy scalar array types.  The code through the %#endif is
// essentially cut-and-paste from pyprimtype.swg

%fragment(SWIG_AsVal_frag(long), "header",
	  fragment="SWIG_CanCastAsInteger")
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
    if (!PyArray_CheckScalar(obj)) return SWIG_TypeError;
    int dataType = PyArray_TYPE(obj);
    switch (dataType)
    {
      case NPY_BYTE:
      {
	char data = *((char*)PyArray_DATA(obj));
	if (val) *val = %numeric_cast(data, long);
	break;
      }
      case NPY_UBYTE:
      {
	unsigned char data = *((unsigned char*)PyArray_DATA(obj));
	if (val) *val = %numeric_cast(data, long);
	break;
      }
      case NPY_SHORT:
      {
	short data = *((short*)PyArray_DATA(obj));
	if (val) *val = %numeric_cast(data, long);
	break;
      }
      case NPY_USHORT:
      {
	unsigned short data = *((unsigned short*)PyArray_DATA(obj));
	if (val) *val = %numeric_cast(data, long);
	break;
      }
      case NPY_INT:
      {
	int data = *((int*)PyArray_DATA(obj));
	if (val) *val = %numeric_cast(data, long);
	break;
      }
      case NPY_UINT:
      {
	unsigned int data = *((unsigned int*)PyArray_DATA(obj));
	if (data > NPY_MAX_LONG) return SWIG_OverflowError;
	if (val) *val = %numeric_cast(data, long);
	break;
      }
      case NPY_LONG:
      {
	*val = *((long*)PyArray_DATA(obj));
	break;
      }
      case NPY_ULONG:
      {
	unsigned long data = *((unsigned long*)PyArray_DATA(obj));
	if (data > NPY_MAX_LONG) return SWIG_OverflowError;
	if (val) *val = %numeric_cast(data, long);
	break;
      }
      default:
	return SWIG_TypeError;
    }
    return SWIG_OK;
  }
}




// Override the SWIG_AsVal_frag(unsigned long) fragment so that it
// also checks for numpy scalar array types.  The code through the
// %#endif is essentially cut-and-paste from pyprimtype.swg

%fragment(SWIG_AsVal_frag(unsigned long),"header",
	  fragment="SWIG_CanCastAsInteger")
{
  SWIGINTERN int
  SWIG_AsVal_dec(unsigned long)(PyObject *obj, unsigned long *val) 
  {
    if (PyInt_Check(obj)) {
      long v = PyInt_AsLong(obj);
      if (v >= 0) {
	if (val) *val = v;
	return SWIG_OK;
      } else {
	return SWIG_OverflowError;
      }
    } else if (PyLong_Check(obj)) {
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
    if (!PyArray_CheckScalar(obj)) return SWIG_TypeError;
    int dataType = PyArray_TYPE(obj);
    switch (dataType)
    {
      case NPY_BYTE:
      {
	char data = *((char*)PyArray_DATA(obj));
	if (data < 0) return SWIG_OverflowError;
	if (val) *val = %numeric_cast(data, unsigned long);
	break;
      }
      case NPY_UBYTE:
      {
	unsigned char data = *((unsigned char*)PyArray_DATA(obj));
	if (val) *val = %numeric_cast(data, unsigned long);
	break;
      }
      case NPY_SHORT:
      {
	short data = *((short*)PyArray_DATA(obj));
	if (data < 0) return SWIG_OverflowError;
	if (val) *val = %numeric_cast(data, unsigned long);
	break;
      }
      case NPY_USHORT:
      {
	unsigned short data = *((unsigned short*)PyArray_DATA(obj));
	if (val) *val = %numeric_cast(data, unsigned long);
	break;
      }
      case NPY_INT:
      {
	int data = *((int*)PyArray_DATA(obj));
	if (data < 0) return SWIG_OverflowError;
	if (val) *val = %numeric_cast(data, unsigned long);
	break;
      }
      case NPY_UINT:
      {
	unsigned int data = *((unsigned int*)PyArray_DATA(obj));
	if (val) *val = %numeric_cast(data, unsigned long);
	break;
      }
      case NPY_LONG:
      {
	long data = *((long*)PyArray_DATA(obj));
	if (data < 0) return SWIG_OverflowError;
	if (val) *val = %numeric_cast(data, unsigned long);
	break;
      }
      case NPY_ULONG:
      {
	*val = *((unsigned long*)PyArray_DATA(obj));
	break;
      }
      default:
	return SWIG_TypeError;
    }
    return SWIG_OK;
  }
}
