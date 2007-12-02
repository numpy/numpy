/*-*- C -*-*/

/**********************************************************************/

/* For numpy versions prior to 1.0, the names of certain data types
 * are different than in later versions.  This fragment provides macro
 * substitutions that allow us to support old and new versions of
 * numpy.
 */

%fragment("NumPy_Backward_Compatibility", "header")
{
/* Support older NumPy data type names
 */
%#if NDARRAY_VERSION < 0x01000000
%#define NPY_BOOL          PyArray_BOOL
%#define NPY_BYTE          PyArray_BYTE
%#define NPY_UBYTE         PyArray_UBYTE
%#define NPY_SHORT         PyArray_SHORT
%#define NPY_USHORT        PyArray_USHORT
%#define NPY_INT           PyArray_INT
%#define NPY_UINT          PyArray_UINT
%#define NPY_LONG          PyArray_LONG
%#define NPY_ULONG         PyArray_ULONG
%#define NPY_LONGLONG      PyArray_LONGLONG
%#define NPY_ULONGLONG     PyArray_ULONGLONG
%#define NPY_FLOAT         PyArray_FLOAT
%#define NPY_DOUBLE        PyArray_DOUBLE
%#define NPY_LONGDOUBLE    PyArray_LONGDOUBLE
%#define NPY_CFLOAT        PyArray_CFLOAT
%#define NPY_CDOUBLE       PyArray_CDOUBLE
%#define NPY_CLONGDOUBLE   PyArray_CLONGDOUBLE
%#define NPY_OBJECT        PyArray_OBJECT
%#define NPY_STRING        PyArray_STRING
%#define NPY_UNICODE       PyArray_UNICODE
%#define NPY_VOID          PyArray_VOID
%#define NPY_NTYPES        PyArray_NTYPES
%#define NPY_NOTYPE        PyArray_NOTYPE
%#define NPY_CHAR          PyArray_CHAR
%#define NPY_USERDEF       PyArray_USERDEF
%#define npy_intp          intp

%#define NPY_MAX_BYTE      MAX_BYTE
%#define NPY_MIN_BYTE      MIN_BYTE
%#define NPY_MAX_UBYTE     MAX_UBYTE
%#define NPY_MAX_SHORT     MAX_SHORT
%#define NPY_MIN_SHORT     MIN_SHORT
%#define NPY_MAX_USHORT    MAX_USHORT
%#define NPY_MAX_INT       MAX_INT
%#define NPY_MIN_INT       MIN_INT
%#define NPY_MAX_UINT      MAX_UINT
%#define NPY_MAX_LONG      MAX_LONG
%#define NPY_MIN_LONG      MIN_LONG
%#define NPY_MAX_ULONG     MAX_ULONG
%#define NPY_MAX_LONGLONG  MAX_LONGLONG
%#define NPY_MIN_LONGLONG  MIN_LONGLONG
%#define NPY_MAX_ULONGLONG MAX_ULONGLONG
%#define NPY_MAX_INTP      MAX_INTP
%#define NPY_MIN_INTP      MIN_INTP

%#define NPY_FARRAY        FARRAY
%#define NPY_F_CONTIGUOUS  F_CONTIGUOUS
%#endif
}

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
    static PyArray_Descr * longDescr = PyArray_DescrNewFromType(NPY_LONG);
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
    PyArray_CastScalarToCtype(obj, (void*)val, longDescr);
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
    static PyArray_Descr * ulongDescr = PyArray_DescrNewFromType(NPY_ULONG);
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
    if (!PyArray_IsScalar(obj,Integer)) return SWIG_TypeError;
    PyArray_CastScalarToCtype(obj, (void*)val, ulongDescr);
    return SWIG_OK;
  }
}
