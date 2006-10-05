"""
Generate
  int pyobj_to_<ctype>(PyObject* obj, <ctype>* value)
  PyObject* pyobj_from_<stype>(<ctype>* value)
functions.
"""
__all__ = ['pyobj_to_npy_scalar','pyobj_to_f2py_string','pyobj_from_npy_scalar']

from parser.api import CHAR_BIT

def pyobj_from_npy_int(ctype):
    dtype = ctype.upper()
    cls = 'Int'+ctype[7:]
    return '''\
/* depends: SCALARS_IN_BITS2.cpp */
static PyObject* pyobj_from_%(ctype)s(%(ctype)s* value) {
  PyObject* obj = PyArrayScalar_New(%(cls)s);
  if (obj==NULL) /* TODO: set exception */ return NULL;
  PyArrayScalar_ASSIGN(obj,%(cls)s,*value);
  return obj;
}
''' % (locals())

def pyobj_from_npy_float(ctype):
    dtype = ctype.upper()
    cls = 'Float'+ctype[9:]
    return '''\
/* depends: SCALARS_IN_BITS2.cpp */
static PyObject* pyobj_from_%(ctype)s(%(ctype)s* value) {
  PyObject* obj = PyArrayScalar_New(%(cls)s);
  if (obj==NULL) /* TODO: set exception */ return NULL;
  PyArrayScalar_ASSIGN(obj,%(cls)s,*value);
  return obj;
}
''' % (locals())

def pyobj_from_npy_complex(ctype):
    dtype = ctype.upper()
    cls = 'Complex'+ctype[11:]
    return '''\
/* depends: SCALARS_IN_BITS2.cpp */
static PyObject* pyobj_from_%(ctype)s(%(ctype)s* value) {
  PyObject* obj = PyArrayScalar_New(%(cls)s);
  if (obj==NULL) /* TODO: set exception */ return NULL;
  PyArrayScalar_ASSIGN(obj,%(cls)s,*value);
  return obj;
}
''' % (locals())

def pyobj_from_f2py_type(ctype):
    ctype_bits = int(ctype[10:])
    raise NotImplementedError,`ctype`
    return '''\
static PyObject* pyobj_from_%(ctype)s(%(ctype)s* value) {
  fprintf(stderr,"In pyobj_from_%(ctype)s (%%p)\\n", value);
}
'''

def pyobj_to_npy_int(ctype):
    ctype_bits = int(ctype[7:])
    return '''
/* depends: pyobj_to_long.c, pyobj_to_npy_longlong.c */
#if NPY_BITSOF_LONG == %(ctype_bits)s
#define pyobj_to_%(ctype)s pyobj_to_long
#else
#if NPY_BITSOF_LONG > %(ctype_bits)s
static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  long tmp;
  if (pyobj_to_long(obj,&tmp)) {
    *value = (%(ctype)s)tmp;
    return 1;
  }
  return 0;
}
#else
static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  npy_longlong tmp;
  if (pyobj_to_npy_longlong(obj,&tmp)) {
    *value = (%(ctype)s)tmp;
    return 1;
  }
  return 0;
}
#endif
#endif
''' % (locals())

def pyobj_to_npy_float(ctype):
    ctype_bits = int(ctype[9:])
    return '''
/* depends: pyobj_to_double.c */
#if NPY_BITSOF_DOUBLE == %(ctype_bits)s
#define pyobj_to_%(ctype)s pyobj_to_double
#else
#if NPY_BITSOF_DOUBLE > %(ctype_bits)s
static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  double tmp;
  if (pyobj_to_double(obj,&tmp)) {
    *value = (%(ctype)s)tmp;
    return 1;
  }
  return 0;
}
#else
#error, "NOTIMPLEMENTED pyobj_to_%(ctype)s"
#endif
#endif
''' % (locals())

def pyobj_to_npy_complex(ctype):
    ctype_bits = int(ctype[11:])
    cfloat_bits = ctype_bits/2
    return '''
/* depends: pyobj_to_Py_complex.c */
#if NPY_BITSOF_DOUBLE >= %(cfloat_bits)s
static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  int return_value = 0;
  Py_complex c;
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_%(ctype)s(type=%%s)\\n",PyString_AS_STRING(PyObject_Repr(PyObject_Type(obj))));
#endif
  if (pyobj_to_Py_complex(obj,&c)) {
    (*value).real = (npy_float%(cfloat_bits)s)c.real;
    (*value).imag = (npy_float%(cfloat_bits)s)c.imag;
    return_value = 1;
  }
#if defined(F2PY_DEBUG_PYOBJ_TOFROM)
  fprintf(stderr,"pyobj_to_%(ctype)s: return_value=%%d, PyErr_Occurred()=%%p\\n", return_value, PyErr_Occurred());
#endif
  return return_value;
}
#else
#error, "NOTIMPLEMENTED pyobj_to_%(ctype)s"
#endif
''' % (locals())

def pyobj_to_npy_scalar(ctype):
    if ctype.startswith('npy_int'):
        return dict(c_code=pyobj_to_npy_int(ctype))
    elif ctype.startswith('npy_float'):
        return dict(c_code=pyobj_to_npy_float(ctype))
    elif ctype.startswith('npy_complex'):
        return dict(c_code=pyobj_to_npy_complex(ctype))
    raise NotImplementedError,`ctype`

def pyobj_to_f2py_string(ctype):
    ctype_bits = int(ctype[11:])
    ctype_bytes = ctype_bits / CHAR_BIT
    return dict(
        c_code = '''
/* depends: pyobj_to_string_len.c */
static int pyobj_to_%(ctype)s(PyObject *obj, %(ctype)s* value) {
  return pyobj_to_string_len(obj, (f2py_string*)value, %(ctype_bytes)s);
}
''' % (locals()),
        typedef = ['typedef char * f2py_string;',
                         'typedef struct { char data[%(ctype_bytes)s]; } %(ctype); ' % (locals())],
        header = ['#include <string.h>'],
        )

def pyobj_from_npy_scalar(ctype):
    if ctype.startswith('npy_int'):
        return dict(c_code=pyobj_from_npy_int(ctype))
    elif ctype.startswith('npy_float'):
        return dict(c_code=pyobj_from_npy_float(ctype))
    elif ctype.startswith('npy_complex'):
        return dict(c_code=pyobj_from_npy_complex(ctype))
    raise NotImplementedError,`ctype`

