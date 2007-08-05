
from base import Component
from c_code import CCode

Component.register(
    CCode('''
static int pyobj_to_int(PyObject *obj, int* value) {
  int status = 1;
  if (PyInt_Check(obj)) {
    *value = PyInt_AS_LONG(obj);
    status = 0;
  }
  return status;
}
''', provides='pyobj_to_int'),
    CCode('''\
static PyObject* pyobj_from_int(int* value) {
  return PyInt_FromLong(*value);
}
''', provides='pyobj_from_int'),

    )
