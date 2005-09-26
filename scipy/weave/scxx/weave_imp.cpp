/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com

  modified for weave by eric jones.
*********************************************/
#include "object.h"

using namespace py;

//---------------------------------------------------------------------------
// object
//
// !! Wish I knew how to get these defined inline in object.h...
//---------------------------------------------------------------------------

object::keyed_ref object::operator [] (object& key) {
  object rslt = PyObject_GetItem(_obj, key);
  lose_ref(rslt);
  if (!(PyObject*)rslt)
  {
    // don't throw error for when [] fails because it might be on left hand 
    // side (a[0] = 1).  If the obj was just created, it will be filled 
    // with NULL values, and setting the values should be ok.  However, we
    // do want to catch index errors that might occur on the right hand side
    // (obj = a[4] when a has len==3).
      if (PyErr_ExceptionMatches(PyExc_KeyError))
           PyErr_Clear(); // Ignore key errors
      else if (PyErr_ExceptionMatches(PyExc_IndexError))
        throw 1;
  }
  return object::keyed_ref(rslt, *this, key);
};
object::keyed_ref object::operator [] (const char* key) {
  object _key = object(key);
  return operator[](_key);
};
object::keyed_ref object::operator [] (const std::string& key) {
  object _key = object(key);
  return operator [](_key);
};
object::keyed_ref object::operator [] (int key) {
  object _key = object(key);
  return operator [](_key);
};
object::keyed_ref object::operator [] (double key) {
  object _key = object(key);
  return operator [](_key);
};
object::keyed_ref object::operator [] (const std::complex<double>& key) {
  object _key = object(key);
  return operator [](_key);
};

std::ostream& operator <<(std::ostream& os, py::object& obj)
{
    os << obj.repr();
    return os;
}

//---------------------------------------------------------------------------
// Fail method for throwing exceptions with a given message.
//---------------------------------------------------------------------------

void py::fail(PyObject* exc, const char* msg)
{
  PyErr_SetString(exc, msg);
  throw 1;
}
