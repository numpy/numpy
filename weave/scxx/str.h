/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com
  
  modified heavily for weave by eric jones
*********************************************/
#if !defined(STR_H_INCLUDED_)
#define STR_H_INCLUDED_

#include <string>
#include "object.h"
#include "sequence.h"

namespace py {

class str : public sequence
{
public:
  str() : sequence() {};
  str(const char* s)
    : sequence(PyString_FromString((char* )s)) { lose_ref(_obj); }
  str(const char* s, int sz)
    : sequence(PyString_FromStringAndSize((char* )s, sz)) {  lose_ref(_obj); }
  str(const str& other)
    : sequence(other) {};
  str(PyObject* obj)
    : sequence(obj) { _violentTypeCheck(); };
  str(const object& other)
    : sequence(other) { _violentTypeCheck(); };
  virtual ~str() {};

  virtual str& operator=(const str& other) {
    grab_ref(other);
    return *this;
  };
  str& operator=(const object& other) {
    grab_ref(other);
    _violentTypeCheck();
    return *this;
  };
  virtual void _violentTypeCheck() {
    if (!PyString_Check(_obj)) {
      grab_ref(0);
      fail(PyExc_TypeError, "Not a Python String");
    }
  };
  operator const char* () const {
    return PyString_AsString(_obj);
  };
  /*
  static str format(const str& fmt, tuple& args){
    PyObject * rslt =PyString_Format(fmt, args);
    if (rslt==0)
      fail(PyExc_RuntimeError, "string format failed");
    return lose_ref(rslt);
  };
  */
}; // class str

} // namespace

#endif // STR_H_INCLUDED_
