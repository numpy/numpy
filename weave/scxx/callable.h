/******************************************** 
  copyright 2000 McMillan Enterprises, Inc.
  www.mcmillan-inc.com
  
  modified for weave by eric jones
*********************************************/

#if !defined(CALLABLE_H_INCLUDED_)
#define CALLABLE_H_INCLUDED_

#include "object.h"
#include "tuple.h"
#include "dict.h"

namespace py {
    
class callable : public object
{
public:
  callable() : object() {};
  callable(PyObject *obj) : object(obj) {
    _violentTypeCheck();
  };
  virtual ~callable() {};
  virtual callable& operator=(const callable& other) {
    GrabRef(other);
    return *this;
  };
  callable& operator=(const object& other) {
    GrabRef(other);
    _violentTypeCheck();
    return *this;
  };
  virtual void _violentTypeCheck() {
    if (!is_callable()) {
      GrabRef(0);
      Fail(PyExc_TypeError, "Not a callable object");
    }
  };
  object call() const;
  object call(tuple& args) const;
  object call(tuple& args, dict& kws) const;
};

} // namespace py
#endif
