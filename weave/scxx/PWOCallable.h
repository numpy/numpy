/******************************************** 
  copyright 2000 McMillan Enterprises, Inc.
  www.mcmillan-inc.com
*********************************************/
#if !defined(PWOCALLABLE_H_INCLUDED_)
#define PWOCALLABLE_H_INCLUDED_

#include "PWOBase.h"
#include "PWOSequence.h"
#include "PWOMapping.h"

class PWOCallable : public PWOBase
{
public:
  PWOCallable() : PWOBase() {};
  PWOCallable(PyObject *obj) : PWOBase(obj) {
    _violentTypeCheck();
  };
  virtual ~PWOCallable() {};
  virtual PWOCallable& operator=(const PWOCallable& other) {
    GrabRef(other);
    return *this;
  };
  PWOCallable& operator=(const PWOBase& other) {
    GrabRef(other);
    _violentTypeCheck();
    return *this;
  };
  virtual void _violentTypeCheck() {
    if (!isCallable()) {
      GrabRef(0);
      Fail(PyExc_TypeError, "Not a callable object");
    }
  };
  PWOBase call() const;
  PWOBase call(PWOTuple& args) const;
  PWOBase call(PWOTuple& args, PWOMapping& kws) const;
};

#endif
