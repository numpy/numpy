/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com

  modified heavily for weave by eric jones.
*********************************************/

#if !defined(OBJECT_H_INCLUDED_)
#define OBJECT_H_INCLUDED_

#include <Python.h>
#include <limits.h>

namespace py {

void Fail(PyObject*, const char* msg);
    
class object  
{
protected:
  PyObject* _obj;

    // incref new owner, decref old owner, and adjust to new owner
  void GrabRef(PyObject* newObj);
    // decrease reference count without destroying the object
  static PyObject* LoseRef(PyObject* o)
    { if (o != 0) --(o->ob_refcnt); return o; }

private:
  PyObject* _own; // set to _obj if we "own" a reference to _obj, else zero

public:
  object()
    : _obj (0), _own (0) { }
  object(const object& other)
    : _obj (0), _own (0) { GrabRef(other); }
  object(PyObject* obj)
    : _obj (0), _own (0) { GrabRef(obj); }

  virtual ~object()
    { Py_XDECREF(_own); }
  
  object& operator=(const object& other) {
    GrabRef(other);
    return *this;
  };
  operator PyObject* () const {
    return _obj;
  };
  int print(FILE *f, int flags) const {
    return PyObject_Print(_obj, f, flags);
  };
  bool hasattr(const char* nm) const {
    return PyObject_HasAttrString(_obj, (char*) nm) == 1;
  };

  // Need to change return type?
  PyObject* attr(const char* nm) const {
    return LoseRef(PyObject_GetAttrString(_obj, (char*) nm));
  };
  PyObject* attr(const object& nm) const {
    return LoseRef(PyObject_GetAttr(_obj, nm));
  };
  
  
  int set_attr(const char* nm, object& val) {
    return PyObject_SetAttrString(_obj, (char*) nm, val);
  };
  int set_attr(PyObject* nm, object& val) {
    return PyObject_SetAttr(_obj, nm, val);
  };
  
  int del(const char* nm) {
    return PyObject_DelAttrString(_obj, (char*) nm);
  };
  int del(const object& nm) {
    return PyObject_DelAttr(_obj, nm);
  };
  
  int cmp(const object& other) const {
    int rslt = 0;
    int rc = PyObject_Cmp(_obj, other, &rslt);
    if (rc == -1)
      Fail(PyExc_TypeError, "cannot make the comparison");
    return rslt;
  };
  bool operator == (const object& other) const {
    return cmp(other) == 0;
  };
  bool operator != (const object& other) const {
    return cmp(other) != 0;
  };
  bool operator > (const object& other) const {
    return cmp(other) > 0;
  };
  bool operator < (const object& other) const {
    return cmp(other) < 0;
  };
  bool operator >= (const object& other) const {
    return cmp(other) >= 0;
  };
  bool operator <= (const object& other) const {
    return cmp(other) <= 0;
  };
      
  PyObject* repr() const {
    return LoseRef(PyObject_Repr(_obj));
  };
  /*
  PyObject* str() const {
    return LoseRef(PyObject_Str(_obj));
  };
  */
  bool is_callable() const {
    return PyCallable_Check(_obj) == 1;
  };
  int hash() const {
    return PyObject_Hash(_obj);
  };
  bool is_true() const {
    return PyObject_IsTrue(_obj) == 1;
  };
  PyObject* type() const {
    return LoseRef(PyObject_Type(_obj));
  };
  PyObject* disown() {
    _own = 0;
    return _obj;
  };
};

} // namespace

#endif // !defined(OBJECT_H_INCLUDED_)
