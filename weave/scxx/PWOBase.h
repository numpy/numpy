/******************************************** 
  copyright 1999 McMillan Enterprises, Inc.
  www.mcmillan-inc.com
*********************************************/
// PWOBase.h: interface for the PWOBase class.

#if !defined(PWOBASE_H_INCLUDED_)
#define PWOBASE_H_INCLUDED_

#include <Python.h>
#include <limits.h>

void Fail(PyObject*, const char* msg);

class PWOBase  
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
  PWOBase()
    : _obj (0), _own (0) { }
  PWOBase(const PWOBase& other)
    : _obj (0), _own (0) { GrabRef(other); }
  PWOBase(PyObject* obj)
    : _obj (0), _own (0) { GrabRef(obj); }

  virtual ~PWOBase()
    { Py_XDECREF(_own); }
  
  PWOBase& operator=(const PWOBase& other) {
    GrabRef(other);
    return *this;
  };
  operator PyObject* () const {
    return _obj;
  };
  int print(FILE *f, int flags) const {
    return PyObject_Print(_obj, f, flags);
  };
  bool hasAttr(const char* nm) const {
    return PyObject_HasAttrString(_obj, (char*) nm) == 1;
  };
  //bool hasAttr(PyObject* nm) const {
  //      return PyObject_HasAttr(_obj, nm);
  //};
  PyObject* getAttr(const char* nm) const {
    return LoseRef(PyObject_GetAttrString(_obj, (char*) nm));
  };
  PyObject* getAttr(const PWOBase& nm) const {
    return LoseRef(PyObject_GetAttr(_obj, nm));
  };
  int setAttr(const char* nm, PWOBase& val) {
    return PyObject_SetAttrString(_obj, (char*) nm, val);
  };
  int setAttr(PyObject* nm, PWOBase& val) {
    return PyObject_SetAttr(_obj, nm, val);
  };
  int delAttr(const char* nm) {
    return PyObject_DelAttrString(_obj, (char*) nm);
  };
  int delAttr(const PWOBase& nm) {
    return PyObject_DelAttr(_obj, nm);
  };
  int cmp(const PWOBase& other) const {
    int rslt = 0;
    int rc = PyObject_Cmp(_obj, other, &rslt);
    if (rc == -1)
      Fail(PyExc_TypeError, "cannot make the comparison");
    return rslt;
  };
  bool operator == (const PWOBase& other) const {
    return cmp(other) == 0;
  };
  bool operator != (const PWOBase& other) const {
    return cmp(other) != 0;
  };
  bool operator > (const PWOBase& other) const {
    return cmp(other) > 0;
  };
  bool operator < (const PWOBase& other) const {
    return cmp(other) < 0;
  };
  bool operator >= (const PWOBase& other) const {
    return cmp(other) >= 0;
  };
  bool operator <= (const PWOBase& other) const {
    return cmp(other) <= 0;
  };
      
  PyObject* repr() const {
    return LoseRef(PyObject_Repr(_obj));
  };
  PyObject* str() const {
    return LoseRef(PyObject_Str(_obj));
  };
  bool isCallable() const {
    return PyCallable_Check(_obj) == 1;
  };
  int hash() const {
    return PyObject_Hash(_obj);
  };
  bool isTrue() const {
    return PyObject_IsTrue(_obj) == 1;
  };
  PyObject* type() const {
    return LoseRef(PyObject_Type(_obj));
  };
  PyObject* disOwn() {
    _own = 0;
    return _obj;
  };
};

#endif // !defined(PWOBASE_H_INCLUDED_)
