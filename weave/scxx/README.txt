Overview
========

 SCXX (Simplified CXX) is a lightweight C++ wrapper for dealing with PyObjects.
 
 It is inspired by Paul Dubois' CXX (available from LLNL), but is much simpler.
 It does not use templates, so almost any compiler should work. It does not try
 to hide things like the Python method tables, or the init function. In fact, it
 only covers wrapping the most common PyObjects. No extra support is added (for
 example, you'll only get STL support for Python sequences from CXX).

 It lets you write C++ that looks a lot more like Python than the C API. Reference
 counts are handled automatically. It has not been optimized - it generally uses
 the highest possible level of the Python C API.

Classes
=======
 PWOBase	Base class; wraps any PyObject *
 PWONumber	Uses PyNumber_xxx calls. Automatically does the right thing.
 PWOSequence    Base class for all Python sequences. 
  PWOTuple	
  PWOString
  PWOList
 PWOMapping	Wraps a dictionary

internal
--------
 PWOMappingMmbr	Used to give PWOMappings Python (reference) semantics.
 PWOListMmbr	Used to give PWOLists Python (reference) semantics.

error
-----
 PWException	A C++ class that holds appropriate Python exception info.

General Notes
=============

 These classes can be used to create new Python objects, or wrap existing ones.

 Wrapping an existing one forces a typecheck (except for PWOBase, which doesn't
 care). So "PWOMapping dict(d);" will throw an exception if d is not a Python
 Mapping object (a set with one member - dicts). Or you can use the Python C API
 by casting to a PyObject * (e.g. "Pyxxx_Check((PyObject *)x)"). 

 Since errors are normally reported through exceptions, use code like this:

  try {
    //....
  }
  catch(PWException e) {
    return e.toPython();
  }

 To signal errors in your own code, use:
  throw PWException(PyExc_xxxx, msg);

 That is: throw a stack-based instance (not heap based), and give it the appropriate
 PyExc_xxx type and a string that the Python exception can show.

 To return a PWOxxx wrapped (or created) instance to Python, use disOwn():

  return PWONumber(7.0).disOwn();

 Without the disOwn(), the object would be deallocated before Python can get it.

 Note that the PWOxxx classes are generally designed to be created on the stack. The
 corresponding PyObject is on the heap. When the PWOxxx instance goes out of scope, 
 the PyObject is automatically decreffed (unless you've used disOwn()).

 See the MkWrap (http://www.equi4.com/metakit/mk4py/mkwrap/) project for extensive 
 use of SCXX. Just don't confuse the PWOxxx classes (which _wrap_ Python objects) and 
 the classes exposed by MkWrap (which are _both_ C++ objects _and_ Python objects).

Why SCXX
========

 I realize that a lot of effort has gone into CXX and some of the initiatives on
 the C++ SIG. I applaud those efforts; indeed, SCXX was inspired by CXX. But CXX
 uses fairly up to date features of C++, and wouldn't compile with most of the
 compilers I have, (I have to keep old versions around to support clients who still
 use them). On the one where it worked, it produced bloated code (because of the 
 way the compiler handles templates).

 For my purposes, I really only wanted one thing: wrap Python objects, and take
 care of the refcounts. The result is lightweight, and has the great advantage 
 (like CXX) that using PyObjects in C++ looks very much like using them in Python.

Instructions for use
====================

 Point to the installation directory with a -I directive.
 Include PWOImp.cpp in your make.

License
=======

 No restrictions on usage, modification or redistribution, as long as the copyright
 notice is maintained. No warranty whatsoever.

Contact
=======

 Gordon McMillan (McMillan Enterprises, Inc.) gmcm@hypernet.com.


 