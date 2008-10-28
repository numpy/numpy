"""
=============================
Subclassing ndarray in python
=============================

Credits
-------

This page is based with thanks on the wiki page on subclassing by Pierre
Gerard-Marchant - http://www.scipy.org/Subclasses.

Introduction
------------
Subclassing ndarray is relatively simple, but you will need to
understand some behavior of ndarrays to understand some minor
complications to subclassing.  There are examples at the bottom of the
page, but you will probably want to read the background to understand
why subclassing works as it does.

ndarrays and object creation
============================
The creation of ndarrays is complicated by the need to return views of
ndarrays, that are also ndarrays.  For example:

>>> import numpy as np
>>> arr = np.zeros((3,))
>>> type(arr)
<type 'numpy.ndarray'>
>>> v = arr[1:]
>>> type(v)
<type 'numpy.ndarray'>
>>> v is arr
False

So, when we take a view (here a slice) from the ndarray, we return a
new ndarray, that points to the data in the original.  When we
subclass ndarray, taking a view (such as a slice) needs to return an
object of our own class.  There is machinery to do this, but it is
this machinery that makes subclassing slightly non-standard.

To allow subclassing, and views of subclasses, ndarray uses the
ndarray ``__new__`` method for the main work of object initialization,
rather then the more usual ``__init__`` method.

``__new__`` and ``__init__``
============================

``__new__`` is a standard python method, and, if present, is called
before ``__init__`` when we create a class instance. Consider the
following::

  class C(object):
      def __new__(cls, *args):
          print 'Args in __new__:', args
          return object.__new__(cls, *args)
      def __init__(self, *args):
          print 'Args in __init__:', args

  C('hello')

The code gives the following output::

  cls is: <class '__main__.C'>
  Args in __new__: ('hello',)
  self is : <__main__.C object at 0xb7dc720c>
  Args in __init__: ('hello',)

When we call ``C('hello')``, the ``__new__`` method gets its own class
as first argument, and the passed argument, which is the string
``'hello'``.  After python calls ``__new__``, it usually (see below)
calls our ``__init__`` method, with the output of ``__new__`` as the
first argument (now a class instance), and the passed arguments
following.

As you can see, the object can be initialized in the ``__new__``
method or the ``__init__`` method, or both, and in fact ndarray does
not have an ``__init__`` method, because all the initialization is
done in the ``__new__`` method.

Why use ``__new__`` rather than just the usual ``__init__``?  Because
in some cases, as for ndarray, we want to be able to return an object
of some other class.  Consider the following::

  class C(object):
      def __new__(cls, *args):
          print 'cls is:', cls
          print 'Args in __new__:', args
          return object.__new__(cls, *args)
      def __init__(self, *args):
          print 'self is :', self
          print 'Args in __init__:', args

  class D(C):
      def __new__(cls, *args):
          print 'D cls is:', cls
          print 'D args in __new__:', args
          return C.__new__(C, *args)
      def __init__(self, *args):
          print 'D self is :', self
          print 'D args in __init__:', args

  D('hello')

which gives::

  D cls is: <class '__main__.D'>
  D args in __new__: ('hello',)
  cls is: <class '__main__.C'>
  Args in __new__: ('hello',)

The definition of ``C`` is the same as before, but for ``D``, the
``__new__`` method returns an instance of class ``C`` rather than
``D``.  Note that the ``__init__`` method of ``D`` does not get
called.  In general, when the ``__new__`` method returns an object of
class other than the class in which it is defined, the ``__init__``
method of that class is not called.

This is how subclasses of the ndarray class are able to return views
that preserve the class type.  When taking a view, the standard
ndarray machinery creates the new ndarray object with something
like::

  obj = ndarray.__new__(subtype, shape, ...

where ``subdtype`` is the subclass.  Thus the returned view is of the
same class as the subclass, rather than being of class ``ndarray``.

That solves the problem of returning views of the same type, but now
we have a new problem.  The machinery of ndarray can set the class
this way, in its standard methods for taking views, but the ndarray
``__new__`` method knows nothing of what we have done in our own
``__new__`` method in order to set attributes, and so on.  (Aside -
why not call ``obj = subdtype.__new__(...`` then?  Because we may not
have a ``__new__`` method with the same call signature).

So, when creating a new view object of our subclass, we need to be
able to set any extra attributes from the original object of our
class. This is the role of the ``__array_finalize__`` method of
ndarray.  ``__array_finalize__`` is called from within the
ndarray machinery, each time we create an ndarray of our own class,
and passes in the new view object, created as above, as well as the
old object from which the view has been taken.  In it we can take any
attributes from the old object and put then into the new view object,
or do any other related processing.  Now we are ready for a simple
example.

Simple example - adding an extra attribute to ndarray
-----------------------------------------------------

::

  import numpy as np

  class InfoArray(np.ndarray):

      def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
            strides=None, order=None, info=None):
          # Create the ndarray instance of our type, given the usual
          # input arguments.  This will call the standard ndarray
          # constructor, but return an object of our type
          obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
                           order)
          # add the new attribute to the created instance
          obj.info = info
          # Finally, we must return the newly created object:
          return obj

      def __array_finalize__(self,obj):
          # reset the attribute from passed original object
          self.info = getattr(obj, 'info', None)
          # We do not need to return anything

  obj = InfoArray(shape=(3,), info='information')
  print type(obj)
  print obj.info
  v = obj[1:]
  print type(v)
  print v.info

which gives::

  <class '__main__.InfoArray'>
  information
  <class '__main__.InfoArray'>
  information

This class isn't very useful, because it has the same constructor as
the bare ndarray object, including passing in buffers and shapes and
so on.   We would probably prefer to be able to take an already formed
ndarray from the usual numpy calls to ``np.array`` and return an
object.

Slightly more realistic example - attribute added to existing array
-------------------------------------------------------------------
Here is a class (with thanks to Pierre GM for the original example),
that takes array that already exists, casts as our type, and adds an
extra attribute::

  import numpy as np

  class RealisticInfoArray(np.ndarray):

      def __new__(cls, input_array, info=None):
          # Input array is an already formed ndarray instance
          # We first cast to be our class type
          obj = np.asarray(input_array).view(cls)
          # add the new attribute to the created instance
          obj.info = info
          # Finally, we must return the newly created object:
          return obj

      def __array_finalize__(self,obj):
          # reset the attribute from passed original object
          self.info = getattr(obj, 'info', None)
          # We do not need to return anything

  arr = np.arange(5)
  obj = RealisticInfoArray(arr, info='information')
  print type(obj)
  print obj.info
  v = obj[1:]
  print type(v)
  print v.info

which gives::

  <class '__main__.RealisticInfoArray'>
  information
  <class '__main__.RealisticInfoArray'>
  information

``__array_wrap__`` for ufuncs
-----------------------------

Let's say you have an instance ``obj`` of your new subclass,
``RealisticInfoArray``, and you pass it into a ufunc with another
array::

  arr = np.arange(5)
  ret = np.multiply.outer(arr, obj)

When a numpy ufunc is called on a subclass of ndarray, the
__array_wrap__ method is called to transform the result into a new
instance of the subclass. By default, __array_wrap__ will call
__array_finalize__, and the attributes will be inherited.

By defining a specific __array_wrap__ method for our subclass, we can
tweak the output. The __array_wrap__ method requires one argument, the
object on which the ufunc is applied, and an optional parameter
*context*. This parameter is returned by some ufuncs as a 3-element
tuple: (name of the ufunc, argument of the ufunc, domain of the
ufunc). See the masked array subclass for an implementation.

Extra gotchas - custom __del__ methods and ndarray.base
-------------------------------------------------------
One of the problems that ndarray solves is that of memory ownership of
ndarrays and their views.  Consider the case where we have created an
ndarray, ``arr`` and then taken a view with ``v = arr[1:]``.  If we
then do ``del v``, we need to make sure that the ``del`` does not
delete the memory pointed to by the view, because we still need it for
the original ``arr`` object.  Numpy therefore keeps track of where the
data came from for a particular array or view, with the ``base`` attribute::

  import numpy as np

  # A normal ndarray, that owns its own data
  arr = np.zeros((4,))
  # In this case, base is None
  assert arr.base is None
  # We take a view
  v1 = arr[1:]
  # base now points to the array that it derived from
  assert v1.base is arr
  # Take a view of a view
  v2 = v1[1:]
  # base points to the view it derived from
  assert v2.base is v1

The assertions all succeed in this case.  In general, if the array
owns its own memory, as for ``arr`` in this case, then ``arr.base``
will be None - there are some exceptions to this - see the numpy book
for more details.

The ``base`` attribute is useful in being able to tell whether we have
a view or the original array.  This in turn can be useful if we need
to know whether or not to do some specific cleanup when the subclassed
array is deleted.  For example, we may only want to do the cleanup if
the original array is deleted, but not the views.  For an example of
how this can work, have a look at the ``memmap`` class in
``numpy.core``.

"""
