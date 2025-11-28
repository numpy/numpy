====================
Using Python as glue
====================

.. warning::
   This was written in 2008 as part of the original
   `Guide to NumPy <https://archive.org/details/NumPyBook>`_ book
   by Travis E. Oliphant and is out of date.

|    There is no conversation more boring than the one where everybody
|    agrees.
|    --- *Michel de Montaigne*

|    Duct tape is like the force. It has a light side, and a dark side, and
|    it holds the universe together.
|    --- *Carl Zwanzig*

Many people like to say that Python is a fantastic glue language.
Hopefully, this Chapter will convince you that this is true. The first
adopters of Python for science were typically people who used it to
glue together large application codes running on super-computers. Not
only was it much nicer to code in Python than in a shell script or
Perl, in addition, the ability to easily extend Python made it
relatively easy to create new classes and types specifically adapted
to the problems being solved. From the interactions of these early
contributors, Numeric emerged as an array-like object that could be
used to pass data between these applications.

As Numeric has matured and developed into NumPy, people have been able
to write more code directly in NumPy. Often this code is fast-enough
for production use, but there are still times that there is a need to
access compiled code. Either to get that last bit of efficiency out of
the algorithm or to make it easier to access widely-available codes
written in C/C++ or Fortran.

This chapter will review many of the tools that are available for the
purpose of accessing code written in other compiled languages. There
are many resources available for learning to call other compiled
libraries from Python and the purpose of this Chapter is not to make
you an expert. The main goal is to make you aware of some of the
possibilities so that you will know what to "Google" in order to learn more.


Calling other compiled libraries from Python
============================================

While Python is a great language and a pleasure to code in, its
dynamic nature results in overhead that can cause some code ( *i.e.*
raw computations inside of for loops) to be up 10-100 times slower
than equivalent code written in a static compiled language. In
addition, it can cause memory usage to be larger than necessary as
temporary arrays are created and destroyed during computation. For
many types of computing needs, the extra slow-down and memory
consumption can often not be spared (at least for time- or memory-
critical portions of your code). Therefore one of the most common
needs is to call out from Python code to a fast, machine-code routine
(e.g. compiled using C/C++ or Fortran). The fact that this is
relatively easy to do is a big reason why Python is such an excellent
high-level language for scientific and engineering programming.

There are two basic approaches to calling compiled code: writing an
extension module that is then imported to Python using the import
command, or calling a shared-library subroutine directly from Python
using the `ctypes <https://docs.python.org/3/library/ctypes.html>`_
module.  Writing an extension module is the most common method.

.. warning::

    Calling C-code from Python can result in Python crashes if you are not
    careful. None of the approaches in this chapter are immune. You have
    to know something about the way data is handled by both NumPy and by
    the third-party library being used.


Hand-generated wrappers
=======================

Extension modules were discussed in :ref:`writing-an-extension`.
The most basic way to interface with compiled code is to write
an extension module and construct a module method that calls
the compiled code. For improved readability, your method should
take advantage of the ``PyArg_ParseTuple`` call to convert between
Python objects and C data-types. For standard C data-types there
is probably already a built-in converter. For others you may need 
to write your own converter and use the ``"O&"`` format string which
allows you to specify a function that will be used to perform the
conversion from the Python object to whatever C-structures are needed.

Once the conversions to the appropriate C-structures and C data-types
have been performed, the next step in the wrapper is to call the
underlying function. This is straightforward if the underlying
function is in C or C++. However, in order to call Fortran code you
must be familiar with how Fortran subroutines are called from C/C++
using your compiler and platform. This can vary somewhat platforms and
compilers (which is another reason f2py makes life much simpler for
interfacing Fortran code) but generally involves underscore mangling
of the name and the fact that all variables are passed by reference
(i.e. all arguments are pointers).

The advantage of the hand-generated wrapper is that you have complete
control over how the C-library gets used and called which can lead to
a lean and tight interface with minimal over-head. The disadvantage is
that you have to write, debug, and maintain C-code, although most of
it can be adapted using the time-honored technique of
"cutting-pasting-and-modifying" from other extension modules. Because
the procedure of calling out to additional C-code is fairly
regimented, code-generation procedures have been developed to make
this process easier. One of these code-generation techniques is
distributed with NumPy and allows easy integration with Fortran and
(simple) C code. This package, f2py, will be covered briefly in the
next section.


F2PY
====

F2PY allows you to automatically construct an extension module that
interfaces to routines in Fortran 77/90/95 code. It has the ability to
parse Fortran 77/90/95 code and automatically generate Python
signatures for the subroutines it encounters, or you can guide how the
subroutine interfaces with Python by constructing an interface-definition-file
(or modifying the f2py-produced one).

See the :ref:`F2PY documentation <f2py>` for more information and examples.

The f2py method of linking compiled code is currently the most
sophisticated and integrated approach. It allows clean separation of
Python with compiled code while still allowing for separate
distribution of the extension module. The only draw-back is that it
requires the existence of a Fortran compiler in order for a user to
install the code. However, with the existence of the free-compilers
g77, gfortran, and g95, as well as high-quality commercial compilers,
this restriction is not particularly onerous. In our opinion, Fortran
is still the easiest way to write fast and clear code for scientific
computing. It handles complex numbers, and multi-dimensional indexing
in the most straightforward way. Be aware, however, that some Fortran
compilers will not be able to optimize code as well as good hand-
written C-code.

.. index::
   single: f2py


Cython
======

`Cython <https://cython.org>`_ is a compiler for a Python dialect that adds
(optional) static typing for speed, and allows mixing C or C++ code
into your modules. It produces C or C++ extensions that can be compiled
and imported in Python code.

If you are writing an extension module that will include quite a bit of your
own algorithmic code as well, then Cython is a good match. Among its
features is the ability to easily and quickly
work with multidimensional arrays.

.. index::
   single: cython

Notice that Cython is an extension-module generator only. Unlike f2py,
it includes no automatic facility for compiling and linking
the extension module (which must be done in the usual fashion). It
does provide a modified distutils class called ``build_ext`` which lets
you build an extension module from a ``.pyx`` source. Thus, you could
write in a ``setup.py`` file:

.. code-block:: python

    from Cython.Distutils import build_ext
    from distutils.extension import Extension
    from distutils.core import setup
    import numpy

    setup(name='mine', description='Nothing',
          ext_modules=[Extension('filter', ['filter.pyx'],
                                 include_dirs=[numpy.get_include()])],
          cmdclass = {'build_ext':build_ext})

Adding the NumPy include directory is, of course, only necessary if
you are using NumPy arrays in the extension module (which is what we
assume you are using Cython for). The distutils extensions in NumPy
also include support for automatically producing the extension-module
and linking it from a ``.pyx`` file. It works so that if the user does
not have Cython installed, then it looks for a file with the same
file-name but a ``.c`` extension which it then uses instead of trying
to produce the ``.c`` file again.

If you just use Cython to compile a standard Python module, then you
will get a C extension module that typically runs a bit faster than the
equivalent Python module. Further speed increases can be gained by using
the ``cdef`` keyword to statically define C variables.

Let's look at two examples we've seen before to see how they might be
implemented using Cython. These examples were compiled into extension
modules using Cython 0.21.1.


Complex addition in Cython
--------------------------

Here is part of a Cython module named ``add.pyx`` which implements the
complex addition functions we previously implemented using f2py:

.. code-block:: cython

    cimport cython
    cimport numpy as np
    import numpy as np

    # We need to initialize NumPy.
    np.import_array()

    #@cython.boundscheck(False)
    def zadd(in1, in2):
        cdef double complex[:] a = in1.ravel()
        cdef double complex[:] b = in2.ravel()

        out = np.empty(a.shape[0], np.complex64)
        cdef double complex[:] c = out.ravel()

        for i in range(c.shape[0]):
            c[i].real = a[i].real + b[i].real
            c[i].imag = a[i].imag + b[i].imag

        return out

This module shows use of the ``cimport`` statement to load the definitions
from the ``numpy.pxd`` header that ships with Cython. It looks like NumPy is
imported twice; ``cimport`` only makes the NumPy C-API available, while the
regular ``import`` causes a Python-style import at runtime and makes it
possible to call into the familiar NumPy Python API.

The example also demonstrates Cython's "typed memoryviews", which are like
NumPy arrays at the C level, in the sense that they are shaped and strided
arrays that know their own extent (unlike a C array addressed through a bare
pointer). The syntax ``double complex[:]`` denotes a one-dimensional array
(vector) of doubles, with arbitrary strides. A contiguous array of ints would
be ``int[::1]``, while a matrix of floats would be ``float[:, :]``.

Shown commented is the ``cython.boundscheck`` decorator, which turns
bounds-checking for memory view accesses on or off on a per-function basis.
We can use this to further speed up our code, at the expense of safety
(or a manual check prior to entering the loop).

Other than the view syntax, the function is immediately readable to a Python
programmer. Static typing of the variable ``i`` is implicit. Instead of the
view syntax, we could also have used Cython's special NumPy array syntax,
but the view syntax is preferred.


Image filter in Cython
----------------------

The two-dimensional example we created using Fortran is just as easy to write
in Cython:

.. code-block:: cython

    cimport numpy as np
    import numpy as np

    np.import_array()

    def filter(img):
        cdef double[:, :] a = np.asarray(img, dtype=np.double)
        out = np.zeros(img.shape, dtype=np.double)
        cdef double[:, ::1] b = out

        cdef np.npy_intp i, j

        for i in range(1, a.shape[0] - 1):
            for j in range(1, a.shape[1] - 1):
                b[i, j] = (a[i, j]
                           + .5 * (  a[i-1, j] + a[i+1, j]
                                   + a[i, j-1] + a[i, j+1])
                           + .25 * (  a[i-1, j-1] + a[i-1, j+1]
                                    + a[i+1, j-1] + a[i+1, j+1]))

        return out

This 2-d averaging filter runs quickly because the loop is in C and
the pointer computations are done only as needed. If the code above is
compiled as a module ``image``, then a 2-d image, ``img``, can be filtered
using this code very quickly using:

.. code-block:: python

    import image
    out = image.filter(img)

Regarding the code, two things are of note: firstly, it is impossible to
return a memory view to Python. Instead, a NumPy array ``out`` is first
created, and then a view ``b`` onto this array is used for the computation.
Secondly, the view ``b`` is typed ``double[:, ::1]``. This means 2-d array
with contiguous rows, i.e., C matrix order. Specifying the order explicitly
can speed up some algorithms since they can skip stride computations.


Conclusion
----------

Cython is the extension mechanism of choice for several scientific Python
libraries, including Scipy, Pandas, SAGE, scikit-image and scikit-learn,
as well as the XML processing library LXML.
The language and compiler are well-maintained.

There are several disadvantages of using Cython:

1. When coding custom algorithms, and sometimes when wrapping existing C
   libraries, some familiarity with C is required. In particular, when using
   C memory management (``malloc`` and friends), it's easy to introduce
   memory leaks. However, just compiling a Python module renamed to ``.pyx``
   can already speed it up, and adding a few type declarations can give
   dramatic speedups in some code.

2. It is easy to lose a clean separation between Python and C which makes
   re-using your C-code for other non-Python-related projects more
   difficult.

3. The C-code generated by Cython is hard to read and modify (and typically
   compiles with annoying but harmless warnings).

One big advantage of Cython-generated extension modules is that they are
easy to distribute. In summary, Cython is a very capable tool for either
gluing C code or generating an extension module quickly and should not be
over-looked. It is especially useful for people that can't or won't write
C or Fortran code.

.. index::
   single: cython


ctypes
======

`ctypes <https://docs.python.org/3/library/ctypes.html>`_
is a Python extension module, included in the stdlib, that
allows you to call an arbitrary function in a shared library directly
from Python. This approach allows you to interface with C-code directly
from Python. This opens up an enormous number of libraries for use from
Python. The drawback, however, is that coding mistakes can lead to ugly
program crashes very easily (just as can happen in C) because there is
little type or bounds checking done on the parameters. This is especially
true when array data is passed in as a pointer to a raw memory
location. The responsibility is then on you that the subroutine will
not access memory outside the actual array area. But, if you don't
mind living a little dangerously ctypes can be an effective tool for
quickly taking advantage of a large shared library (or writing
extended functionality in your own shared library).

.. index::
   single: ctypes

Because the ctypes approach exposes a raw interface to the compiled
code it is not always tolerant of user mistakes. Robust use of the
ctypes module typically involves an additional layer of Python code in
order to check the data types and array bounds of objects passed to
the underlying subroutine. This additional layer of checking (not to
mention the conversion from ctypes objects to C-data-types that ctypes
itself performs), will make the interface slower than a hand-written
extension-module interface. However, this overhead should be negligible
if the C-routine being called is doing any significant amount of work.
If you are a great Python programmer with weak C skills, ctypes is an
easy way to write a useful interface to a (shared) library of compiled
code.

To use ctypes you must

1. Have a shared library.

2. Load the shared library.

3. Convert the Python objects to ctypes-understood arguments.

4. Call the function from the library with the ctypes arguments.


Having a shared library
-----------------------

There are several requirements for a shared library that can be used
with ctypes that are platform specific. This guide assumes you have
some familiarity with making a shared library on your system (or
simply have a shared library available to you). Items to remember are:

- A shared library must be compiled in a special way ( *e.g.* using
  the ``-shared`` flag with gcc).

- On some platforms (*e.g.* Windows), a shared library requires a
  .def file that specifies the functions to be exported. For example a
  mylib.def file might contain::

      LIBRARY mylib.dll
      EXPORTS
      cool_function1
      cool_function2

  Alternatively, you may be able to use the storage-class specifier
  ``__declspec(dllexport)`` in the C-definition of the function to avoid
  the need for this ``.def`` file.

There is no standard way in Python distutils to create a standard
shared library (an extension module is a "special" shared library
Python understands) in a cross-platform manner. Thus, a big
disadvantage of ctypes at the time of writing this book is that it is
difficult to distribute in a cross-platform manner a Python extension
that uses ctypes and includes your own code which should be compiled
as a shared library on the users system.


Loading the shared library
--------------------------

A simple, but robust way to load the shared library is to get the
absolute path name and load it using the cdll object of ctypes:

.. code-block:: python

    lib = ctypes.cdll[<full_path_name>]

However, on Windows accessing an attribute of the ``cdll`` method will
load the first DLL by that name found in the current directory or on
the PATH. Loading the absolute path name requires a little finesse for
cross-platform work since the extension of shared libraries varies.
There is a ``ctypes.util.find_library`` utility available that can
simplify the process of finding the library to load but it is not
foolproof. Complicating matters, different platforms have different
default extensions used by shared libraries (e.g. .dll -- Windows, .so
-- Linux, .dylib -- Mac OS X). This must also be taken into account if
you are using ctypes to wrap code that needs to work on several
platforms.

NumPy provides a convenience function called
``ctypeslib.load_library`` (name, path). This function takes the name
of the shared library (including any prefix like 'lib' but excluding
the extension) and a path where the shared library can be located. It
returns a ctypes library object or raises an ``OSError`` if the library
cannot be found or raises an ``ImportError`` if the ctypes module is not
available. (Windows users: the ctypes library object loaded using
``load_library`` is always loaded assuming cdecl calling convention.
See the ctypes documentation under ``ctypes.windll`` and/or ``ctypes.oledll``
for ways to load libraries under other calling conventions).

The functions in the shared library are available as attributes of the
ctypes library object (returned from ``ctypeslib.load_library``) or
as items using ``lib['func_name']`` syntax. The latter method for
retrieving a function name is particularly useful if the function name
contains characters that are not allowable in Python variable names.


Converting arguments
--------------------

Python ints/longs, strings, and unicode objects are automatically
converted as needed to equivalent ctypes arguments The None object is
also converted automatically to a NULL pointer. All other Python
objects must be converted to ctypes-specific types. There are two ways
around this restriction that allow ctypes to integrate with other
objects.

1. Don't set the argtypes attribute of the function object and define an
   ``_as_parameter_`` method for the object you want to pass in. The
   ``_as_parameter_`` method must return a Python int which will be passed
   directly to the function.

2. Set the argtypes attribute to a list whose entries contain objects
   with a classmethod named from_param that knows how to convert your
   object to an object that ctypes can understand (an int/long, string,
   unicode, or object with the ``_as_parameter_`` attribute).

NumPy uses both methods with a preference for the second method
because it can be safer. The ctypes attribute of the ndarray returns
an object that has an ``_as_parameter_`` attribute which returns an
integer representing the address of the ndarray to which it is
associated. As a result, one can pass this ctypes attribute object
directly to a function expecting a pointer to the data in your
ndarray. The caller must be sure that the ndarray object is of the
correct type, shape, and has the correct flags set or risk nasty
crashes if the data-pointer to inappropriate arrays are passed in.

To implement the second method, NumPy provides the class-factory
function :func:`ndpointer` in the :mod:`numpy.ctypeslib` module. This
class-factory function produces an appropriate class that can be
placed in an argtypes attribute entry of a ctypes function. The class
will contain a from_param method which ctypes will use to convert any
ndarray passed in to the function to a ctypes-recognized object. In
the process, the conversion will perform checking on any properties of
the ndarray that were specified by the user in the call to :func:`ndpointer`.
Aspects of the ndarray that can be checked include the data-type, the
number-of-dimensions, the shape, and/or the state of the flags on any
array passed. The return value of the from_param method is the ctypes
attribute of the array which (because it contains the ``_as_parameter_``
attribute pointing to the array data area) can be used by ctypes
directly.

The ctypes attribute of an ndarray is also endowed with additional
attributes that may be convenient when passing additional information
about the array into a ctypes function. The attributes **data**,
**shape**, and **strides** can provide ctypes compatible types
corresponding to the data-area, the shape, and the strides of the
array. The data attribute returns a ``c_void_p`` representing a
pointer to the data area. The shape and strides attributes each return
an array of ctypes integers (or None representing a NULL pointer, if a
0-d array). The base ctype of the array is a ctype integer of the same
size as a pointer on the platform. There are also methods
``data_as({ctype})``, ``shape_as(<base ctype>)``, and ``strides_as(<base
ctype>)``. These return the data as a ctype object of your choice and
the shape/strides arrays using an underlying base type of your choice.
For convenience, the ``ctypeslib`` module also contains ``c_intp`` as
a ctypes integer data-type whose size is the same as the size of
``c_void_p`` on the platform (its value is None if ctypes is not
installed).


Calling the function
--------------------

The function is accessed as an attribute of or an item from the loaded
shared-library. Thus, if ``./mylib.so`` has a function named
``cool_function1``, it may be accessed either as:

.. code-block:: python

    lib = numpy.ctypeslib.load_library('mylib','.')
    func1 = lib.cool_function1  # or equivalently
    func1 = lib['cool_function1']

In ctypes, the return-value of a function is set to be 'int' by
default. This behavior can be changed by setting the restype attribute
of the function. Use None for the restype if the function has no
return value ('void'):

.. code-block:: python

    func1.restype = None

As previously discussed, you can also set the argtypes attribute of
the function in order to have ctypes check the types of the input
arguments when the function is called. Use the :func:`ndpointer` factory
function to generate a ready-made class for data-type, shape, and
flags checking on your new function. The :func:`ndpointer` function has the
signature

.. function:: ndpointer(dtype=None, ndim=None, shape=None, flags=None)

    Keyword arguments with the value ``None`` are not checked.
    Specifying a keyword enforces checking of that aspect of the
    ndarray on conversion to a ctypes-compatible object. The dtype
    keyword can be any object understood as a data-type object. The
    ndim keyword should be an integer, and the shape keyword should be
    an integer or a sequence of integers. The flags keyword specifies
    the minimal flags that are required on any array passed in. This
    can be specified as a string of comma separated requirements, an
    integer indicating the requirement bits OR'd together, or a flags
    object returned from the flags attribute of an array with the
    necessary requirements.

Using an ndpointer class in the argtypes method can make it
significantly safer to call a C function using ctypes and the data-
area of an ndarray. You may still want to wrap the function in an
additional Python wrapper to make it user-friendly (hiding some
obvious arguments and making some arguments output arguments). In this
process, the ``requires`` function in NumPy may be useful to return the right
kind of array from a given input.


Complete example
----------------

In this example, we will demonstrate how the addition function and the filter
function implemented previously using the other approaches can be
implemented using ctypes. First, the C code which implements the
algorithms contains the functions ``zadd``, ``dadd``, ``sadd``, ``cadd``,
and ``dfilter2d``. The ``zadd`` function is:

.. code-block:: c

    /* Add arrays of contiguous data */
    typedef struct {double real; double imag;} cdouble;
    typedef struct {float real; float imag;} cfloat;
    void zadd(cdouble *a, cdouble *b, cdouble *c, long n)
    {
        while (n--) {
            c->real = a->real + b->real;
            c->imag = a->imag + b->imag;
            a++; b++; c++;
        }
    }

with similar code for ``cadd``, ``dadd``, and ``sadd`` that handles complex
float, double, and float data-types, respectively:

.. code-block:: c

    void cadd(cfloat *a, cfloat *b, cfloat *c, long n)
    {
            while (n--) {
                    c->real = a->real + b->real;
                    c->imag = a->imag + b->imag;
                    a++; b++; c++;
            }
    }
    void dadd(double *a, double *b, double *c, long n)
    {
            while (n--) {
                    *c++ = *a++ + *b++;
            }
    }
    void sadd(float *a, float *b, float *c, long n)
    {
            while (n--) {
                    *c++ = *a++ + *b++;
            }
    }

The ``code.c`` file also contains the function ``dfilter2d``:

.. code-block:: c

    /*
     * Assumes b is contiguous and has strides that are multiples of
     * sizeof(double)
     */
    void
    dfilter2d(double *a, double *b, ssize_t *astrides, ssize_t *dims)
    {
        ssize_t i, j, M, N, S0, S1;
        ssize_t r, c, rm1, rp1, cp1, cm1;

        M = dims[0]; N = dims[1];
        S0 = astrides[0]/sizeof(double);
        S1 = astrides[1]/sizeof(double);
        for (i = 1; i < M - 1; i++) {
            r = i*S0;
            rp1 = r + S0;
            rm1 = r - S0;
            for (j = 1; j < N - 1; j++) {
                c = j*S1;
                cp1 = j + S1;
                cm1 = j - S1;
                b[i*N + j] = a[r + c] +
                    (a[rp1 + c] + a[rm1 + c] +
                     a[r + cp1] + a[r + cm1])*0.5 +
                    (a[rp1 + cp1] + a[rp1 + cm1] +
                     a[rm1 + cp1] + a[rm1 + cp1])*0.25;
            }
        }
    }

A possible advantage this code has over the Fortran-equivalent code is
that it takes arbitrarily strided (i.e. non-contiguous arrays) and may
also run faster depending on the optimization capability of your
compiler. But, it is an obviously more complicated than the simple code
in ``filter.f``. This code must be compiled into a shared library. On my
Linux system this is accomplished using::

    gcc -o code.so -shared code.c

Which creates a shared_library named code.so in the current directory.
On Windows don't forget to either add ``__declspec(dllexport)`` in front
of void on the line preceding each function definition, or write a
``code.def`` file that lists the names of the functions to be exported.

A suitable Python interface to this shared library should be
constructed. To do this create a file named interface.py with the
following lines at the top:

.. code-block:: python

    __all__ = ['add', 'filter2d']

    import numpy as np
    import os

    _path = os.path.dirname('__file__')
    lib = np.ctypeslib.load_library('code', _path)
    _typedict = {'zadd' : complex, 'sadd' : np.single,
                 'cadd' : np.csingle, 'dadd' : float}
    for name in _typedict.keys():
        val = getattr(lib, name)
        val.restype = None
        _type = _typedict[name]
        val.argtypes = [np.ctypeslib.ndpointer(_type,
                          flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(_type,
                          flags='aligned, contiguous'),
                        np.ctypeslib.ndpointer(_type,
                          flags='aligned, contiguous,'\
                                'writeable'),
                        np.ctypeslib.c_intp]

This code loads the shared library named ``code.{ext}`` located in the
same path as this file. It then adds a return type of void to the
functions contained in the library. It also adds argument checking to
the functions in the library so that ndarrays can be passed as the
first three arguments along with an integer (large enough to hold a
pointer on the platform) as the fourth argument.

Setting up the filtering function is similar and allows the filtering
function to be called with ndarray arguments as the first two
arguments and with pointers to integers (large enough to handle the
strides and shape of an ndarray) as the last two arguments.:

.. code-block:: python

    lib.dfilter2d.restype=None
    lib.dfilter2d.argtypes = [np.ctypeslib.ndpointer(float, ndim=2,
                                           flags='aligned'),
                              np.ctypeslib.ndpointer(float, ndim=2,
                                     flags='aligned, contiguous,'\
                                           'writeable'),
                              ctypes.POINTER(np.ctypeslib.c_intp),
                              ctypes.POINTER(np.ctypeslib.c_intp)]

Next, define a simple selection function that chooses which addition
function to call in the shared library based on the data-type:

.. code-block:: python

    def select(dtype):
        if dtype.char in ['?bBhHf']:
            return lib.sadd, single
        elif dtype.char in ['F']:
            return lib.cadd, csingle
        elif dtype.char in ['DG']:
            return lib.zadd, complex
        else:
            return lib.dadd, float
        return func, ntype

Finally, the two functions to be exported by the interface can be
written simply as:

.. code-block:: python

    def add(a, b):
        requires = ['CONTIGUOUS', 'ALIGNED']
        a = np.asanyarray(a)
        func, dtype = select(a.dtype)
        a = np.require(a, dtype, requires)
        b = np.require(b, dtype, requires)
        c = np.empty_like(a)
        func(a,b,c,a.size)
        return c

and:

.. code-block:: python

    def filter2d(a):
        a = np.require(a, float, ['ALIGNED'])
        b = np.zeros_like(a)
        lib.dfilter2d(a, b, a.ctypes.strides, a.ctypes.shape)
        return b


Conclusion
----------

.. index::
   single: ctypes

Using ctypes is a powerful way to connect Python with arbitrary
C-code. Its advantages for extending Python include

- clean separation of C code from Python code

    - no need to learn a new syntax except Python and C

    - allows reuse of C code

    - functionality in shared libraries written for other purposes can be
      obtained with a simple Python wrapper and search for the library.


- easy integration with NumPy through the ctypes attribute

- full argument checking with the ndpointer class factory

Its disadvantages include

- It is difficult to distribute an extension module made using ctypes
  because of a lack of support for building shared libraries in
  distutils.

- You must have shared-libraries of your code (no static libraries).

- Very little support for C++ code and its different library-calling
  conventions. You will probably need a C wrapper around C++ code to use
  with ctypes (or just use Boost.Python instead).

Because of the difficulty in distributing an extension module made
using ctypes, f2py and Cython are still the easiest ways to extend Python
for package creation. However, ctypes is in some cases a useful alternative.
This should bring more features to ctypes that should
eliminate the difficulty in extending Python and distributing the
extension using ctypes.


Additional tools you may find useful
====================================

These tools have been found useful by others using Python and so are
included here. They are discussed separately because they are
either older ways to do things now handled by f2py, Cython, or ctypes
(SWIG, PyFort) or because of a lack of reasonable documentation (SIP, Boost).
Links to these methods are not included since the most relevant
can be found using Google or some other search engine, and any links provided
here would be quickly dated. Do not assume that inclusion in this list means
that the package deserves attention. Information about these packages are
collected here because many people have found them useful and we'd like to give
you as many options as possible for tackling the problem of easily integrating
your code.


SWIG
----

.. index::
   single: swig

Simplified Wrapper and Interface Generator (SWIG) is an old and fairly
stable method for wrapping C/C++-libraries to a large variety of other
languages. It does not specifically understand NumPy arrays but can be
made usable with NumPy through the use of typemaps. There are some
sample typemaps in the numpy/tools/swig directory under numpy.i together
with an example module that makes use of them. SWIG excels at wrapping
large C/C++ libraries because it can (almost) parse their headers and
auto-produce an interface. Technically, you need to generate a ``.i``
file that defines the interface. Often, however, this ``.i`` file can
be parts of the header itself. The interface usually needs a bit of
tweaking to be very useful. This ability to parse C/C++ headers and
auto-generate the interface still makes SWIG a useful approach to
adding functionality from C/C++ into Python, despite the other
methods that have emerged that are more targeted to Python. SWIG can
actually target extensions for several languages, but the typemaps
usually have to be language-specific. Nonetheless, with modifications
to the Python-specific typemaps, SWIG can be used to interface a
library with other languages such as Perl, Tcl, and Ruby.

My experience with SWIG has been generally positive in that it is
relatively easy to use and quite powerful. It has been used
often before becoming more proficient at writing C-extensions.
However, writing custom interfaces with SWIG is often troublesome because it
must be done using the concept of typemaps which are not Python
specific and are written in a C-like syntax. Therefore, other gluing strategies
are preferred and SWIG would be probably considered only to
wrap a very-large C/C++ library. Nonetheless, there are others who use
SWIG quite happily.


SIP
---

.. index::
   single: SIP

SIP is another tool for wrapping C/C++ libraries that is Python
specific and appears to have very good support for C++. Riverbank
Computing developed SIP in order to create Python bindings to the QT
library. An interface file must be written to generate the binding,
but the interface file looks a lot like a C/C++ header file. While SIP
is not a full C++ parser, it understands quite a bit of C++ syntax as
well as its own special directives that allow modification of how the
Python binding is accomplished. It also allows the user to define
mappings between Python types and C/C++ structures and classes.


Boost Python
------------

.. index::
   single: Boost.Python

Boost is a repository of C++ libraries and Boost.Python is one of
those libraries which provides a concise interface for binding C++
classes and functions to Python. The amazing part of the Boost.Python
approach is that it works entirely in pure C++ without introducing a
new syntax. Many users of C++ report that Boost.Python makes it
possible to combine the best of both worlds in a seamless fashion. Using Boost
to wrap simple C-subroutines is usually over-kill. Its primary purpose is to
make C++ classes available in Python. So, if you have a set of C++ classes that
need to be integrated cleanly into Python, consider learning about and using
Boost.Python.


Pyfort
------

Pyfort is a nice tool for wrapping Fortran and Fortran-like C-code
into Python with support for Numeric arrays. It was written by Paul
Dubois, a distinguished computer scientist and the very first
maintainer of Numeric (now retired). It is worth mentioning in the
hopes that somebody will update PyFort to work with NumPy arrays as
well which now support either Fortran or C-style contiguous arrays.
