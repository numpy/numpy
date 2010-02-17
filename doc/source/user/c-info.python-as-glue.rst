********************
Using Python as glue
********************

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

The http://www.scipy.org website also contains a great deal of useful
information about many of these tools. For example, there is a nice
description of using several of the tools explained in this chapter at
http://www.scipy.org/PerformancePython. This link provides several
ways to solve the same problem showing how to use and connect with
compiled code to get the best performance. In the process you can get
a taste for several of the approaches that will be discussed in this
chapter.


Calling other compiled libraries from Python
============================================

While Python is a great language and a pleasure to code in, its
dynamic nature results in overhead that can cause some code ( *i.e.*
raw computations inside of for loops) to be up 10-100 times slower
than equivalent code written in a static compiled language. In
addition, it can cause memory usage to be larger than necessary as
temporary arrays are created and destroyed during computation. For
many types of computing needs the extra slow-down and memory
consumption can often not be spared (at least for time- or memory-
critical portions of your code). Therefore one of the most common
needs is to call out from Python code to a fast, machine-code routine
(e.g. compiled using C/C++ or Fortran). The fact that this is
relatively easy to do is a big reason why Python is such an excellent
high-level language for scientific and engineering programming.

Their are two basic approaches to calling compiled code: writing an
extension module that is then imported to Python using the import
command, or calling a shared-library subroutine directly from Python
using the ctypes module (included in the standard distribution with
Python 2.5). The first method is the most common (but with the
inclusion of ctypes into Python 2.5 this status may change).

.. warning::

    Calling C-code from Python can result in Python crashes if you are not
    careful. None of the approaches in this chapter are immune. You have
    to know something about the way data is handled by both NumPy and by
    the third-party library being used.


Hand-generated wrappers
=======================

Extension modules were discussed in Chapter `1
<#sec-writing-an-extension>`__ . The most basic way to interface with
compiled code is to write an extension module and construct a module
method that calls the compiled code. For improved readability, your
method should take advantage of the PyArg_ParseTuple call to convert
between Python objects and C data-types. For standard C data-types
there is probably already a built-in converter. For others you may
need to write your own converter and use the "O&" format string which
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
"cutting-pasting-and-modifying" from other extension modules. Because,
the procedure of calling out to additional C-code is fairly
regimented, code-generation procedures have been developed to make
this process easier. One of these code- generation techniques is
distributed with NumPy and allows easy integration with Fortran and
(simple) C code. This package, f2py, will be covered briefly in the
next session.


f2py
====

F2py allows you to automatically construct an extension module that
interfaces to routines in Fortran 77/90/95 code. It has the ability to
parse Fortran 77/90/95 code and automatically generate Python
signatures for the subroutines it encounters, or you can guide how the
subroutine interfaces with Python by constructing an interface-definition-file (or modifying the f2py-produced one).

.. index::
   single: f2py

Creating source for a basic extension module
--------------------------------------------

Probably the easiest way to introduce f2py is to offer a simple
example. Here is one of the subroutines contained in a file named
:file:`add.f`:

.. code-block:: none

    C
          SUBROUTINE ZADD(A,B,C,N)
    C
          DOUBLE COMPLEX A(*)
          DOUBLE COMPLEX B(*)
          DOUBLE COMPLEX C(*)
          INTEGER N
          DO 20 J = 1, N
             C(J) = A(J)+B(J)
     20   CONTINUE
          END

This routine simply adds the elements in two contiguous arrays and
places the result in a third. The memory for all three arrays must be
provided by the calling routine. A very basic interface to this
routine can be automatically generated by f2py::

    f2py -m add add.f

You should be able to run this command assuming your search-path is
set-up properly. This command will produce an extension module named
addmodule.c in the current directory. This extension module can now be
compiled and used from Python just like any other extension module.


Creating a compiled extension module
------------------------------------

You can also get f2py to compile add.f and also compile its produced
extension module leaving only a shared-library extension file that can
be imported from Python::

    f2py -c -m add add.f

This command leaves a file named add.{ext} in the current directory
(where {ext} is the appropriate extension for a python extension
module on your platform --- so, pyd, *etc.* ). This module may then be
imported from Python. It will contain a method for each subroutine in
add (zadd, cadd, dadd, sadd). The docstring of each method contains
information about how the module method may be called:

    >>> import add
    >>> print add.zadd.__doc__
    zadd - Function signature:
      zadd(a,b,c,n)
    Required arguments:
      a : input rank-1 array('D') with bounds (*)
      b : input rank-1 array('D') with bounds (*)
      c : input rank-1 array('D') with bounds (*)
      n : input int


Improving the basic interface
-----------------------------

The default interface is a very literal translation of the fortran
code into Python. The Fortran array arguments must now be NumPy arrays
and the integer argument should be an integer. The interface will
attempt to convert all arguments to their required types (and shapes)
and issue an error if unsuccessful. However, because it knows nothing
about the semantics of the arguments (such that C is an output and n
should really match the array sizes), it is possible to abuse this
function in ways that can cause Python to crash. For example:

    >>> add.zadd([1,2,3],[1,2],[3,4],1000)

will cause a program crash on most systems. Under the covers, the
lists are being converted to proper arrays but then the underlying add
loop is told to cycle way beyond the borders of the allocated memory.

In order to improve the interface, directives should be provided. This
is accomplished by constructing an interface definition file. It is
usually best to start from the interface file that f2py can produce
(where it gets its default behavior from). To get f2py to generate the
interface file use the -h option::

    f2py -h add.pyf -m add add.f

This command leaves the file add.pyf in the current directory. The
section of this file corresponding to zadd is:

.. code-block:: none

    subroutine zadd(a,b,c,n) ! in :add:add.f
       double complex dimension(*) :: a
       double complex dimension(*) :: b
       double complex dimension(*) :: c
       integer :: n
    end subroutine zadd

By placing intent directives and checking code, the interface can be
cleaned up quite a bit until the Python module method is both easier
to use and more robust.

.. code-block:: none

    subroutine zadd(a,b,c,n) ! in :add:add.f
       double complex dimension(n) :: a
       double complex dimension(n) :: b
       double complex intent(out),dimension(n) :: c
       integer intent(hide),depend(a) :: n=len(a)
    end subroutine zadd

The intent directive, intent(out) is used to tell f2py that ``c`` is
an output variable and should be created by the interface before being
passed to the underlying code. The intent(hide) directive tells f2py
to not allow the user to specify the variable, ``n``, but instead to
get it from the size of ``a``. The depend( ``a`` ) directive is
necessary to tell f2py that the value of n depends on the input a (so
that it won't try to create the variable n until the variable a is
created).

The new interface has docstring:

    >>> print add.zadd.__doc__
    zadd - Function signature:
      c = zadd(a,b)
    Required arguments:
      a : input rank-1 array('D') with bounds (n)
      b : input rank-1 array('D') with bounds (n)
    Return objects:
      c : rank-1 array('D') with bounds (n)

Now, the function can be called in a much more robust way:

    >>> add.zadd([1,2,3],[4,5,6])
    array([ 5.+0.j,  7.+0.j,  9.+0.j])

Notice the automatic conversion to the correct format that occurred.


Inserting directives in Fortran source
--------------------------------------

The nice interface can also be generated automatically by placing the
variable directives as special comments in the original fortran code.
Thus, if I modify the source code to contain:

.. code-block:: none

    C
          SUBROUTINE ZADD(A,B,C,N)
    C
    CF2PY INTENT(OUT) :: C
    CF2PY INTENT(HIDE) :: N
    CF2PY DOUBLE COMPLEX :: A(N)
    CF2PY DOUBLE COMPLEX :: B(N)
    CF2PY DOUBLE COMPLEX :: C(N)
          DOUBLE COMPLEX A(*)
          DOUBLE COMPLEX B(*)
          DOUBLE COMPLEX C(*)
          INTEGER N
          DO 20 J = 1, N
             C(J) = A(J) + B(J)
     20   CONTINUE
          END

Then, I can compile the extension module using::

    f2py -c -m add add.f

The resulting signature for the function add.zadd is exactly the same
one that was created previously. If the original source code had
contained A(N) instead of A(\*) and so forth with B and C, then I
could obtain (nearly) the same interface simply by placing the
INTENT(OUT) :: C comment line in the source code. The only difference
is that N would be an optional input that would default to the length
of A.


A filtering example
-------------------

For comparison with the other methods to be discussed. Here is another
example of a function that filters a two-dimensional array of double
precision floating-point numbers using a fixed averaging filter. The
advantage of using Fortran to index into multi-dimensional arrays
should be clear from this example.

.. code-block:: none

          SUBROUTINE DFILTER2D(A,B,M,N)
    C
          DOUBLE PRECISION A(M,N)
          DOUBLE PRECISION B(M,N)
          INTEGER N, M
    CF2PY INTENT(OUT) :: B
    CF2PY INTENT(HIDE) :: N
    CF2PY INTENT(HIDE) :: M
          DO 20 I = 2,M-1
             DO 40 J=2,N-1
                B(I,J) = A(I,J) +
         $           (A(I-1,J)+A(I+1,J) +
         $            A(I,J-1)+A(I,J+1) )*0.5D0 +
         $           (A(I-1,J-1) + A(I-1,J+1) +
         $            A(I+1,J-1) + A(I+1,J+1))*0.25D0
     40      CONTINUE
     20   CONTINUE
          END

This code can be compiled and linked into an extension module named
filter using::

    f2py -c -m filter filter.f

This will produce an extension module named filter.so in the current
directory with a method named dfilter2d that returns a filtered
version of the input.


Calling f2py from Python
------------------------

The f2py program is written in Python and can be run from inside your
module. This provides a facility that is somewhat similar to the use
of weave.ext_tools described below. An example of the final interface
executed using Python code is:

.. code-block:: python

    import numpy.f2py as f2py
    fid = open('add.f')
    source = fid.read()
    fid.close()
    f2py.compile(source, modulename='add')
    import add

The source string can be any valid Fortran code. If you want to save
the extension-module source code then a suitable file-name can be
provided by the source_fn keyword to the compile function.


Automatic extension module generation
-------------------------------------

If you want to distribute your f2py extension module, then you only
need to include the .pyf file and the Fortran code. The distutils
extensions in NumPy allow you to define an extension module entirely
in terms of this interface file. A valid setup.py file allowing
distribution of the add.f module (as part of the package f2py_examples
so that it would be loaded as f2py_examples.add) is:

.. code-block:: python

    def configuration(parent_package='', top_path=None)
        from numpy.distutils.misc_util import Configuration
        config = Configuration('f2py_examples',parent_package, top_path)
        config.add_extension('add', sources=['add.pyf','add.f'])
        return config

    if __name__ == '__main__':
        from numpy.distutils.core import setup
        setup(**configuration(top_path='').todict())

Installation of the new package is easy using::

    python setup.py install

assuming you have the proper permissions to write to the main site-
packages directory for the version of Python you are using. For the
resulting package to work, you need to create a file named __init__.py
(in the same directory as add.pyf). Notice the extension module is
defined entirely in terms of the "add.pyf" and "add.f" files. The
conversion of the .pyf file to a .c file is handled by numpy.disutils.


Conclusion
----------

The interface definition file (.pyf) is how you can fine-tune the
interface between Python and Fortran. There is decent documentation
for f2py found in the numpy/f2py/docs directory where-ever NumPy is
installed on your system (usually under site-packages). There is also
more information on using f2py (including how to use it to wrap C
codes) at http://www.scipy.org/Cookbook under the "Using NumPy with
Other Languages" heading.

The f2py method of linking compiled code is currently the most
sophisticated and integrated approach. It allows clean separation of
Python with compiled code while still allowing for separate
distribution of the extension module. The only draw-back is that it
requires the existence of a Fortran compiler in order for a user to
install the code. However, with the existence of the free-compilers
g77, gfortran, and g95, as well as high-quality commerical compilers,
this restriction is not particularly onerous. In my opinion, Fortran
is still the easiest way to write fast and clear code for scientific
computing. It handles complex numbers, and multi-dimensional indexing
in the most straightforward way. Be aware, however, that some Fortran
compilers will not be able to optimize code as well as good hand-
written C-code.

.. index::
   single: f2py


weave
=====

Weave is a scipy package that can be used to automate the process of
extending Python with C/C++ code. It can be used to speed up
evaluation of an array expression that would otherwise create
temporary variables, to directly "inline" C/C++ code into Python, or
to create a fully-named extension module.  You must either install
scipy or get the weave package separately and install it using the
standard python setup.py install. You must also have a C/C++-compiler
installed and useable by Python distutils in order to use weave.

.. index::
   single: weave

Somewhat dated, but still useful documentation for weave can be found
at the link http://www.scipy/Weave. There are also many examples found
in the examples directory which is installed under the weave directory
in the place where weave is installed on your system.


Speed up code involving arrays (also see scipy.numexpr)
-------------------------------------------------------

This is the easiest way to use weave and requires minimal changes to
your Python code. It involves placing quotes around the expression of
interest and calling weave.blitz. Weave will parse the code and
generate C++ code using Blitz C++ arrays. It will then compile the
code and catalog the shared library so that the next time this exact
string is asked for (and the array types are the same), the already-
compiled shared library will be loaded and used. Because Blitz makes
extensive use of C++ templating, it can take a long time to compile
the first time. After that, however, the code should evaluate more
quickly than the equivalent NumPy expression. This is especially true
if your array sizes are large and the expression would require NumPy
to create several temporaries. Only expressions involving basic
arithmetic operations and basic array slicing can be converted to
Blitz C++ code.

For example, consider the expression::

    d = 4*a + 5*a*b + 6*b*c

where a, b, and c are all arrays of the same type and shape. When the
data-type is double-precision and the size is 1000x1000, this
expression takes about 0.5 seconds to compute on an 1.1Ghz AMD Athlon
machine. When this expression is executed instead using blitz:

.. code-block:: python

    d = empty(a.shape, 'd'); weave.blitz(expr)

execution time is only about 0.20 seconds (about 0.14 seconds spent in
weave and the rest in allocating space for d). Thus, we've sped up the
code by a factor of 2 using only a simnple command (weave.blitz). Your
mileage may vary, but factors of 2-8 speed-ups are possible with this
very simple technique.

If you are interested in using weave in this way, then you should also
look at scipy.numexpr which is another similar way to speed up
expressions by eliminating the need for temporary variables. Using
numexpr does not require a C/C++ compiler.


Inline C-code
-------------

Probably the most widely-used method of employing weave is to
"in-line" C/C++ code into Python in order to speed up a time-critical
section of Python code. In this method of using weave, you define a
string containing useful C-code and then pass it to the function
**weave.inline** ( ``code_string``, ``variables`` ), where
code_string is a string of valid C/C++ code and variables is a list of
variables that should be passed in from Python. The C/C++ code should
refer to the variables with the same names as they are defined with in
Python. If weave.line should return anything the the special value
return_val should be set to whatever object should be returned. The
following example shows how to use weave on basic Python objects:

.. code-block:: python

    code = r"""
    int i;
    py::tuple results(2);
    for (i=0; i<a.length(); i++) {
         a[i] = i;
    }
    results[0] = 3.0;
    results[1] = 4.0;
    return_val = results;
    """
    a = [None]*10
    res = weave.inline(code,['a'])

The C++ code shown in the code string uses the name 'a' to refer to
the Python list that is passed in. Because the Python List is a
mutable type, the elements of the list itself are modified by the C++
code. A set of C++ classes are used to access Python objects using
simple syntax.

The main advantage of using C-code, however, is to speed up processing
on an array of data. Accessing a NumPy array in C++ code using weave,
depends on what kind of type converter is chosen in going from NumPy
arrays to C++ code. The default converter creates 5 variables for the
C-code for every NumPy array passed in to weave.inline. The following
table shows these variables which can all be used in the C++ code. The
table assumes that ``myvar`` is the name of the array in Python with
data-type {dtype} (i.e.  float64, float32, int8, etc.)

===========  ==============  =========================================
Variable     Type            Contents
===========  ==============  =========================================
myvar        {dtype}*        Pointer to the first element of the array
Nmyvar       npy_intp*       A pointer to the dimensions array
Smyvar       npy_intp*       A pointer to the strides array
Dmyvar       int             The number of dimensions
myvar_array  PyArrayObject*  The entire structure for the array
===========  ==============  =========================================

The in-lined code can contain references to any of these variables as
well as to the standard macros MYVAR1(i), MYVAR2(i,j), MYVAR3(i,j,k),
and MYVAR4(i,j,k,l). These name-based macros (they are the Python name
capitalized followed by the number of dimensions needed) will de-
reference the memory for the array at the given location with no error
checking (be-sure to use the correct macro and ensure the array is
aligned and in correct byte-swap order in order to get useful
results). The following code shows how you might use these variables
and macros to code a loop in C that computes a simple 2-d weighted
averaging filter.

.. code-block:: c++

    int i,j;
    for(i=1;i<Na[0]-1;i++) {
       for(j=1;j<Na[1]-1;j++) {
           B2(i,j) = A2(i,j) + (A2(i-1,j) +
                     A2(i+1,j)+A2(i,j-1)
                     + A2(i,j+1))*0.5
                     + (A2(i-1,j-1)
                     + A2(i-1,j+1)
                     + A2(i+1,j-1)
                     + A2(i+1,j+1))*0.25
       }
    }

The above code doesn't have any error checking and so could fail with
a Python crash if, ``a`` had the wrong number of dimensions, or ``b``
did not have the same shape as ``a``. However, it could be placed
inside a standard Python function with the necessary error checking to
produce a robust but fast subroutine.

One final note about weave.inline: if you have additional code you
want to include in the final extension module such as supporting
function calls, include statements, etc. you can pass this code in as a
string using the keyword support_code: ``weave.inline(code, variables,
support_code=support)``. If you need the extension module to link
against an additional library then you can also pass in
distutils-style keyword arguments such as library_dirs, libraries,
and/or runtime_library_dirs which point to the appropriate libraries
and directories.

Simplify creation of an extension module
----------------------------------------

The inline function creates one extension module for each function to-
be inlined. It also generates a lot of intermediate code that is
duplicated for each extension module. If you have several related
codes to execute in C, it would be better to make them all separate
functions in a single extension module with multiple functions. You
can also use the tools weave provides to produce this larger extension
module. In fact, the weave.inline function just uses these more
general tools to do its work.

The approach is to:

1. construct a extension module object using
   ext_tools.ext_module(``module_name``);

2. create function objects using ext_tools.ext_function(``func_name``,
   ``code``, ``variables``);

3. (optional) add support code to the function using the
   .customize.add_support_code( ``support_code`` ) method of the
   function object;

4. add the functions to the extension module object using the
   .add_function(``func``) method;

5. when all the functions are added, compile the extension with its
   .compile() method.

Several examples are available in the examples directory where weave
is installed on your system. Look particularly at ramp2.py,
increment_example.py and fibonacii.py


Conclusion
----------

Weave is a useful tool for quickly routines in C/C++ and linking them
into Python. It's caching-mechanism allows for on-the-fly compilation
which makes it particularly attractive for in-house code. Because of
the requirement that the user have a C++-compiler, it can be difficult
(but not impossible) to distribute a package that uses weave to other
users who don't have a compiler installed. Of course, weave could be
used to construct an extension module which is then distributed in the
normal way *(* using a setup.py file). While you can use weave to
build larger extension modules with many methods, creating methods
with a variable- number of arguments is not possible. Thus, for a more
sophisticated module, you will still probably want a Python-layer that
calls the weave-produced extension.

.. index::
   single: weave


Pyrex
=====

Pyrex is a way to write C-extension modules using Python-like syntax.
It is an interesting way to generate extension modules that is growing
in popularity, particularly among people who have rusty or non-
existent C-skills. It does require the user to write the "interface"
code and so is more time-consuming than SWIG or f2py if you are trying
to interface to a large library of code. However, if you are writing
an extension module that will include quite a bit of your own
algorithmic code, as well, then Pyrex is a good match. A big weakness
perhaps is the inability to easily and quickly access the elements of
a multidimensional array.

.. index::
   single: pyrex

Notice that Pyrex is an extension-module generator only. Unlike weave
or f2py, it includes no automatic facility for compiling and linking
the extension module (which must be done in the usual fashion). It
does provide a modified distutils class called build_ext which lets
you build an extension module from a .pyx source. Thus, you could
write in a setup.py file:

.. code-block:: python

    from Pyrex.Distutils import build_ext
    from distutils.extension import Extension
    from distutils.core import setup

    import numpy
    py_ext = Extension('mine', ['mine.pyx'],
             include_dirs=[numpy.get_include()])

    setup(name='mine', description='Nothing',
          ext_modules=[pyx_ext],
          cmdclass = {'build_ext':build_ext})

Adding the NumPy include directory is, of course, only necessary if
you are using NumPy arrays in the extension module (which is what I
assume you are using Pyrex for). The distutils extensions in NumPy
also include support for automatically producing the extension-module
and linking it from a ``.pyx`` file. It works so that if the user does
not have Pyrex installed, then it looks for a file with the same
file-name but a ``.c`` extension which it then uses instead of trying
to produce the ``.c`` file again.

Pyrex does not natively understand NumPy arrays. However, it is not
difficult to include information that lets Pyrex deal with them
usefully. In fact, the numpy.random.mtrand module was written using
Pyrex so an example of Pyrex usage is already included in the NumPy
source distribution. That experience led to the creation of a standard
c_numpy.pxd file that you can use to simplify interacting with NumPy
array objects in a Pyrex-written extension. The file may not be
complete (it wasn't at the time of this writing). If you have
additions you'd like to contribute, please send them. The file is
located in the .../site-packages/numpy/doc/pyrex directory where you
have Python installed. There is also an example in that directory of
using Pyrex to construct a simple extension module. It shows that
Pyrex looks a lot like Python but also contains some new syntax that
is necessary in order to get C-like speed.

If you just use Pyrex to compile a standard Python module, then you
will get a C-extension module that runs either as fast or, possibly,
more slowly than the equivalent Python module. Speed increases are
possible only when you use cdef to statically define C variables and
use a special construct to create for loops:

.. code-block:: none

    cdef int i
    for i from start <= i < stop

Let's look at two examples we've seen before to see how they might be
implemented using Pyrex. These examples were compiled into extension
modules using Pyrex-0.9.3.1.


Pyrex-add
---------

Here is part of a Pyrex-file I named add.pyx which implements the add
functions we previously implemented using f2py:

.. code-block:: none

    cimport c_numpy
    from c_numpy cimport import_array, ndarray, npy_intp, npy_cdouble, \
         npy_cfloat, NPY_DOUBLE, NPY_CDOUBLE, NPY_FLOAT, \
         NPY_CFLOAT

    #We need to initialize NumPy
    import_array()

    def zadd(object ao, object bo):
        cdef ndarray c, a, b
        cdef npy_intp i
        a = c_numpy.PyArray_ContiguousFromAny(ao,
                      NPY_CDOUBLE, 1, 1)
        b = c_numpy.PyArray_ContiguousFromAny(bo,
                      NPY_CDOUBLE, 1, 1)
        c = c_numpy.PyArray_SimpleNew(a.nd, a.dimensions,
                     a.descr.type_num)
        for i from 0 <= i < a.dimensions[0]:
            (<npy_cdouble *>c.data)[i].real = \
                 (<npy_cdouble *>a.data)[i].real + \
                 (<npy_cdouble *>b.data)[i].real
            (<npy_cdouble *>c.data)[i].imag = \
                 (<npy_cdouble *>a.data)[i].imag + \
                 (<npy_cdouble *>b.data)[i].imag
        return c

This module shows use of the ``cimport`` statement to load the
definitions from the c_numpy.pxd file. As shown, both versions of the
import statement are supported. It also shows use of the NumPy C-API
to construct NumPy arrays from arbitrary input objects. The array c is
created using PyArray_SimpleNew. Then the c-array is filled by
addition. Casting to a particiular data-type is accomplished using
<cast \*>. Pointers are de-referenced with bracket notation and
members of structures are accessed using '.' notation even if the
object is techinically a pointer to a structure. The use of the
special for loop construct ensures that the underlying code will have
a similar C-loop so the addition calculation will proceed quickly.
Notice that we have not checked for NULL after calling to the C-API
--- a cardinal sin when writing C-code. For routines that return
Python objects, Pyrex inserts the checks for NULL into the C-code for
you and returns with failure if need be. There is also a way to get
Pyrex to automatically check for exceptions when you call functions
that don't return Python objects. See the documentation of Pyrex for
details.


Pyrex-filter
------------

The two-dimensional example we created using weave is a bit uglier to
implement in Pyrex because two-dimensional indexing using Pyrex is not
as simple. But, it is straightforward (and possibly faster because of
pre-computed indices). Here is the Pyrex-file I named image.pyx.

.. code-block:: none

    cimport c_numpy
    from c_numpy cimport import_array, ndarray, npy_intp,\
         NPY_DOUBLE, NPY_CDOUBLE, \
         NPY_FLOAT, NPY_CFLOAT, NPY_ALIGNED \

    #We need to initialize NumPy
    import_array()
    def filter(object ao):
        cdef ndarray a, b
        cdef npy_intp i, j, M, N, oS
        cdef npy_intp r,rm1,rp1,c,cm1,cp1
        cdef double value
        # Require an ALIGNED array
        # (but not necessarily contiguous)
        #  We will use strides to access the elements.
        a = c_numpy.PyArray_FROMANY(ao, NPY_DOUBLE, \
                    2, 2, NPY_ALIGNED)
        b = c_numpy.PyArray_SimpleNew(a.nd,a.dimensions, \
                                      a.descr.type_num)
        M = a.dimensions[0]
        N = a.dimensions[1]
        S0 = a.strides[0]
        S1 = a.strides[1]
        for i from 1 <= i < M-1:
            r = i*S0
            rm1 = r-S0
            rp1 = r+S0
            oS = i*N
            for j from 1 <= j < N-1:
                c = j*S1
                cm1 = c-S1
                cp1 = c+S1
                (<double *>b.data)[oS+j] = \
                   (<double *>(a.data+r+c))[0] + \
                   ((<double *>(a.data+rm1+c))[0] + \
                    (<double *>(a.data+rp1+c))[0] + \
                    (<double *>(a.data+r+cm1))[0] + \
                    (<double *>(a.data+r+cp1))[0])*0.5 + \
                   ((<double *>(a.data+rm1+cm1))[0] + \
                    (<double *>(a.data+rp1+cm1))[0] + \
                    (<double *>(a.data+rp1+cp1))[0] + \
                    (<double *>(a.data+rm1+cp1))[0])*0.25
        return b

This 2-d averaging filter runs quickly because the loop is in C and
the pointer computations are done only as needed. However, it is not
particularly easy to understand what is happening. A 2-d image, ``in``
, can be filtered using this code very quickly using:

.. code-block:: python

    import image
    out = image.filter(in)


Conclusion
----------

There are several disadvantages of using Pyrex:

1. The syntax for Pyrex can get a bit bulky, and it can be confusing at
   first to understand what kind of objects you are getting and how to
   interface them with C-like constructs.

2. Inappropriate Pyrex syntax or incorrect calls to C-code or type-
   mismatches can result in failures such as

    1. Pyrex failing to generate the extension module source code,

    2. Compiler failure while generating the extension module binary due to
       incorrect C syntax,

    3. Python failure when trying to use the module.


3. It is easy to lose a clean separation between Python and C which makes
   re-using your C-code for other non-Python-related projects more
   difficult.

4. Multi-dimensional arrays are "bulky" to index (appropriate macros
   may be able to fix this).

5. The C-code generated by Pyrex is hard to read and modify (and typically
   compiles with annoying but harmless warnings).

Writing a good Pyrex extension module still takes a bit of effort
because not only does it require (a little) familiarity with C, but
also with Pyrex's brand of Python-mixed-with C. One big advantage of
Pyrex-generated extension modules is that they are easy to distribute
using distutils. In summary, Pyrex is a very capable tool for either
gluing C-code or generating an extension module quickly and should not
be over-looked. It is especially useful for people that can't or won't
write C-code or Fortran code. But, if you are already able to write
simple subroutines in C or Fortran, then I would use one of the other
approaches such as f2py (for Fortran), ctypes (for C shared-
libraries), or weave (for inline C-code).

.. index::
   single: pyrex




ctypes
======

Ctypes is a python extension module (downloaded separately for Python
<2.5 and included with Python 2.5) that allows you to call an
arbitrary function in a shared library directly from Python. This
approach allows you to interface with C-code directly from Python.
This opens up an enormous number of libraries for use from Python. The
drawback, however, is that coding mistakes can lead to ugly program
crashes very easily (just as can happen in C) because there is little
type or bounds checking done on the parameters. This is especially
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
extension-module interface. However, this overhead should be neglible
if the C-routine being called is doing any significant amount of work.
If you are a great Python programmer with weak C-skills, ctypes is an
easy way to write a useful interface to a (shared) library of compiled
code.

To use c-types you must

1. Have a shared library.

2. Load the shared library.

3. Convert the python objects to ctypes-understood arguments.

4. Call the function from the library with the ctypes arguments.


Having a shared library
-----------------------

There are several requirements for a shared library that can be used
with c-types that are platform specific. This guide assumes you have
some familiarity with making a shared library on your system (or
simply have a shared library available to you). Items to remember are:

- A shared library must be compiled in a special way ( *e.g.* using
  the -shared flag with gcc).

- On some platforms (*e.g.* Windows) , a shared library requires a
  .def file that specifies the functions to be exported. For example a
  mylib.def file might contain.

  ::

      LIBRARY mylib.dll
      EXPORTS
      cool_function1
      cool_function2

  Alternatively, you may be able to use the storage-class specifier
  __declspec(dllexport) in the C-definition of the function to avoid the
  need for this .def file.

There is no standard way in Python distutils to create a standard
shared library (an extension module is a "special" shared library
Python understands) in a cross-platform manner. Thus, a big
disadvantage of ctypes at the time of writing this book is that it is
difficult to distribute in a cross-platform manner a Python extension
that uses c-types and includes your own code which should be compiled
as a shared library on the users system.


Loading the shared library
--------------------------

A simple, but robust way to load the shared library is to get the
absolute path name and load it using the cdll object of ctypes.:

.. code-block:: python

    lib = ctypes.cdll[<full_path_name>]

However, on Windows accessing an attribute of the cdll method will
load the first DLL by that name found in the current directory or on
the PATH. Loading the absolute path name requires a little finesse for
cross-platform work since the extension of shared libraries varies.
There is a ``ctypes.util.find_library`` utility available that can
simplify the process of finding the library to load but it is not
foolproof. Complicating matters, different platforms have different
default extensions used by shared libraries (e.g. .dll -- Windows, .so
-- Linux, .dylib -- Mac OS X). This must also be taken into account if
you are using c-types to wrap code that needs to work on several
platforms.

NumPy provides a convenience function called
:func:`ctypeslib.load_library` (name, path). This function takes the name
of the shared library (including any prefix like 'lib' but excluding
the extension) and a path where the shared library can be located. It
returns a ctypes library object or raises an OSError if the library
cannot be found or raises an ImportError if the ctypes module is not
available. (Windows users: the ctypes library object loaded using
:func:`load_library` is always loaded assuming cdecl calling convention.
See the ctypes documentation under ctypes.windll and/or ctypes.oledll
for ways to load libraries under other calling conventions).

The functions in the shared library are available as attributes of the
ctypes library object (returned from :func:`ctypeslib.load_library`) or
as items using ``lib['func_name']`` syntax. The latter method for
retrieving a function name is particularly useful if the function name
contains characters that are not allowable in Python variable names.


Converting arguments
--------------------

Python ints/longs, strings, and unicode objects are automatically
converted as needed to equivalent c-types arguments The None object is
also converted automatically to a NULL pointer. All other Python
objects must be converted to ctypes-specific types. There are two ways
around this restriction that allow c-types to integrate with other
objects.

1. Don't set the argtypes attribute of the function object and define an
   :obj:`_as_parameter_` method for the object you want to pass in. The
   :obj:`_as_parameter_` method must return a Python int which will be passed
   directly to the function.

2. Set the argtypes attribute to a list whose entries contain objects
   with a classmethod named from_param that knows how to convert your
   object to an object that ctypes can understand (an int/long, string,
   unicode, or object with the :obj:`_as_parameter_` attribute).

NumPy uses both methods with a preference for the second method
because it can be safer. The ctypes attribute of the ndarray returns
an object that has an _as_parameter\_ attribute which returns an
integer representing the address of the ndarray to which it is
associated. As a result, one can pass this ctypes attribute object
directly to a function expecting a pointer to the data in your
ndarray. The caller must be sure that the ndarray object is of the
correct type, shape, and has the correct flags set or risk nasty
crashes if the data-pointer to inappropriate arrays are passsed in.

To implement the second method, NumPy provides the class-factory
function :func:`ndpointer` in the :mod:`ctypeslib` module. This
class-factory function produces an appropriate class that can be
placed in an argtypes attribute entry of a ctypes function. The class
will contain a from_param method which ctypes will use to convert any
ndarray passed in to the function to a ctypes-recognized object. In
the process, the conversion will perform checking on any properties of
the ndarray that were specified by the user in the call to :func:`ndpointer`.
Aspects of the ndarray that can be checked include the data-type, the
number-of-dimensions, the shape, and/or the state of the flags on any
array passed. The return value of the from_param method is the ctypes
attribute of the array which (because it contains the _as_parameter\_
attribute pointing to the array data area) can be used by ctypes
directly.

The ctypes attribute of an ndarray is also endowed with additional
attributes that may be convenient when passing additional information
about the array into a ctypes function. The attributes **data**,
**shape**, and **strides** can provide c-types compatible types
corresponding to the data-area, the shape, and the strides of the
array. The data attribute reutrns a ``c_void_p`` representing a
pointer to the data area. The shape and strides attributes each return
an array of ctypes integers (or None representing a NULL pointer, if a
0-d array). The base ctype of the array is a ctype integer of the same
size as a pointer on the platform. There are also methods
data_as({ctype}), shape_as(<base ctype>), and strides_as(<base
ctype>). These return the data as a ctype object of your choice and
the shape/strides arrays using an underlying base type of your choice.
For convenience, the **ctypeslib** module also contains **c_intp** as
a ctypes integer data-type whose size is the same as the size of
``c_void_p`` on the platform (it's value is None if ctypes is not
installed).


Calling the function
--------------------

The function is accessed as an attribute of or an item from the loaded
shared-library. Thus, if "./mylib.so" has a function named
"cool_function1" , I could access this function either as:

.. code-block:: python

    lib = numpy.ctypeslib.load_library('mylib','.')
    func1 = lib.cool_function1 # or equivalently
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
significantly safer to call a C-function using ctypes and the data-
area of an ndarray. You may still want to wrap the function in an
additional Python wrapper to make it user-friendly (hiding some
obvious arguments and making some arguments output arguments). In this
process, the **requires** function in NumPy may be useful to return the right
kind of array from a given input.


Complete example
----------------

In this example, I will show how the addition function and the filter
function implemented previously using the other approaches can be
implemented using ctypes. First, the C-code which implements the
algorithms contains the functions zadd, dadd, sadd, cadd, and
dfilter2d. The zadd function is:

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

with similar code for cadd, dadd, and sadd that handles complex float,
double, and float data-types, respectively:

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

The code.c file also contains the function dfilter2d:

.. code-block:: c

    /* Assumes b is contiguous and
       a has strides that are multiples of sizeof(double)
    */
    void
    dfilter2d(double *a, double *b, int *astrides, int *dims)
    {
        int i, j, M, N, S0, S1;
        int r, c, rm1, rp1, cp1, cm1;

        M = dims[0]; N = dims[1];
        S0 = astrides[0]/sizeof(double);
        S1=astrides[1]/sizeof(double);
        for (i=1; i<M-1; i++) {
            r = i*S0; rp1 = r+S0; rm1 = r-S0;
            for (j=1; j<N-1; j++) {
                c = j*S1; cp1 = j+S1; cm1 = j-S1;
                b[i*N+j] = a[r+c] +                 \
                    (a[rp1+c] + a[rm1+c] +          \
                     a[r+cp1] + a[r+cm1])*0.5 +     \
                    (a[rp1+cp1] + a[rp1+cm1] +      \
                     a[rm1+cp1] + a[rm1+cp1])*0.25;
            }
        }
    }

A possible advantage this code has over the Fortran-equivalent code is
that it takes arbitrarily strided (i.e. non-contiguous arrays) and may
also run faster depending on the optimization capability of your
compiler. But, it is a obviously more complicated than the simple code
in filter.f. This code must be compiled into a shared library. On my
Linux system this is accomplished using::

    gcc -o code.so -shared code.c

Which creates a shared_library named code.so in the current directory.
On Windows don't forget to either add __declspec(dllexport) in front
of void on the line preceeding each function definition, or write a
code.def file that lists the names of the functions to be exported.

A suitable Python interface to this shared library should be
constructed. To do this create a file named interface.py with the
following lines at the top:

.. code-block:: python

    __all__ = ['add', 'filter2d']

    import numpy as N
    import os

    _path = os.path.dirname('__file__')
    lib = N.ctypeslib.load_library('code', _path)
    _typedict = {'zadd' : complex, 'sadd' : N.single,
                 'cadd' : N.csingle, 'dadd' : float}
    for name in _typedict.keys():
        val = getattr(lib, name)
        val.restype = None
        _type = _typedict[name]
        val.argtypes = [N.ctypeslib.ndpointer(_type,
                          flags='aligned, contiguous'),
                        N.ctypeslib.ndpointer(_type,
                          flags='aligned, contiguous'),
                        N.ctypeslib.ndpointer(_type,
                          flags='aligned, contiguous,'\
                                'writeable'),
                        N.ctypeslib.c_intp]

This code loads the shared library named code.{ext} located in the
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
    lib.dfilter2d.argtypes = [N.ctypeslib.ndpointer(float, ndim=2,
                                           flags='aligned'),
                              N.ctypeslib.ndpointer(float, ndim=2,
                                     flags='aligned, contiguous,'\
                                           'writeable'),
                              ctypes.POINTER(N.ctypeslib.c_intp),
                              ctypes.POINTER(N.ctypeslib.c_intp)]

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
        a = N.asanyarray(a)
        func, dtype = select(a.dtype)
        a = N.require(a, dtype, requires)
        b = N.require(b, dtype, requires)
        c = N.empty_like(a)
        func(a,b,c,a.size)
        return c

and:

.. code-block:: python

    def filter2d(a):
        a = N.require(a, float, ['ALIGNED'])
        b = N.zeros_like(a)
        lib.dfilter2d(a, b, a.ctypes.strides, a.ctypes.shape)
        return b


Conclusion
----------

.. index::
   single: ctypes

Using ctypes is a powerful way to connect Python with arbitrary
C-code. It's advantages for extending Python include

- clean separation of C-code from Python code

    - no need to learn a new syntax except Python and C

    - allows re-use of C-code

    - functionality in shared libraries written for other purposes can be
      obtained with a simple Python wrapper and search for the library.


- easy integration with NumPy through the ctypes attribute

- full argument checking with the ndpointer class factory

It's disadvantages include

- It is difficult to distribute an extension module made using ctypes
  because of a lack of support for building shared libraries in
  distutils (but I suspect this will change in time).

- You must have shared-libraries of your code (no static libraries).

- Very little support for C++ code and it's different library-calling
  conventions. You will probably need a C-wrapper around C++ code to use
  with ctypes (or just use Boost.Python instead).

Because of the difficulty in distributing an extension module made
using ctypes, f2py is still the easiest way to extend Python for
package creation. However, ctypes is a close second and will probably
be growing in popularity now that it is part of the Python
distribution. This should bring more features to ctypes that should
eliminate the difficulty in extending Python and distributing the
extension using ctypes.


Additional tools you may find useful
====================================

These tools have been found useful by others using Python and so are
included here. They are discussed separately because I see them as
either older ways to do things more modernly handled by f2py, weave,
Pyrex, or ctypes (SWIG, PyFort, PyInline) or because I don't know much
about them (SIP, Boost, Instant). I have not added links to these
methods because my experience is that you can find the most relevant
link faster using Google or some other search engine, and any links
provided here would be quickly dated. Do not assume that just because
it is included in this list, I don't think the package deserves your
attention. I'm including information about these packages because many
people have found them useful and I'd like to give you as many options
as possible for tackling the problem of easily integrating your code.


SWIG
----

.. index::
   single: swig

Simplified Wrapper and Interface Generator (SWIG) is an old and fairly
stable method for wrapping C/C++-libraries to a large variety of other
languages. It does not specifically understand NumPy arrays but can be
made useable with NumPy through the use of typemaps. There are some
sample typemaps in the numpy/doc/swig directory under numpy.i along
with an example module that makes use of them. SWIG excels at wrapping
large C/C++ libraries because it can (almost) parse their headers and
auto-produce an interface. Technically, you need to generate a ``.i``
file that defines the interface. Often, however, this ``.i`` file can
be parts of the header itself. The interface usually needs a bit of
tweaking to be very useful. This ability to parse C/C++ headers and
auto-generate the interface still makes SWIG a useful approach to
adding functionalilty from C/C++ into Python, despite the other
methods that have emerged that are more targeted to Python. SWIG can
actually target extensions for several languages, but the typemaps
usually have to be language-specific. Nonetheless, with modifications
to the Python-specific typemaps, SWIG can be used to interface a
library with other languages such as Perl, Tcl, and Ruby.

My experience with SWIG has been generally positive in that it is
relatively easy to use and quite powerful. I used to use it quite
often before becoming more proficient at writing C-extensions.
However, I struggled writing custom interfaces with SWIG because it
must be done using the concept of typemaps which are not Python
specific and are written in a C-like syntax. Therefore, I tend to
prefer other gluing strategies and would only attempt to use SWIG to
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
mappings between Python types and C/C++ structrues and classes.


Boost Python
------------

.. index::
   single: Boost.Python

Boost is a repository of C++ libraries and Boost.Python is one of
those libraries which provides a concise interface for binding C++
classes and functions to Python. The amazing part of the Boost.Python
approach is that it works entirely in pure C++ without introducing a
new syntax. Many users of C++ report that Boost.Python makes it
possible to combine the best of both worlds in a seamless fashion. I
have not used Boost.Python because I am not a big user of C++ and
using Boost to wrap simple C-subroutines is usually over-kill. It's
primary purpose is to make C++ classes available in Python. So, if you
have a set of C++ classes that need to be integrated cleanly into
Python, consider learning about and using Boost.Python.


Instant
-------

.. index::
   single: Instant

This is a relatively new package (called pyinstant at sourceforge)
that builds on top of SWIG to make it easy to inline C and C++ code in
Python very much like weave. However, Instant builds extension modules
on the fly with specific module names and specific method names. In
this repsect it is more more like f2py in its behavior. The extension
modules are built on-the fly (as long as the SWIG is installed). They
can then be imported. Here is an example of using Instant with NumPy
arrays (adapted from the test2 included in the Instant distribution):

.. code-block:: python

    code="""
    PyObject* add(PyObject* a_, PyObject* b_){
      /*
      various checks
      */
      PyArrayObject* a=(PyArrayObject*) a_;
      PyArrayObject* b=(PyArrayObject*) b_;
      int n = a->dimensions[0];
      int dims[1];
      dims[0] = n;
      PyArrayObject* ret;
      ret = (PyArrayObject*) PyArray_FromDims(1, dims, NPY_DOUBLE);
      int i;
      char *aj=a->data;
      char *bj=b->data;
      double *retj = (double *)ret->data;
      for (i=0; i < n; i++) {
        *retj++ = *((double *)aj) + *((double *)bj);
        aj += a->strides[0];
        bj += b->strides[0];
      }
    return (PyObject *)ret;
    }
    """
    import Instant, numpy
    ext = Instant.Instant()
    ext.create_extension(code=s, headers=["numpy/arrayobject.h"],
                         include_dirs=[numpy.get_include()],
                         init_code='import_array();', module="test2b_ext")
    import test2b_ext
    a = numpy.arange(1000)
    b = numpy.arange(1000)
    d = test2b_ext.add(a,b)

Except perhaps for the dependence on SWIG, Instant is a
straightforward utility for writing extension modules.


PyInline
--------

This is a much older module that allows automatic building of
extension modules so that C-code can be included with Python code.
It's latest release (version 0.03) was in 2001, and it appears that it
is not being updated.


PyFort
------

PyFort is a nice tool for wrapping Fortran and Fortran-like C-code
into Python with support for Numeric arrays. It was written by Paul
Dubois, a distinguished computer scientist and the very first
maintainer of Numeric (now retired). It is worth mentioning in the
hopes that somebody will update PyFort to work with NumPy arrays as
well which now support either Fortran or C-style contiguous arrays.
