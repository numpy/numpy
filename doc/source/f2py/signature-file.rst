==================
 Signature file
==================

The syntax specification for signature files (.pyf files) is borrowed
from the Fortran 90/95 language specification. Almost all Fortran
90/95 standard constructs are understood, both in free and fixed
format (recall that Fortran 77 is a subset of Fortran 90/95). F2PY
introduces also some extensions to Fortran 90/95 language
specification that help designing Fortran to Python interface, make it
more "Pythonic".

Signature files may contain arbitrary Fortran code (so that Fortran
codes can be considered as signature files). F2PY silently ignores
Fortran constructs that are irrelevant for creating the interface.
However, this includes also syntax errors. So, be careful not making
ones;-).

In general, the contents of signature files is case-sensitive.  When
scanning Fortran codes and writing a signature file, F2PY lowers all
cases automatically except in multiline blocks or when ``--no-lower``
option is used.

The syntax of signature files is presented below.

Python module block
=====================

A signature file may contain one (recommended) or more ``python
module`` blocks.  ``python module`` block describes the contents of
a Python/C extension module ``<modulename>module.c`` that F2PY
generates.

Exception: if ``<modulename>`` contains a substring ``__user__``, then
the corresponding ``python module`` block describes the signatures of
so-called call-back functions (see :ref:`Call-back arguments`).

A ``python module`` block has the following structure::

  python module <modulename>
    [<usercode statement>]...
    [
    interface
      <usercode statement>
      <Fortran block data signatures>
      <Fortran/C routine signatures>
    end [interface]
    ]...
    [
    interface
      module <F90 modulename>
        [<F90 module data type declarations>]
        [<F90 module routine signatures>]
      end [module [<F90 modulename>]]
    end [interface]
    ]...
  end [python module [<modulename>]]

Here brackets ``[]`` indicate an optional part, dots ``...`` indicate
one or more of a previous part. So, ``[]...`` reads zero or more of a
previous part.


Fortran/C routine signatures
=============================

The signature of a Fortran routine has the following structure::

  [<typespec>] function | subroutine <routine name> \
                [ ( [<arguments>] ) ] [ result ( <entityname> ) ]
    [<argument/variable type declarations>]
    [<argument/variable attribute statements>]
    [<use statements>]
    [<common block statements>]
    [<other statements>]
  end [ function | subroutine [<routine name>] ]

From a Fortran routine signature F2PY generates a Python/C extension
function that has the following signature::

  def <routine name>(<required arguments>[,<optional arguments>]):
       ...
       return <return variables>

The signature of a Fortran block data has the following structure::

  block data [ <block data name> ]
    [<variable type declarations>]
    [<variable attribute statements>]
    [<use statements>]
    [<common block statements>]
    [<include statements>]
  end [ block data [<block data name>] ]

Type declarations
-----------------

The definition of the ``<argument/variable type declaration>`` part
is

::

  <typespec> [ [<attrspec>] :: ] <entitydecl>

where

::

  <typespec> := byte | character [<charselector>]
             | complex [<kindselector>] | real [<kindselector>]
             | double complex | double precision
             | integer [<kindselector>] | logical [<kindselector>]

  <charselector> := * <charlen>
                 | ( [len=] <len> [ , [kind=] <kind>] )
                 | ( kind= <kind> [ , len= <len> ] )
  <kindselector> := * <intlen> | ( [kind=] <kind> )

  <entitydecl> := <name> [ [ * <charlen> ] [ ( <arrayspec> ) ]
                        | [ ( <arrayspec> ) ] * <charlen> ]
                       | [ / <init_expr> / | = <init_expr> ] \
                         [ , <entitydecl> ]

and

+ ``<attrspec>`` is a comma separated list of attributes_;

+ ``<arrayspec>`` is a comma separated list of dimension bounds;

+ ``<init_expr>`` is a `C expression`__.

+ ``<intlen>`` may be negative integer for ``integer`` type
  specifications. In such cases ``integer*<negintlen>`` represents
  unsigned C integers.

__ `C expressions`_

If an argument has no ``<argument type declaration>``, its type is
determined by applying ``implicit`` rules to its name.


Statements
----------

Attribute statements:
  The ``<argument/variable attribute statement>`` is
  ``<argument/variable type declaration>`` without ``<typespec>``.
  In addition, in an attribute statement one cannot use other
  attributes, also ``<entitydecl>`` can be only a list of names.

Use statements:
  The definition of the ``<use statement>`` part is

  ::

    use <modulename> [ , <rename_list> | , ONLY : <only_list> ]

  where

  ::

     <rename_list> := <local_name> => <use_name> [ , <rename_list> ]

  Currently F2PY uses ``use`` statement only for linking call-back
  modules and ``external`` arguments (call-back functions), see
  :ref:`Call-back arguments`.

Common block statements:
  The definition of the ``<common block statement>`` part is

  ::

    common / <common name> / <shortentitydecl>

  where

  ::

    <shortentitydecl> := <name> [ ( <arrayspec> ) ] [ , <shortentitydecl> ]

  If a ``python module`` block contains two or more ``common`` blocks
  with the same name, the variables from the additional declarations
  are appended.  The types of variables in ``<shortentitydecl>`` are
  defined using ``<argument type declarations>``. Note that the
  corresponding ``<argument type declarations>`` may contain array
  specifications; then you don't need to specify these in
  ``<shortentitydecl>``.

Other statements:
  The ``<other statement>`` part refers to any other Fortran language
  constructs that are not described above. F2PY ignores most of them
  except

  + ``call`` statements and function calls of ``external`` arguments
    (`more details`__?);

    __ external_

  + ``include`` statements
      ::

        include '<filename>'
        include "<filename>"

      If a file ``<filename>`` does not exist, the ``include``
      statement is ignored. Otherwise, the file ``<filename>`` is
      included to a signature file.  ``include`` statements can be used
      in any part of a signature file, also outside the Fortran/C
      routine signature blocks.

  + ``implicit`` statements
      ::

        implicit none
	implicit <list of implicit maps>

      where

      ::

        <implicit map> := <typespec> ( <list of letters or range of letters> )

      Implicit rules are used to determine the type specification of
      a variable (from the first-letter of its name) if the variable
      is not defined using ``<variable type declaration>``.  Default
      implicit rule is given by

      ::

        implicit real (a-h,o-z,$_), integer (i-m)

  + ``entry`` statements
      ::

        entry <entry name> [([<arguments>])]

      F2PY generates wrappers to all entry names using the signature
      of the routine block.

      Tip: ``entry`` statement can be used to describe the signature
      of an arbitrary routine allowing F2PY to generate a number of
      wrappers from only one routine block signature. There are few
      restrictions while doing this: ``fortranname`` cannot be used,
      ``callstatement`` and ``callprotoargument`` can be used only if
      they are valid for all entry routines, etc.

  In addition, F2PY introduces the following statements:

  + ``threadsafe``
      Use ``Py_BEGIN_ALLOW_THREADS .. Py_END_ALLOW_THREADS`` block
      around the call to Fortran/C function.

  + ``callstatement <C-expr|multi-line block>``
      Replace F2PY generated call statement to Fortran/C function with
      ``<C-expr|multi-line block>``. The wrapped Fortran/C function
      is available as ``(*f2py_func)``. To raise an exception, set
      ``f2py_success = 0`` in ``<C-expr|multi-line block>``.

  + ``callprotoargument <C-typespecs>``
      When ``callstatement`` statement is used then F2PY may not
      generate proper prototypes for Fortran/C functions (because
      ``<C-expr>`` may contain any function calls and F2PY has no way
      to determine what should be the proper prototype). With this
      statement you can explicitly specify the arguments of the
      corresponding prototype::

        extern <return type> FUNC_F(<routine name>,<ROUTINE NAME>)(<callprotoargument>);

  + ``fortranname [<actual Fortran/C routine name>]``
      You can use arbitrary ``<routine name>`` for a given Fortran/C
      function. Then you have to specify
      ``<actual Fortran/C routine name>`` with this statement.

      If ``fortranname`` statement is used without
      ``<actual Fortran/C routine name>`` then a dummy wrapper is
      generated.

  + ``usercode <multi-line block>``
      When used inside ``python module`` block, then given C code
      will be inserted to generated C/API source just before
      wrapper function definitions. Here you can define arbitrary
      C functions to be used in initialization of optional arguments,
      for example. If ``usercode`` is used twice inside ``python
      module`` block then the second multiline block is inserted
      after the definition of external routines.

      When used inside ``<routine signature>``, then given C code will
      be inserted to the corresponding wrapper function just after
      declaring variables but before any C statements. So, ``usercode``
      follow-up can contain both declarations and C statements.

      When used inside the first ``interface`` block, then given C
      code will be inserted at the end of the initialization
      function of the extension module. Here you can modify extension
      modules dictionary. For example, for defining additional
      variables etc.

  + ``pymethoddef <multiline block>``
      Multiline block will be inserted to the definition of
      module methods ``PyMethodDef``-array. It must be a
      comma-separated list of C arrays (see `Extending and Embedding`__
      Python documentation for details).
      ``pymethoddef`` statement can be used only inside
      ``python module`` block.

  __ http://www.python.org/doc/current/ext/ext.html

Attributes
------------

The following attributes are used by F2PY:

``optional``
  The corresponding argument is moved to the end of ``<optional
  arguments>`` list. A default value for an optional argument can be
  specified ``<init_expr>``, see ``entitydecl`` definition. Note that
  the default value must be given as a valid C expression.

  Note that whenever ``<init_expr>`` is used, ``optional`` attribute
  is set automatically by F2PY.

  For an optional array argument, all its dimensions must be bounded.

``required``
  The corresponding argument is considered as a required one. This is
  default. You need to specify ``required`` only if there is a need to
  disable automatic ``optional`` setting when ``<init_expr>`` is used.

  If Python ``None`` object is used as a required argument, the
  argument is treated as optional. That is, in the case of array
  argument, the memory is allocated. And if ``<init_expr>`` is given,
  the corresponding initialization is carried out.

``dimension(<arrayspec>)``
  The corresponding variable is considered as an array with given
  dimensions in ``<arrayspec>``.

``intent(<intentspec>)``
  This specifies the "intention" of the corresponding
  argument. ``<intentspec>`` is a comma separated list of the
  following keys:

  + ``in``
      The argument is considered as an input-only argument. It means
      that the value of the argument is passed to Fortran/C function and
      that function is expected not to change the value of an argument.

  + ``inout``
      The argument is considered as an input/output or *in situ*
      output argument. ``intent(inout)`` arguments can be only
      "contiguous" NumPy arrays with proper type and size.  Here
      "contiguous" can be either in Fortran or C sense. The latter one
      coincides with the contiguous concept used in NumPy and is
      effective only if ``intent(c)`` is used. Fortran contiguity
      is assumed by default.

      Using ``intent(inout)`` is generally not recommended, use
      ``intent(in,out)`` instead. See also ``intent(inplace)`` attribute.

  + ``inplace``
      The argument is considered as an input/output or *in situ*
      output argument. ``intent(inplace)`` arguments must be
      NumPy arrays with proper size. If the type of an array is
      not "proper" or the array is non-contiguous then the array
      will be changed in-place to fix the type and make it contiguous.

      Using ``intent(inplace)`` is generally not recommended either.
      For example, when slices have been taken from an
      ``intent(inplace)`` argument then after in-place changes,
      slices data pointers may point to unallocated memory area.

  + ``out``
      The argument is considered as a return variable. It is appended
      to the ``<returned variables>`` list. Using ``intent(out)``
      sets ``intent(hide)`` automatically, unless also
      ``intent(in)`` or ``intent(inout)`` were used.

      By default, returned multidimensional arrays are
      Fortran-contiguous. If ``intent(c)`` is used, then returned
      multidimensional arrays are C-contiguous.

  + ``hide``
      The argument is removed from the list of required or optional
      arguments. Typically ``intent(hide)`` is used with ``intent(out)``
      or when ``<init_expr>`` completely determines the value of the
      argument like in the following example::

        integer intent(hide),depend(a) :: n = len(a)
        real intent(in),dimension(n) :: a

  + ``c``
      The argument is treated as a C scalar or C array argument.  In
      the case of a scalar argument, its value is passed to C function
      as a C scalar argument (recall that Fortran scalar arguments are
      actually C pointer arguments).  In the case of an array
      argument, the wrapper function is assumed to treat
      multidimensional arrays as C-contiguous arrays.

      There is no need to use ``intent(c)`` for one-dimensional
      arrays, no matter if the wrapped function is either a Fortran or
      a C function. This is because the concepts of Fortran- and
      C contiguity overlap in one-dimensional cases.

      If ``intent(c)`` is used as a statement but without an entity
      declaration list, then F2PY adds the ``intent(c)`` attribute to all
      arguments.

      Also, when wrapping C functions, one must use ``intent(c)``
      attribute for ``<routine name>`` in order to disable Fortran
      specific ``F_FUNC(..,..)`` macros.

  + ``cache``
      The argument is treated as a junk of memory. No Fortran nor C
      contiguity checks are carried out. Using ``intent(cache)``
      makes sense only for array arguments, also in connection with
      ``intent(hide)`` or ``optional`` attributes.

  + ``copy``
      Ensure that the original contents of ``intent(in)`` argument is
      preserved. Typically used in connection with ``intent(in,out)``
      attribute.  F2PY creates an optional argument
      ``overwrite_<argument name>`` with the default value ``0``.

  + ``overwrite``
      The original contents of the ``intent(in)`` argument may be
      altered by the Fortran/C function.  F2PY creates an optional
      argument ``overwrite_<argument name>`` with the default value
      ``1``.

  + ``out=<new name>``
      Replace the return name with ``<new name>`` in the ``__doc__``
      string of a wrapper function.

  + ``callback``
      Construct an external function suitable for calling Python function
      from Fortran. ``intent(callback)`` must be specified before the
      corresponding ``external`` statement. If 'argument' is not in
      argument list then it will be added to Python wrapper but only
      initializing external function.

      Use ``intent(callback)`` in situations where a Fortran/C code
      assumes that a user implements a function with given prototype
      and links it to an executable. Don't use ``intent(callback)``
      if function appears in the argument list of a Fortran routine.

      With ``intent(hide)`` or ``optional`` attributes specified and
      using a wrapper function without specifying the callback argument
      in argument list then call-back function is looked in the
      namespace of F2PY generated extension module where it can be
      set as a module attribute by a user.

  + ``aux``
      Define auxiliary C variable in F2PY generated wrapper function.
      Useful to save parameter values so that they can be accessed
      in initialization expression of other variables. Note that
      ``intent(aux)`` silently implies ``intent(c)``.

  The following rules apply:

  + If no ``intent(in | inout | out | hide)`` is specified,
    ``intent(in)`` is assumed.
  + ``intent(in,inout)`` is ``intent(in)``.
  + ``intent(in,hide)`` or ``intent(inout,hide)`` is
    ``intent(hide)``.
  + ``intent(out)`` is ``intent(out,hide)`` unless ``intent(in)`` or
    ``intent(inout)`` is specified.
  + If ``intent(copy)`` or ``intent(overwrite)`` is used, then an
    additional optional argument is introduced with a name
    ``overwrite_<argument name>`` and a default value 0 or 1, respectively.
  + ``intent(inout,inplace)`` is ``intent(inplace)``.
  + ``intent(in,inplace)`` is ``intent(inplace)``.
  + ``intent(hide)`` disables ``optional`` and ``required``.

``check([<C-booleanexpr>])``
  Perform consistency check of arguments by evaluating
  ``<C-booleanexpr>``; if ``<C-booleanexpr>`` returns 0, an exception
  is raised.

  If ``check(..)`` is not used then F2PY generates few standard checks
  (e.g. in a case of an array argument, check for the proper shape
  and size) automatically. Use ``check()`` to disable checks generated
  by F2PY.

``depend([<names>])``
  This declares that the corresponding argument depends on the values
  of variables in the list ``<names>``. For example, ``<init_expr>``
  may use the values of other arguments.  Using information given by
  ``depend(..)`` attributes, F2PY ensures that arguments are
  initialized in a proper order. If ``depend(..)`` attribute is not
  used then F2PY determines dependence relations automatically. Use
  ``depend()`` to disable dependence relations generated by F2PY.

  When you edit dependence relations that were initially generated by
  F2PY, be careful not to break the dependence relations of other
  relevant variables. Another thing to watch out is cyclic
  dependencies. F2PY is able to detect cyclic dependencies
  when constructing wrappers and it complains if any are found.

``allocatable``
  The corresponding variable is Fortran 90 allocatable array defined
  as Fortran 90 module data.

.. _external:

``external``
  The corresponding argument is a function provided by user. The
  signature of this so-called call-back function can be defined

  - in ``__user__`` module block,
  - or by demonstrative (or real, if the signature file is a real Fortran
    code) call in the ``<other statements>`` block.

  For example, F2PY generates from

  ::

    external cb_sub, cb_fun
    integer n
    real a(n),r
    call cb_sub(a,n)
    r = cb_fun(4)

  the following call-back signatures::

    subroutine cb_sub(a,n)
        real dimension(n) :: a
        integer optional,check(len(a)>=n),depend(a) :: n=len(a)
    end subroutine cb_sub
    function cb_fun(e_4_e) result (r)
        integer :: e_4_e
        real :: r
    end function cb_fun

  The corresponding user-provided Python function are then::

    def cb_sub(a,[n]):
        ...
        return
    def cb_fun(e_4_e):
        ...
        return r

  See also ``intent(callback)`` attribute.

``parameter``
  The corresponding variable is a parameter and it must have a fixed
  value. F2PY replaces all parameter occurrences by their
  corresponding values.

Extensions
============

F2PY directives
-----------------

The so-called F2PY directives allow using F2PY signature file
constructs also in Fortran 77/90 source codes. With this feature you
can skip (almost) completely intermediate signature file generations
and apply F2PY directly to Fortran source codes.

F2PY directive has the following form::

  <comment char>f2py ...

where allowed comment characters for fixed and free format Fortran
codes are ``cC*!#`` and ``!``, respectively. Everything that follows
``<comment char>f2py`` is ignored by a compiler but read by F2PY as a
normal Fortran, non-comment line:

  When F2PY finds a line with F2PY directive, the directive is first
  replaced by 5 spaces and then the line is reread.

For fixed format Fortran codes, ``<comment char>`` must be at the
first column of a file, of course. For free format Fortran codes,
F2PY directives can appear anywhere in a file.

C expressions
--------------

C expressions are used in the following parts of signature files:

* ``<init_expr>`` of variable initialization;
* ``<C-booleanexpr>`` of the ``check`` attribute;
* ``<arrayspec> of the ``dimension`` attribute;
* ``callstatement`` statement, here also a C multiline block can be used.

A C expression may contain:

* standard C constructs;
* functions from ``math.h`` and ``Python.h``;
* variables from the argument list, presumably initialized before
  according to given dependence relations;
* the following CPP macros:

  ``rank(<name>)``
    Returns the rank of an array ``<name>``.
  ``shape(<name>,<n>)``
    Returns the ``<n>``-th dimension of an array ``<name>``.
  ``len(<name>)``
    Returns the length of an array ``<name>``.
  ``size(<name>)``
    Returns the size of an array ``<name>``.
  ``slen(<name>)``
    Returns the length of a string ``<name>``.

For initializing an array ``<array name>``, F2PY generates a loop over
all indices and dimensions that executes the following
pseudo-statement::

  <array name>(_i[0],_i[1],...) = <init_expr>;

where ``_i[<i>]`` refers to the ``<i>``-th index value and that runs
from ``0`` to ``shape(<array name>,<i>)-1``.

For example, a function ``myrange(n)`` generated from the following
signature

::

       subroutine myrange(a,n)
         fortranname        ! myrange is a dummy wrapper
         integer intent(in) :: n
         real*8 intent(c,out),dimension(n),depend(n) :: a = _i[0]
       end subroutine myrange

is equivalent to ``numpy.arange(n,dtype=float)``.

.. warning::

  F2PY may lower cases also in C expressions when scanning Fortran codes
  (see ``--[no]-lower`` option).

Multiline blocks
------------------

A multiline block starts with ``'''`` (triple single-quotes) and ends
with ``'''`` in some *strictly* subsequent line.  Multiline blocks can
be used only within .pyf files. The contents of a multiline block can
be arbitrary (except that it cannot contain ``'''``) and no
transformations (e.g. lowering cases) are applied to it.

Currently, multiline blocks can be used in the following constructs:

+ as a C expression of the ``callstatement`` statement;

+ as a C type specification of the ``callprotoargument`` statement;

+ as a C code block of the ``usercode`` statement;

+ as a list of C arrays of the ``pymethoddef`` statement;

+ as documentation string.
