import os
import sys
import pkgutil
import types
import re

from numpy.core.numerictypes import obj2sctype, generic
from numpy.core.multiarray import dtype as _dtype
from numpy.core import product, ndarray

__all__ = ['issubclass_', 'get_numpy_include', 'issubsctype',
           'issubdtype', 'deprecate', 'deprecate_with_doc',
           'get_numarray_include',
           'get_include', 'info', 'source', 'who', 'lookfor',
           'byte_bounds', 'may_share_memory', 'safe_eval']

def issubclass_(arg1, arg2):
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False

def issubsctype(arg1, arg2):
    return issubclass(obj2sctype(arg1), obj2sctype(arg2))

def issubdtype(arg1, arg2):
    """
    Returns True if first argument is a typecode lower/equal in type hierarchy.

    Parameters
    ----------
    arg1 : dtype_like
        dtype or string representing a typecode.
    arg2 : dtype_like
        dtype or string representing a typecode.


    See Also
    --------
    numpy.core.numerictypes : Overview of numpy type hierarchy.

    Examples
    --------
    >>> np.issubdtype('S1', str)
    True
    >>> np.issubdtype(np.float64, np.float32)
    False

    """
    if issubclass_(arg2, generic):
        return issubclass(_dtype(arg1).type, arg2)
    mro = _dtype(arg2).type.mro()
    if len(mro) > 1:
        val = mro[1]
    else:
        val = mro[0]
    return issubclass(_dtype(arg1).type, val)

def get_include():
    """
    Return the directory that contains the numpy \\*.h header files.

    Extension modules that need to compile against numpy should use this
    function to locate the appropriate include directory.

    Notes
    -----
    When using ``distutils``, for example in ``setup.py``.
    ::

        import numpy as np
        ...
        Extension('extension_name', ...
                include_dirs=[np.get_include()])
        ...

    """
    import numpy
    if numpy.show_config is None:
        # running from numpy source directory
        d = os.path.join(os.path.dirname(numpy.__file__), 'core', 'include')
    else:
        # using installed numpy core headers
        import numpy.core as core
        d = os.path.join(os.path.dirname(core.__file__), 'include')
    return d

def get_numarray_include(type=None):
    """Return the directory in the package that contains the numpy/*.h header
    files.

    Extension modules that need to compile against numpy should use this
    function to locate the appropriate include directory. Using distutils:

      import numpy
      Extension('extension_name', ...
                include_dirs=[numpy.get_numarray_include()])
    """
    from numpy.numarray import get_numarray_include_dirs
    include_dirs = get_numarray_include_dirs()
    if type is None:
        return include_dirs[0]
    else:
        return include_dirs + [get_include()]


if sys.version_info < (2, 4):
    # Can't set __name__ in 2.3
    import new
    def _set_function_name(func, name):
        func = new.function(func.func_code, func.func_globals,
                            name, func.func_defaults, func.func_closure)
        return func
else:
    def _set_function_name(func, name):
        func.__name__ = name
        return func

def deprecate(func, oldname=None, newname=None):
    """Deprecate old functions.
    Issues a DeprecationWarning, adds warning to oldname's docstring,
    rebinds oldname.__name__ and returns new function object.

    Example:
    oldfunc = deprecate(newfunc, 'oldfunc', 'newfunc')

    """

    import warnings
    if oldname is None:
        try:
            oldname = func.func_name
        except AttributeError:
            oldname = func.__name__
    if newname is None:
        str1 = "%s is deprecated" % (oldname,)
        depdoc = "%s is DEPRECATED!!" % (oldname,)
    else:
        str1 = "%s is deprecated, use %s" % (oldname, newname),
        depdoc = '%s is DEPRECATED!! -- use %s instead' % (oldname, newname,)

    def newfunc(*args,**kwds):
        """
        Use get_include, get_numpy_include is DEPRECATED.

        """
        warnings.warn(str1, DeprecationWarning)
        return func(*args, **kwds)

    newfunc = _set_function_name(newfunc, oldname)
    doc = func.__doc__
    if doc is None:
        doc = depdoc
    else:
        doc = '\n\n'.join([depdoc, doc])
    newfunc.__doc__ = doc
    try:
        d = func.__dict__
    except AttributeError:
        pass
    else:
        newfunc.__dict__.update(d)
    return newfunc

def deprecate_with_doc(somestr):
    """Decorator to deprecate functions and provide detailed documentation
    with 'somestr' that is added to the functions docstring.

    Example:
    depmsg = 'function scipy.foo has been merged into numpy.foobar'
    @deprecate_with_doc(depmsg)
    def foo():
        pass

    """

    def _decorator(func):
        newfunc = deprecate(func)
        newfunc.__doc__ += "\n" + somestr
        return newfunc
    return _decorator

get_numpy_include = deprecate(get_include, 'get_numpy_include', 'get_include')


#--------------------------------------------
# Determine if two arrays can share memory
#--------------------------------------------

def byte_bounds(a):
    """(low, high) are pointers to the end-points of an array

    low is the first byte
    high is just *past* the last byte

    If the array is not single-segment, then it may not actually
    use every byte between these bounds.

    The array provided must conform to the Python-side of the array interface
    """
    ai = a.__array_interface__
    a_data = ai['data'][0]
    astrides = ai['strides']
    ashape = ai['shape']
    nd_a = len(ashape)
    bytes_a = int(ai['typestr'][2:])

    a_low = a_high = a_data
    if astrides is None: # contiguous case
        a_high += product(ashape, dtype=int)*bytes_a
    else:
        for shape, stride in zip(ashape, astrides):
            if stride < 0:
                a_low += (shape-1)*stride
            else:
                a_high += (shape-1)*stride
        a_high += bytes_a
    return a_low, a_high


def may_share_memory(a, b):
    """Determine if two arrays can share memory

    The memory-bounds of a and b are computed.  If they overlap then
    this function returns True.  Otherwise, it returns False.

    A return of True does not necessarily mean that the two arrays
    share any element.  It just means that they *might*.
    """
    a_low, a_high = byte_bounds(a)
    b_low, b_high = byte_bounds(b)
    if b_low >= a_high or a_low >= b_high:
        return False
    return True

#-----------------------------------------------------------------------------
# Function for output and information on the variables used.
#-----------------------------------------------------------------------------


def who(vardict=None):
    """
    Print the Numpy arrays in the given dictionary.

    If there is no dictionary passed in or `vardict` is None then returns
    Numpy arrays in the globals() dictionary (all Numpy arrays in the
    namespace).

    Parameters
    ----------
    vardict : dict, optional
        A dictionary possibly containing ndarrays.  Default is globals().

    Returns
    -------
    out : None
        Returns 'None'.

    Notes
    -----
    Prints out the name, shape, bytes and type of all of the ndarrays present
    in `vardict`.

    Examples
    --------
    >>> d = {'x': arange(2.0), 'y': arange(3.0), 'txt': 'Some str', 'idx': 5}
    >>> np.whos(d)
    Name            Shape            Bytes            Type
    ===========================================================
    <BLANKLINE>
    y               3                24               float64
    x               2                16               float64
    <BLANKLINE>
    Upper bound on total bytes  =       40

    """
    if vardict is None:
        frame = sys._getframe().f_back
        vardict = frame.f_globals
    sta = []
    cache = {}
    for name in vardict.keys():
        if isinstance(vardict[name],ndarray):
            var = vardict[name]
            idv = id(var)
            if idv in cache.keys():
                namestr = name + " (%s)" % cache[idv]
                original=0
            else:
                cache[idv] = name
                namestr = name
                original=1
            shapestr = " x ".join(map(str, var.shape))
            bytestr = str(var.itemsize*product(var.shape))
            sta.append([namestr, shapestr, bytestr, var.dtype.name,
                        original])

    maxname = 0
    maxshape = 0
    maxbyte = 0
    totalbytes = 0
    for k in range(len(sta)):
        val = sta[k]
        if maxname < len(val[0]):
            maxname = len(val[0])
        if maxshape < len(val[1]):
            maxshape = len(val[1])
        if maxbyte < len(val[2]):
            maxbyte = len(val[2])
        if val[4]:
            totalbytes += int(val[2])

    if len(sta) > 0:
        sp1 = max(10,maxname)
        sp2 = max(10,maxshape)
        sp3 = max(10,maxbyte)
        prval = "Name %s Shape %s Bytes %s Type" % (sp1*' ', sp2*' ', sp3*' ')
        print prval + "\n" + "="*(len(prval)+5) + "\n"

    for k in range(len(sta)):
        val = sta[k]
        print "%s %s %s %s %s %s %s" % (val[0], ' '*(sp1-len(val[0])+4),
                                        val[1], ' '*(sp2-len(val[1])+5),
                                        val[2], ' '*(sp3-len(val[2])+5),
                                        val[3])
    print "\nUpper bound on total bytes  =       %d" % totalbytes
    return

#-----------------------------------------------------------------------------


# NOTE:  pydoc defines a help function which works simliarly to this
#  except it uses a pager to take over the screen.

# combine name and arguments and split to multiple lines of
#  width characters.  End lines on a comma and begin argument list
#  indented with the rest of the arguments.
def _split_line(name, arguments, width):
    firstwidth = len(name)
    k = firstwidth
    newstr = name
    sepstr = ", "
    arglist = arguments.split(sepstr)
    for argument in arglist:
        if k == firstwidth:
            addstr = ""
        else:
            addstr = sepstr
        k = k + len(argument) + len(addstr)
        if k > width:
            k = firstwidth + 1 + len(argument)
            newstr = newstr + ",\n" + " "*(firstwidth+2) + argument
        else:
            newstr = newstr + addstr + argument
    return newstr

_namedict = None
_dictlist = None

# Traverse all module directories underneath globals
# to see if something is defined
def _makenamedict(module='numpy'):
    module = __import__(module, globals(), locals(), [])
    thedict = {module.__name__:module.__dict__}
    dictlist = [module.__name__]
    totraverse = [module.__dict__]
    while 1:
        if len(totraverse) == 0:
            break
        thisdict = totraverse.pop(0)
        for x in thisdict.keys():
            if isinstance(thisdict[x],types.ModuleType):
                modname = thisdict[x].__name__
                if modname not in dictlist:
                    moddict = thisdict[x].__dict__
                    dictlist.append(modname)
                    totraverse.append(moddict)
                    thedict[modname] = moddict
    return thedict, dictlist

def info(object=None,maxwidth=76,output=sys.stdout,toplevel='numpy'):
    """Get help information for a function, class, or module.

       Example:
          >>> np.info(np.polyval) # doctest: +SKIP

          polyval(p, x)

            Evaluate the polymnomial p at x.

            Description:
                If p is of length N, this function returns the value:
                p[0]*(x**N-1) + p[1]*(x**N-2) + ... + p[N-2]*x + p[N-1]
    """
    global _namedict, _dictlist
    # Local import to speed up numpy's import time.
    import pydoc, inspect

    if hasattr(object,'_ppimport_importer') or \
       hasattr(object, '_ppimport_module'):
        object = object._ppimport_module
    elif hasattr(object, '_ppimport_attr'):
        object = object._ppimport_attr

    if object is None:
        info(info)
    elif isinstance(object, ndarray):
        import numpy.numarray as nn
        nn.info(object, output=output, numpy=1)
    elif isinstance(object, str):
        if _namedict is None:
            _namedict, _dictlist = _makenamedict(toplevel)
        numfound = 0
        objlist = []
        for namestr in _dictlist:
            try:
                obj = _namedict[namestr][object]
                if id(obj) in objlist:
                    print >> output, "\n     *** Repeat reference found in %s *** " % namestr
                else:
                    objlist.append(id(obj))
                    print >> output, "     *** Found in %s ***" % namestr
                    info(obj)
                    print >> output, "-"*maxwidth
                numfound += 1
            except KeyError:
                pass
        if numfound == 0:
            print >> output, "Help for %s not found." % object
        else:
            print >> output, "\n     *** Total of %d references found. ***" % numfound

    elif inspect.isfunction(object):
        name = object.func_name
        arguments = inspect.formatargspec(*inspect.getargspec(object))

        if len(name+arguments) > maxwidth:
            argstr = _split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        print >> output, " " + argstr + "\n"
        print >> output, inspect.getdoc(object)

    elif inspect.isclass(object):
        name = object.__name__
        arguments = "()"
        try:
            if hasattr(object, '__init__'):
                arguments = inspect.formatargspec(*inspect.getargspec(object.__init__.im_func))
                arglist = arguments.split(', ')
                if len(arglist) > 1:
                    arglist[1] = "("+arglist[1]
                    arguments = ", ".join(arglist[1:])
        except:
            pass

        if len(name+arguments) > maxwidth:
            argstr = _split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        print >> output, " " + argstr + "\n"
        doc1 = inspect.getdoc(object)
        if doc1 is None:
            if hasattr(object,'__init__'):
                print >> output, inspect.getdoc(object.__init__)
        else:
            print >> output, inspect.getdoc(object)

        methods = pydoc.allmethods(object)
        if methods != []:
            print >> output, "\n\nMethods:\n"
            for meth in methods:
                if meth[0] == '_':
                    continue
                thisobj = getattr(object, meth, None)
                if thisobj is not None:
                    methstr, other = pydoc.splitdoc(inspect.getdoc(thisobj) or "None")
                print >> output, "  %s  --  %s" % (meth, methstr)

    elif type(object) is types.InstanceType: ## check for __call__ method
        print >> output, "Instance of class: ", object.__class__.__name__
        print >> output
        if hasattr(object, '__call__'):
            arguments = inspect.formatargspec(*inspect.getargspec(object.__call__.im_func))
            arglist = arguments.split(', ')
            if len(arglist) > 1:
                arglist[1] = "("+arglist[1]
                arguments = ", ".join(arglist[1:])
            else:
                arguments = "()"

            if hasattr(object,'name'):
                name = "%s" % object.name
            else:
                name = "<name>"
            if len(name+arguments) > maxwidth:
                argstr = _split_line(name, arguments, maxwidth)
            else:
                argstr = name + arguments

            print >> output, " " + argstr + "\n"
            doc = inspect.getdoc(object.__call__)
            if doc is not None:
                print >> output, inspect.getdoc(object.__call__)
            print >> output, inspect.getdoc(object)

        else:
            print >> output, inspect.getdoc(object)

    elif inspect.ismethod(object):
        name = object.__name__
        arguments = inspect.formatargspec(*inspect.getargspec(object.im_func))
        arglist = arguments.split(', ')
        if len(arglist) > 1:
            arglist[1] = "("+arglist[1]
            arguments = ", ".join(arglist[1:])
        else:
            arguments = "()"

        if len(name+arguments) > maxwidth:
            argstr = _split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        print >> output, " " + argstr + "\n"
        print >> output, inspect.getdoc(object)

    elif hasattr(object, '__doc__'):
        print >> output, inspect.getdoc(object)


def source(object, output=sys.stdout):
    """
    Print or write to a file the source code for a Numpy object.

    Parameters
    ----------
    object : numpy object
        Input object.
    output : file object, optional
        If `output` not supplied then source code is printed to screen
        (sys.stdout).  File object must be created with either write 'w' or
        append 'a' modes.

    """
    # Local import to speed up numpy's import time.
    import inspect
    try:
        print >> output,  "In file: %s\n" % inspect.getsourcefile(object)
        print >> output,  inspect.getsource(object)
    except:
        print >> output,  "Not available for this object."


# Cache for lookfor: {id(module): {name: (docstring, kind, index), ...}...}
# where kind: "func", "class", "module", "object"
# and index: index in breadth-first namespace traversal
_lookfor_caches = {}

# regexp whose match indicates that the string may contain a function signature
_function_signature_re = re.compile(r"[a-z_]+\(.*[,=].*\)", re.I)

def lookfor(what, module=None, import_modules=True, regenerate=False):
    """
    Do a keyword search on docstrings.

    A list of of objects that matched the search is displayed,
    sorted by relevance.

    Parameters
    ----------
    what : str
        String containing words to look for.
    module : str, module
        Module whose docstrings to go through.
    import_modules : bool
        Whether to import sub-modules in packages.
        Will import only modules in ``__all__``.
    regenerate : bool
        Whether to re-generate the docstring cache.

    Examples
    --------

    >>> np.lookfor('binary representation')
    Search results for 'binary representation'
    ------------------------------------------
    numpy.binary_repr
        Return the binary representation of the input number as a string.

    """
    import pydoc

    # Cache
    cache = _lookfor_generate_cache(module, import_modules, regenerate)

    # Search
    # XXX: maybe using a real stemming search engine would be better?
    found = []
    whats = str(what).lower().split()
    if not whats: return

    for name, (docstring, kind, index) in cache.iteritems():
        if kind in ('module', 'object'):
            # don't show modules or objects
            continue
        ok = True
        doc = docstring.lower()
        for w in whats:
            if w not in doc:
                ok = False
                break
        if ok:
            found.append(name)

    # Relevance sort
    # XXX: this is full Harrison-Stetson heuristics now,
    # XXX: it probably could be improved

    kind_relevance = {'func': 1000, 'class': 1000,
                      'module': -1000, 'object': -1000}

    def relevance(name, docstr, kind, index):
        r = 0
        # do the keywords occur within the start of the docstring?
        first_doc = "\n".join(docstr.lower().strip().split("\n")[:3])
        r += sum([200 for w in whats if w in first_doc])
        # do the keywords occur in the function name?
        r += sum([30 for w in whats if w in name])
        # is the full name long?
        r += -len(name) * 5
        # is the object of bad type?
        r += kind_relevance.get(kind, -1000)
        # is the object deep in namespace hierarchy?
        r += -name.count('.') * 10
        r += max(-index / 100, -100)
        return r

    def relevance_sort(a, b):
        dr = relevance(b, *cache[b]) - relevance(a, *cache[a])
        if dr != 0: return dr
        else: return cmp(a, b)
    found.sort(relevance_sort)

    # Pretty-print
    s = "Search results for '%s'" % (' '.join(whats))
    help_text = [s, "-"*len(s)]
    for name in found:
        doc, kind, ix = cache[name]

        doclines = [line.strip() for line in doc.strip().split("\n")
                    if line.strip()]

        # find a suitable short description
        try:
            first_doc = doclines[0].strip()
            if _function_signature_re.search(first_doc):
                first_doc = doclines[1].strip()
        except IndexError:
            first_doc = ""
        help_text.append("%s\n    %s" % (name, first_doc))

    # Output
    if len(help_text) > 10:
        pager = pydoc.getpager()
        pager("\n".join(help_text))
    else:
        print "\n".join(help_text)

def _lookfor_generate_cache(module, import_modules, regenerate):
    """
    Generate docstring cache for given module.

    Parameters
    ----------
    module : str, None, module
        Module for which to generate docstring cache
    import_modules : bool
        Whether to import sub-modules in packages.
        Will import only modules in __all__
    regenerate: bool
        Re-generate the docstring cache

    Returns
    -------
    cache : dict {obj_full_name: (docstring, kind, index), ...}
        Docstring cache for the module, either cached one (regenerate=False)
        or newly generated.

    """
    global _lookfor_caches
    # Local import to speed up numpy's import time.
    import inspect

    if module is None:
        module = "numpy"

    if isinstance(module, str):
        module = __import__(module)

    if id(module) in _lookfor_caches and not regenerate:
        return _lookfor_caches[id(module)]

    # walk items and collect docstrings
    cache = {}
    _lookfor_caches[id(module)] = cache
    seen = {}
    index = 0
    stack = [(module.__name__, module)]
    while stack:
        name, item = stack.pop(0)
        if id(item) in seen: continue
        seen[id(item)] = True

        index += 1
        kind = "object"

        if inspect.ismodule(item):
            kind = "module"
            try:
                _all = item.__all__
            except AttributeError:
                _all = None
            # import sub-packages
            if import_modules and hasattr(item, '__path__'):
                for m in pkgutil.iter_modules(item.__path__):
                    if _all is not None and m[1] not in _all:
                        continue
                    try:
                        __import__("%s.%s" % (name, m[1]))
                    except ImportError:
                        continue
            for n, v in inspect.getmembers(item):
                if _all is not None and n not in _all:
                    continue
                stack.append(("%s.%s" % (name, n), v))
        elif inspect.isclass(item):
            kind = "class"
            for n, v in inspect.getmembers(item):
                stack.append(("%s.%s" % (name, n), v))
        elif callable(item):
            kind = "func"

        doc = inspect.getdoc(item)
        if doc is not None:
            cache[name] = (doc, kind, index)

    return cache

#-----------------------------------------------------------------------------

# The following SafeEval class and company are adapted from Michael Spencer's
# ASPN Python Cookbook recipe:
#   http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/364469
# Accordingly it is mostly Copyright 2006 by Michael Spencer.
# The recipe, like most of the other ASPN Python Cookbook recipes was made
# available under the Python license.
#   http://www.python.org/license

# It has been modified to:
#   * handle unary -/+
#   * support True/False/None
#   * raise SyntaxError instead of a custom exception.

class SafeEval(object):

    def visit(self, node, **kw):
        cls = node.__class__
        meth = getattr(self,'visit'+cls.__name__,self.default)
        return meth(node, **kw)

    def default(self, node, **kw):
        raise SyntaxError("Unsupported source construct: %s" % node.__class__)

    def visitExpression(self, node, **kw):
        for child in node.getChildNodes():
            return self.visit(child, **kw)

    def visitConst(self, node, **kw):
        return node.value

    def visitDict(self, node,**kw):
        return dict([(self.visit(k),self.visit(v)) for k,v in node.items])

    def visitTuple(self, node, **kw):
        return tuple([self.visit(i) for i in node.nodes])

    def visitList(self, node, **kw):
        return [self.visit(i) for i in node.nodes]

    def visitUnaryAdd(self, node, **kw):
        return +self.visit(node.getChildNodes()[0])

    def visitUnarySub(self, node, **kw):
        return -self.visit(node.getChildNodes()[0])

    def visitName(self, node, **kw):
        if node.name == 'False':
            return False
        elif node.name == 'True':
            return True
        elif node.name == 'None':
            return None
        else:
            raise SyntaxError("Unknown name: %s" % node.name)

def safe_eval(source):
    """
    Protected string evaluation.

    Evaluate a string containing a Python literal expression without
    allowing the execution of arbitrary non-literal code.

    Parameters
    ----------
    source : str

    Returns
    -------
    obj : object

    Raises
    ------
    SyntaxError
        If the code has invalid Python syntax, or if it contains non-literal
        code.

    Examples
    --------
    >>> from numpy.lib.utils import safe_eval
    >>> safe_eval('1')
    1
    >>> safe_eval('[1, 2, 3]')
    [1, 2, 3]
    >>> safe_eval('{"foo": ("bar", 10.0)}')
    {'foo': ('bar', 10.0)}
    >>> safe_eval('import os')
    Traceback (most recent call last):
      ...
    SyntaxError: invalid syntax
    >>> safe_eval('open("/home/user/.ssh/id_dsa").read()')
    Traceback (most recent call last):
      ...
    SyntaxError: Unsupported source construct: compiler.ast.CallFunc
    >>> safe_eval('dict')
    Traceback (most recent call last):
      ...
    SyntaxError: Unknown name: dict

    """
    # Local import to speed up numpy's import time.
    import compiler
    walker = SafeEval()
    try:
        ast = compiler.parse(source, "eval")
    except SyntaxError, err:
        raise
    try:
        return walker.visit(ast)
    except SyntaxError, err:
        raise

#-----------------------------------------------------------------------------
