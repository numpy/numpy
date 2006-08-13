import sys, os
import inspect
import types
from numpy.core.numerictypes import obj2sctype, integer
from numpy.core.multiarray import dtype, _flagdict
from numpy.core import product, ndarray

__all__ = ['issubclass_', 'get_numpy_include', 'issubsctype',
           'issubdtype', 'deprecate', 'get_numarray_include',
	   'get_include', 'ctypes_load_library', 'ndpointer',
           'info', 'source', 'who']

def issubclass_(arg1, arg2):
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False

def issubsctype(arg1, arg2):
    return issubclass(obj2sctype(arg1), obj2sctype(arg2))

def issubdtype(arg1, arg2):
    return issubclass(dtype(arg1).type, dtype(arg2).type)

def get_include():
    """Return the directory in the package that contains the numpy/*.h header
    files.

    Extension modules that need to compile against numpy should use this
    function to locate the appropriate include directory. Using distutils:

      import numpy
      Extension('extension_name', ...
                include_dirs=[numpy.get_include()])
    """
    from numpy.distutils.misc_util import get_numpy_include_dirs
    include_dirs = get_numpy_include_dirs()
    assert len(include_dirs)==1,`include_dirs`
    return include_dirs[0]

def get_numarray_include(type=None):
    """Return the directory in the package that contains the numpy/*.h header
    files.

    Extension modules that need to compile against numpy should use this
    function to locate the appropriate include directory. Using distutils:

      import numpy
      Extension('extension_name', ...
                include_dirs=[numpy.get_include()])
    """
    from numpy.numarray import get_numarray_include_dirs
    from numpy.distutils.misc_util import get_numpy_include_dirs
    include_dirs = get_numarray_include_dirs()
    if type is None:
        return include_dirs[0]
    else:
        return include_dirs + get_numpy_include_dirs()


# Adapted from Albert Strasheim
def ctypes_load_library(libname, loader_path):
    if '.' not in libname:
        if sys.platform == 'win32':
            libname = '%s.dll' % libname
        elif sys.platform == 'darwin':
            libname = '%s.dylib' % libname
        else:
            libname = '%s.so' % libname
    loader_path = os.path.abspath(loader_path)
    if not os.path.isdir(loader_path):
        libdir = os.path.dirname(loader_path)
    else:
        libdir = loader_path
    import ctypes
    libpath = os.path.join(libdir, libname)
    return ctypes.cdll[libpath]

def _num_fromflags(flaglist):
    num = 0
    for val in flaglist:
        num += _flagdict[val]
    return num

def _flags_fromnum(num):
    res = []
    for key, value in _flagdict.items():
        if (num & value):
            res.append(key)
    return res

class _ndptr(object):
    def from_param(cls, obj):
        if not isinstance(obj, ndarray):
            raise TypeError("argument must be an ndarray")
        if obj.dtype != cls._dtype_:
            raise TypeError("array must have data type", cls._dtype_)
        if cls._ndim_ and obj.ndim != cls._ndim_:
            raise TypeError("array must have %d dimension(s)" % cls._ndim_)
        if cls._shape_ and obj.shape != cls._shape_:
            raise TypeError("array must have shape ", cls._shape_)
        if cls._flags_ and ((obj.flags.num & cls._flags_) != cls._flags_):
            raise TypeError("array must have flags ",
                            _flags_fromnum(cls._flags_))
        return obj.ctypes
    from_param = classmethod(from_param)

# Factory for an array-checking object with from_param defined
_pointer_type_cache = {}
def ndpointer(datatype, ndim=None, shape=None, flags=None):
    datatype = dtype(datatype)
    num = None
    if flags is not None:
        if isinstance(flags, str):
            flags = flags.split(',')
        elif isinstance(flags, (int, integer)):
            num = flags
            flags = _flags_fromnum(flags)
        if num is None:
            flags = [x.strip().upper() for x in flags]
            num = _num_fromflags(flags)
    try:
        return _pointer_type_cache[(datatype, ndim, shape, num)]
    except KeyError:
        pass        
    if datatype.names:
        name = str(id(datatype))
    else:
        name = datatype.str
    if ndim is not None:
        name += "_%dd" % ndim
    if shape is not None:
        try:
            strshape = [str(x) for x in shape]
        except TypeError:
            strshape = [str(shape)]
            shape = (shape,)
        shape = tuple(shape)
        name += "_"+"x".join(strshape)
    if flags is not None:
        name += "_"+"_".join(flags)
    else:
        flags = []
    klass = type("ndpointer_%s"%name, (_ndptr,),
                 {"_dtype_": datatype,
                  "_shape_" : shape,
                  "_ndim_" : ndim,
                  "_flags_" : num})
    _pointer_type_cache[datatype] = klass    
    return klass


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

def deprecate(func, oldname, newname):
    import warnings
    def newfunc(*args,**kwds):
        warnings.warn("%s is deprecated, use %s" % (oldname, newname),
                      DeprecationWarning)
        return func(*args, **kwds)
    newfunc = _set_function_name(newfunc, oldname)
    doc = func.__doc__
    depdoc = '%s is DEPRECATED in numpy: use %s instead' % (oldname, newname,)
    if doc is None:
        doc = depdoc
    else:
        doc = '\n'.join([depdoc, doc])
    newfunc.__doc__ = doc
    try:
        d = func.__dict__
    except AttributeError:
        pass
    else:
        newfunc.__dict__.update(d)
    return newfunc

get_numpy_include = deprecate(get_include, 'get_numpy_include', 'get_include')


#-----------------------------------------------------------------------------
# Function for output and information on the variables used.
#-----------------------------------------------------------------------------


def who(vardict=None):
    """Print the scipy arrays in the given dictionary (or globals() if None).
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
          >>> from numpy import *
          >>> info(polyval)
          polyval(p, x)

            Evaluate the polymnomial p at x.

            Description:
                If p is of length N, this function returns the value:
                p[0]*(x**N-1) + p[1]*(x**N-2) + ... + p[N-2]*x + p[N-1]
    """
    global _namedict, _dictlist
    import pydoc

    if hasattr(object,'_ppimport_importer') or \
       hasattr(object, '_ppimport_module'):
        object = object._ppimport_module
    elif hasattr(object, '_ppimport_attr'):
        object = object._ppimport_attr

    if object is None:
        info(info)
    elif isinstance(object, types.StringType):
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
        arguments = apply(inspect.formatargspec, inspect.getargspec(object))

        if len(name+arguments) > maxwidth:
            argstr = _split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        print >> output, " " + argstr + "\n"
        print >> output, inspect.getdoc(object)

    elif inspect.isclass(object):
        name = object.__name__
        if hasattr(object, '__init__'):
            arguments = apply(inspect.formatargspec, inspect.getargspec(object.__init__.im_func))
            arglist = arguments.split(', ')
            if len(arglist) > 1:
                arglist[1] = "("+arglist[1]
                arguments = ", ".join(arglist[1:])
            else:
                arguments = "()"
        else:
            arguments = "()"

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
            arguments = apply(inspect.formatargspec, inspect.getargspec(object.__call__.im_func))
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
        arguments = apply(inspect.formatargspec, inspect.getargspec(object.im_func))
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
    """Write source for this object to output.
    """
    try:
        print >> output,  "In file: %s\n" % inspect.getsourcefile(object)
        print >> output,  inspect.getsource(object)
    except:
        print >> output,  "Not available for this object."

