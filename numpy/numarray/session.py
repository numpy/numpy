""" This module contains a "session saver" which saves the state of a
NumPy session to a file.  At a later time, a different Python
process can be started and the saved session can be restored using
load().

The session saver relies on the Python pickle protocol to save and
restore objects.  Objects which are not themselves picklable (e.g.
modules) can sometimes be saved by "proxy",  particularly when they
are global constants of some kind.  If it's not known that proxying
will work,  a warning is issued at save time.  If a proxy fails to
reload properly (e.g. because it's not a global constant),  a warning
is issued at reload time and that name is bound to a _ProxyFailure
instance which tries to identify what should have been restored.

First, some unfortunate (probably unnecessary) concessions to doctest
to keep the test run free of warnings.

>>> del _PROXY_ALLOWED
>>> del __builtins__

By default, save() stores every variable in the caller's namespace:

>>> import numpy as na
>>> a = na.arange(10)
>>> save()

Alternately,  save() can be passed a comma seperated string of variables:

>>> save("a,na")

Alternately,  save() can be passed a dictionary, typically one you already
have lying around somewhere rather than created inline as shown here:

>>> save(dictionary={"a":a,"na":na})

If both variables and a dictionary are specified, the variables to be
saved are taken from the dictionary.

>>> save(variables="a,na",dictionary={"a":a,"na":na})

Remove names from the session namespace

>>> del a, na

By default, load() restores every variable/object in the session file
to the caller's namespace.

>>> load()

load() can be passed a comma seperated string of variables to be
restored from the session file to the caller's namespace:

>>> load("a,na")

load() can also be passed a dictionary to *restore to*:

>>> d = {}
>>> load(dictionary=d)

load can be passed both a list variables of variables to restore and a
dictionary to restore to:

>>> load(variables="a,na", dictionary=d)

>>> na.all(a == na.arange(10))
1
>>> na.__name__
'numpy'

NOTE:  session saving is faked for modules using module proxy objects.
Saved modules are re-imported at load time but any "state" in the module
which is not restored by a simple import is lost.

"""

__all__ = ['load', 'save']

import sys
import pickle

SAVEFILE="session.dat"
VERBOSE = False           # global import-time  override

def _foo(): pass

_PROXY_ALLOWED = (type(sys),  # module
                  type(_foo), # function
                  type(None)) # None

def _update_proxy_types():
    """Suppress warnings for known un-picklables with working proxies."""
    pass

def _unknown(_type):
    """returns True iff _type isn't known as OK to proxy"""
    return (_type is not None) and (_type not in _PROXY_ALLOWED)

# caller() from the following article with one extra f_back added.
# from http://www.python.org/search/hypermail/python-1994q1/0506.html
# SUBJECT: import ( how to put a symbol into caller's namespace )
# SENDER:  Steven D. Majewski (sdm7g@elvis.med.virginia.edu)
# DATE:  Thu, 24 Mar 1994 15:38:53 -0500

def _caller():
    """caller() returns the frame object of the function's caller."""
    try:
        1 + '' # make an error happen
    except: # and return the caller's caller's frame
        return sys.exc_traceback.tb_frame.f_back.f_back.f_back

def _callers_globals():
    """callers_globals() returns the global dictionary of the caller."""
    frame = _caller()
    return frame.f_globals

def _callers_modules():
    """returns a list containing the names of all the modules in the caller's
    global namespace."""
    g = _callers_globals()
    mods = []
    for k,v in g.items():
        if type(v) == type(sys):
            mods.append(getattr(v,"__name__"))
    return mods

def _errout(*args):
    for a in args:
        print >>sys.stderr, a,
    print >>sys.stderr

def _verbose(*args):
    if VERBOSE:
        _errout(*args)

class _ProxyingFailure:
    """Object which is bound to a variable for a proxy pickle which failed to reload"""
    def __init__(self, module, name, type=None):
        self.module = module
        self.name = name
        self.type = type
    def __repr__(self):
        return "ProxyingFailure('%s','%s','%s')" % (self.module, self.name, self.type)

class _ModuleProxy(object):
    """Proxy object which fakes pickling a module"""
    def __new__(_type, name, save=False):
        if save:
            _verbose("proxying module", name)
            self = object.__new__(_type)
            self.name = name
        else:
            _verbose("loading module proxy", name)
            try:
                self = _loadmodule(name)
            except ImportError:
                _errout("warning: module", name,"import failed.")
        return self

    def __getnewargs__(self):
        return (self.name,)

    def __getstate__(self):
        return False

def _loadmodule(module):
    if module not in sys.modules:
        modules = module.split(".")
        s = ""
        for i in range(len(modules)):
            s = ".".join(modules[:i+1])
            exec "import " + s
    return sys.modules[module]

class _ObjectProxy(object):
    """Proxy object which fakes pickling an arbitrary object.  Only global
    constants can really be proxied."""
    def __new__(_type, module, name, _type2, save=False):
        if save:
            if _unknown(_type2):
                _errout("warning: proxying object", module + "." + name,
                        "of type", _type2, "because it wouldn't pickle...",
                        "it may not reload later.")
            else:
                _verbose("proxying object", module, name)
            self = object.__new__(_type)
            self.module, self.name, self.type = module, name, str(_type2)
        else:
            _verbose("loading object proxy", module, name)
            try:
                m = _loadmodule(module)
            except (ImportError, KeyError):
                _errout("warning: loading object proxy", module + "." + name,
                        "module import failed.")
                return _ProxyingFailure(module,name,_type2)
            try:
                self = getattr(m, name)
            except AttributeError:
                _errout("warning: object proxy", module + "." + name,
                        "wouldn't reload from", m)
                return _ProxyingFailure(module,name,_type2)
        return self

    def __getnewargs__(self):
        return (self.module, self.name, self.type)

    def __getstate__(self):
        return False


class _SaveSession(object):
    """Tag object which marks the end of a save session and holds the
    saved session variable names as a list of strings in the same
    order as the session pickles."""
    def __new__(_type, keys, save=False):
        if save:
            _verbose("saving session", keys)
        else:
            _verbose("loading session", keys)
        self = object.__new__(_type)
        self.keys = keys
        return self

    def __getnewargs__(self):
        return (self.keys,)

    def __getstate__(self):
        return False

class ObjectNotFound(RuntimeError):
    pass

def _locate(modules, object):
    for mname in modules:
        m = sys.modules[mname]
        if m:
            for k,v in m.__dict__.items():
                if v is object:
                    return m.__name__, k
    else:
        raise ObjectNotFound(k)

def save(variables=None, file=SAVEFILE, dictionary=None, verbose=False):

    """saves variables from a numpy session to a file.  Variables
    which won't pickle are "proxied" if possible.

    'variables'       a string of comma seperated variables: e.g. "a,b,c"
                      Defaults to dictionary.keys().

    'file'            a filename or file object for the session file.

    'dictionary'      the dictionary in which to look up the variables.
                      Defaults to the caller's globals()

    'verbose'         print additional debug output when True.
    """

    global VERBOSE
    VERBOSE = verbose

    _update_proxy_types()

    if isinstance(file, str):
        file = open(file, "wb")

    if dictionary is None:
        dictionary = _callers_globals()

    if variables is None:
        keys = dictionary.keys()
    else:
        keys = variables.split(",")

    source_modules = _callers_modules() + sys.modules.keys()

    p = pickle.Pickler(file, protocol=2)

    _verbose("variables:",keys)
    for k in keys:
        v = dictionary[k]
        _verbose("saving", k, type(v))
        try:  # Try to write an ordinary pickle
            p.dump(v)
            _verbose("pickled", k)
        except (pickle.PicklingError, TypeError, SystemError):
            # Use proxies for stuff that won't pickle
            if isinstance(v, type(sys)): # module
                proxy = _ModuleProxy(v.__name__, save=True)
            else:
                try:
                    module, name = _locate(source_modules, v)
                except ObjectNotFound:
                    _errout("warning: couldn't find object",k,
                            "in any module... skipping.")
                    continue
                else:
                    proxy = _ObjectProxy(module, name, type(v), save=True)
            p.dump(proxy)
    o = _SaveSession(keys, save=True)
    p.dump(o)
    file.close()

def load(variables=None, file=SAVEFILE, dictionary=None, verbose=False):

    """load a numpy session from a file and store the specified
    'variables' into 'dictionary'.

    'variables'       a string of comma seperated variables: e.g. "a,b,c"
                      Defaults to dictionary.keys().

    'file'            a filename or file object for the session file.

    'dictionary'      the dictionary in which to look up the variables.
                      Defaults to the caller's globals()

    'verbose'         print additional debug output when True.
    """

    global VERBOSE
    VERBOSE = verbose

    if isinstance(file, str):
        file = open(file, "rb")
    if dictionary is None:
        dictionary = _callers_globals()
    values = []
    p = pickle.Unpickler(file)
    while 1:
        o = p.load()
        if isinstance(o, _SaveSession):
            session = dict(zip(o.keys, values))
            _verbose("updating dictionary with session variables.")
            if variables is None:
                keys = session.keys()
            else:
                keys = variables.split(",")
            for k in keys:
                dictionary[k] = session[k]
            return None
        else:
            _verbose("unpickled object", str(o))
            values.append(o)

def test():
    import doctest, numpy.numarray.session
    return doctest.testmod(numpy.numarray.session)
