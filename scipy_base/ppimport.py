#!/usr/bin/env python
"""
Postpone module import to future.

Python versions: 1.5.2 - 2.3.x
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: March 2003
$Revision$
$Date$
"""
__all__ = ['ppimport','ppimport_attr']

import os
import sys
import string
import types

def _get_so_ext(_cache={}):
    so_ext = _cache.get('so_ext')
    if so_ext is None:
        if sys.platform[:5]=='linux':
            so_ext = '.so'
        else:
            try:
                # if possible, avoid expensive get_config_vars call
                from distutils.sysconfig import get_config_vars
                so_ext = get_config_vars('SO')[0] or ''
            except ImportError:
                #XXX: implement hooks for .sl, .dll to fully support
                #     Python 1.5.x   
                so_ext = '.so'
        _cache['so_ext'] = so_ext
    return so_ext

def _get_frame(level=0):
    try:
        return sys._getframe(level+1)
    except AttributeError:
        # Python<=2.0 support
        frame = sys.exc_info()[2].tb_frame
        for i in range(level+1):
            frame = frame.f_back
        return frame

def ppimport_attr(module, name):
    """ ppimport(module, name) is 'postponed' getattr(module, name)
    """
    if isinstance(module, _ModuleLoader):
        return _AttrLoader(module, name)
    return getattr(module, name)

class _AttrLoader:
    def __init__(self, module, name):
        self.__dict__['_ppimport_attr_module'] = module
        self.__dict__['_ppimport_attr_name'] = name

    def _ppimport_attr_getter(self):
        attr = getattr(self.__dict__['_ppimport_attr_module'],
                       self.__dict__['_ppimport_attr_name'])
        try:
            d = attr.__dict__
            if d is not None:
                self.__dict__ = d
        except AttributeError:
            pass
        self.__dict__['_ppimport_attr'] = attr
        return attr

    def __getattr__(self, name):
        try:
            attr = self.__dict__['_ppimport_attr']
        except KeyError:
            attr = self._ppimport_attr_getter()
        if name=='_ppimport_attr':
            return attr
        return getattr(attr, name)

    def __repr__(self):
        if self.__dict__.has_key('_ppimport_attr'):
            return repr(self._ppimport_attr)
        module = self.__dict__['_ppimport_attr_module']
        name = self.__dict__['_ppimport_attr_name']
        return "<attribute %s of %s>" % (`name`,`module`)

    __str__ = __repr__

    # For function and class attributes.
    def __call__(self, *args, **kwds):
        return self._ppimport_attr(*args,**kwds)



def _is_local_module(p_dir,name,suffices):
    base = os.path.join(p_dir,name)
    for suffix in suffices:
        if os.path.isfile(base+suffix):
            if p_dir:
                return base+suffix
            return name+suffix

def ppimport(name):
    """ ppimport(name) -> module or module wrapper

    If name has been imported before, return module. Otherwise
    return ModuleLoader instance that transparently postpones
    module import until the first attempt to access module name
    attributes.
    """
    p_frame = _get_frame(1)
    p_name = p_frame.f_locals['__name__']
    if p_name=='__main__':
        p_dir = ''
        fullname = name
    elif p_frame.f_locals.has_key('__path__'):
        # python package
        p_path = p_frame.f_locals['__path__']
        p_dir = p_path[0]
        fullname = p_name + '.' + name
    else:
        # python module, not tested
        p_file = p_frame.f_locals['__file__']
        p_dir = os.path.dirname(p_file)
        fullname = p_name + '.' + name

    module = sys.modules.get(fullname)
    if module is not None:
        return module

    so_ext = _get_so_ext()
    py_exts = ('.py','.pyc','.pyo')
    so_exts = (so_ext,'module'+so_ext)
    
    for d,n,fn,e in [\
        # name is local python module or local extension module
        (p_dir, name, fullname, py_exts+so_exts),
        # name is local package
        (os.path.join(p_dir, name), '__init__', fullname, py_exts),
        # name is package in parent directory (scipy specific)
        (os.path.join(os.path.dirname(p_dir), name), '__init__', name, py_exts),
        ]:
        location = _is_local_module(d, n, e)
        if location is not None:
            fullname = fn
            break

    if location is None:
        # name is to be looked in python sys.path.
        # It is OK if name does not exists. The ImportError is
        # postponed until trying to use the module.
        fullname = name
        location = 'sys.path'

    return _ModuleLoader(fullname,location)

class _ModuleLoader:
    # Don't use it directly. Use ppimport instead.

    def __init__(self,name,location):

        # set attributes, avoid calling __setattr__
        self.__dict__['__name__'] = name
        self.__dict__['__file__'] = location

        if location != 'sys.path':
            # get additional attributes (doc strings, etc)
            # from pre_<name>.py file.
            #filename = os.path.splitext(location)[0] + '.py'
            filename = location
            dirname,basename = os.path.split(filename)
            preinit = os.path.join(dirname,'pre_'+basename)
            if os.path.isfile(preinit):
                execfile(preinit, self.__dict__)

        # install loader
        sys.modules[name] = self

    def _ppimport_importer(self):
        name = self.__name__
	try:
	    module = sys.modules[name]
	except KeyError:
	    raise ImportError,self.__dict__.get('_ppimport_exc_value')
        assert module is self,`module`

        # uninstall loader
        del sys.modules[name]

        #print 'Executing postponed import for %s' %(name)
	try:
	    module = __import__(name,None,None,['*'])
	except ImportError:
	    self.__dict__['_ppimport_exc_value'] = str(sys.exc_value)
	    raise
        assert isinstance(module,types.ModuleType),`module`

        self.__dict__ = module.__dict__
        self.__dict__['_ppimport_module'] = module
        return module

    def __setattr__(self, name, value):
        try:
            module = self.__dict__['_ppimport_module']
        except KeyError:
            module = self._ppimport_importer()
        return setattr(module, name, value)

    def __getattr__(self, name):
        try:
            module = self.__dict__['_ppimport_module']
        except KeyError:
            module = self._ppimport_importer()
        return getattr(module, name)

    def __repr__(self):
        if self.__dict__.has_key('_ppimport_module'):
            status = 'imported'
        elif self.__dict__.has_key('_ppimport_exc_value'):
            status = 'import error'
        else:
            status = 'import postponed'
        return '<module %s from %s [%s]>' \
               % (`self.__name__`,`self.__file__`, status)

    __str__ = __repr__

try:
    import pydoc as _pydoc
except ImportError:
    _pydoc = None
if _pydoc is not None:
    # Define new built-in 'help'.
    # This is a wrapper around pydoc.help (with a twist
    # (as in debian site.py) and ppimport support).
    class _Helper:
        def __repr__ (self):
            return "Type help () for interactive help, " \
                   "or help (object) for help about object."
        def __call__ (self, *args, **kwds):
            new_args = []
            for a in args:
                if hasattr(a,'_ppimport_importer') or \
		   hasattr(a,'_ppimport_module'):
                    a = a._ppimport_module
                if hasattr(a,'_ppimport_attr'):
		    a = a._ppimport_attr
                new_args.append(a)
            return _pydoc.help(*new_args, **kwds)
    import __builtin__
    __builtin__.help = _Helper()

    import inspect as _inspect
    _old_inspect_getfile = _inspect.getfile
    def _inspect_getfile(object):
	try:
	    if hasattr(object,'_ppimport_importer') or \
	       hasattr(object,'_ppimport_module'):
                object = object._ppimport_module
            if hasattr(object,'_ppimport_attr'):
		object = object._ppimport_attr
	except ImportError:
	    object = object.__class__
	return _old_inspect_getfile(object)
    _inspect.getfile = _inspect_getfile
