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
try:
    from distutils.sysconfig import get_config_vars
    so_ext = get_config_vars('SO')[0] or ''
except ImportError:
    #XXX: implement hooks for .sl, .dll to fully support Python 1.5
    so_ext = '.so'

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
    if not isinstance(module, _ModuleLoader):
        return getattr(module, name)
    return _AttrLoader(module, name)

class _AttrLoader:
    def __init__(self, module, name):
        self._ppimport_attr_module = module
        self._ppimport_attr_name = name
        self._ppimport_attr = None       

    def __getattr__(self, name):
        a = self._ppimport_attr
        if a is None:
            a = getattr(self.__dict__['_ppimport_attr_module'],
                        self.__dict__['_ppimport_attr_name'])
            self._ppimport_attr = a
        return getattr(a, name)

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
    else:
        p_path = p_frame.f_locals['__path__']
        p_dir = p_path[0]
        fullname = p_name + '.' + name

    try:
        return sys.modules[fullname]
    except KeyError:
        pass

    # name is local python or extension module
    location = _is_local_module(p_dir, name,
                                ('.py','.pyc','.pyo',so_ext,'module'+so_ext))
    if location is None:
        # name is local package
        location = _is_local_module(os.path.join(p_dir, name), '__init__',
                                    ('.py','.pyc','.pyo'))

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

        # install loader
        sys.modules[name] = self

    def _ppimport_importer(self):
        name = self.__name__
        module = sys.modules[name]
        if module is self:
            # uninstall loader
            del sys.modules[name]
        #print 'Executing postponed import for %s' %(name)
        module = __import__(name,None,None,['*'])
        self.__dict__ = module.__dict__
        self.__dict__['_ppimport_module'] = module
        return module

    def __repr__(self):
        if self.__dict__.has_key('_ppimport_module'):
            status = 'imported'
        else:
            status = 'import postponed'
        return '<module %s from %s [%s]>' \
               % (`self.__name__`,`self.__file__`, status)

    __str__ = __repr__

    def __setattr__(self, name, value):
        module = self.__dict__.get('_ppimport_module',
                                   self._ppimport_importer())
        return setattr(module, name, value)

    def __getattr__(self, name):
        module = self.__dict__.get('_ppimport_module',
                                   self._ppimport_importer())
        return getattr(module, name)
