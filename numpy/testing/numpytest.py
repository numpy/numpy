import os
import re
import sys
import imp
import glob
import types
import shlex
import unittest
import traceback
import warnings

__all__ = ['set_package_path', 'set_local_path', 'restore_path',
           'IgnoreException', 'importall',]

DEBUG=0
from numpy.testing.utils import jiffies
get_frame = sys._getframe

class IgnoreException(Exception):
    "Ignoring this exception due to disabled feature"


def set_package_path(level=1):
    """ Prepend package directory to sys.path.

    set_package_path should be called from a test_file.py that
    satisfies the following tree structure:

      <somepath>/<somedir>/test_file.py

    Then the first existing path name from the following list

      <somepath>/build/lib.<platform>-<version>
      <somepath>/..

    is prepended to sys.path.
    The caller is responsible for removing this path by using

      restore_path()
    """
    from distutils.util import get_platform
    f = get_frame(level)
    if f.f_locals['__name__']=='__main__':
        testfile = sys.argv[0]
    else:
        testfile = f.f_locals['__file__']
    d = os.path.dirname(os.path.dirname(os.path.abspath(testfile)))
    d1 = os.path.join(d,'build','lib.%s-%s'%(get_platform(),sys.version[:3]))
    if not os.path.isdir(d1):
        d1 = os.path.dirname(d)
    if DEBUG:
        print 'Inserting %r to sys.path for test_file %r' % (d1, testfile)
    sys.path.insert(0,d1)
    return


def set_local_path(reldir='', level=1):
    """ Prepend local directory to sys.path.

    The caller is responsible for removing this path by using

      restore_path()
    """
    f = get_frame(level)
    if f.f_locals['__name__']=='__main__':
        testfile = sys.argv[0]
    else:
        testfile = f.f_locals['__file__']
    local_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(testfile)),reldir))
    if DEBUG:
        print 'Inserting %r to sys.path' % (local_path)
    sys.path.insert(0,local_path)
    return


def restore_path():
    if DEBUG:
        print 'Removing %r from sys.path' % (sys.path[0])
    del sys.path[0]
    return


def output_exception(printstream = sys.stdout):
    try:
        type, value, tb = sys.exc_info()
        info = traceback.extract_tb(tb)
        #this is more verbose
        #traceback.print_exc()
        filename, lineno, function, text = info[-1] # last line only
        print>>printstream, "%s:%d: %s: %s (in %s)" %\
                            (filename, lineno, type.__name__, str(value), function)
    finally:
        type = value = tb = None # clean up
    return


class _dummy_stream:
    def __init__(self,stream):
        self.data = []
        self.stream = stream
    def write(self,message):
        if not self.data and not message.startswith('E'):
            self.stream.write(message)
            self.stream.flush()
            message = ''
        self.data.append(message)
    def writeln(self,message):
        self.write(message+'\n')
    def flush(self):
        self.stream.flush()



def _get_all_method_names(cls):
    names = dir(cls)
    if sys.version[:3]<='2.1':
        for b in cls.__bases__:
            for n in dir(b)+_get_all_method_names(b):
                if n not in names:
                    names.append(n)
    return names


# for debug build--check for memory leaks during the test.

def importall(package):
    """
    Try recursively to import all subpackages under package.
    """
    if isinstance(package,str):
        package = __import__(package)

    package_name = package.__name__
    package_dir = os.path.dirname(package.__file__)
    for subpackage_name in os.listdir(package_dir):
        subdir = os.path.join(package_dir, subpackage_name)
        if not os.path.isdir(subdir):
            continue
        if not os.path.isfile(os.path.join(subdir,'__init__.py')):
            continue
        name = package_name+'.'+subpackage_name
        try:
            exec 'import %s as m' % (name)
        except Exception, msg:
            print 'Failed importing %s: %s' %(name, msg)
            continue
        importall(m)
    return
