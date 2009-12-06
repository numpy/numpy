import os
import sys
import traceback

__all__ = ['IgnoreException', 'importall',]

DEBUG=0
get_frame = sys._getframe

class IgnoreException(Exception):
    "Ignoring this exception due to disabled feature"


def output_exception(printstream = sys.stdout):
    try:
        type, value, tb = sys.exc_info()
        info = traceback.extract_tb(tb)
        #this is more verbose
        #traceback.print_exc()
        filename, lineno, function, text = info[-1] # last line only
        msg = "%s:%d: %s: %s (in %s)\n" % (
            filename, lineno, type.__name__, str(value), function)
        printstream.write(msg)
    finally:
        type = value = tb = None # clean up
    return

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
