
import os
import sys
from glob import glob
#from distutils.command.build_py import *
from distutils.command.build_py import build_py as old_build_py
from fnmatch import fnmatch
from scipy_distutils import log

def is_setup_script(file):
    file = os.path.basename(file)
    return fnmatch(file,"setup.py")
    #return (fnmatch(file,"setup.py") or fnmatch(file,"setup_*.py"))

def in_build_py_ignore(file, _cache={}):
    base,file = os.path.split(file)
    ignore_list = _cache.get(base)
    if ignore_list is None:
        ignore_list = []
        fn = os.path.join(base,'.build_py_ignore')
        if os.path.isfile(fn):
            f = open(fn,'r')
            ignore_list = [x for x in f.read().split('\n') if x]
            f.close()
        _cache[base] = ignore_list
    return file in ignore_list

class build_py(old_build_py):

    def find_package_modules(self, package, package_dir):
        # we filter all files that are setup.py or setup_xxx.py
        # or listed in .build_py_ignore file of files base directory.
        if 'sdist' in sys.argv:
            return old_build_py.find_package_modules(self,package,package_dir)

        self.check_package(package, package_dir)
        module_files = glob(os.path.join(package_dir, "*.py"))
        modules = []
        setup_script = os.path.abspath(self.distribution.script_name)

        for f in module_files:
            abs_f = os.path.abspath(f)
            if not in_build_py_ignore(abs_f) \
               and abs_f != setup_script and not is_setup_script(f):
                module = os.path.splitext(os.path.basename(f))[0]
                modules.append((package, module, f))
            else:
                log.debug("excluding %s", f)
        return modules
