from distutils.command.build_py import *
from distutils.command.build_py import build_py as old_build_py
from fnmatch import fnmatch

def is_setup_script(file):
    file = os.path.basename(file)
    return (fnmatch(file,"setup.py") or fnmatch(file,"setup_*.py"))
    
class build_py(old_build_py):
    def find_package_modules (self, package, package_dir):
        # we filter all files that are setup.py or setup_xxx.py        
        self.check_package(package, package_dir)
        module_files = glob(os.path.join(package_dir, "*.py"))
        modules = []
        setup_script = os.path.abspath(self.distribution.script_name)

        for f in module_files:
            abs_f = os.path.abspath(f)
            if abs_f != setup_script and not is_setup_script(f):
                module = os.path.splitext(os.path.basename(f))[0]
                modules.append((package, module, f))
            else:
                self.debug_print("excluding %s" % setup_script)
        return modules

