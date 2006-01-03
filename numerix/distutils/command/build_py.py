
from distutils.command.build_py import build_py as old_build_py

class build_py(old_build_py):

    def find_package_modules(self, package, package_dir):
        modules = old_build_py.find_package_modules(self, package, package_dir)

        # Find build_src generated *.py files.
        build_src = self.get_finalized_command('build_src')
        modules += build_src.py_modules.get(package,[])

        return modules
