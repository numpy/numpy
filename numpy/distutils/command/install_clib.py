import os
from distutils.core import Command
from numpy.distutils.misc_util import get_cmd

class install_clib(Command):
    description = "Command to install installable C libraries"

    user_options = []

    def initialize_options(self):
        self.install_dir = None
        self.outfiles = []

    def finalize_options(self):
        self.set_undefined_options('install', ('install_lib', 'install_dir'))

    def run (self):
        # We need the compiler to get the library name -> filename association
        from distutils.ccompiler import new_compiler
        compiler = new_compiler(compiler=None)
        compiler.customize(self.distribution)

        build_dir = get_cmd("build_clib").build_clib

        for l in self.distribution.installed_libraries:
            target_dir = os.path.join(self.install_dir, l.target_dir)
            name = compiler.library_filename(l.name)
            source = os.path.join(build_dir, name)
            self.mkpath(target_dir)
            self.outfiles.append(self.copy_file(source, target_dir)[0])

    def get_outputs(self):
        return self.outfiles
