
from distutils.command.install import *
from distutils.command.install import install as old_install

class install(old_install):

    def finalize_options (self):
        old_install.finalize_options(self)
        self.install_lib = self.install_libbase
