# Need to override the build command to include building of fortran libraries
# This class must be used as the entry for the build key in the cmdclass
#    dictionary which is given to the setup command.

from distutils.command.build import *
from distutils.command.build import build as old_build

class build(old_build):
    def has_f_libraries(self):
        return self.distribution.has_f_libraries()

    sub_commands = [('build_py',      old_build.has_pure_modules),
                    ('build_clib',    old_build.has_c_libraries),
                    ('build_flib',    has_f_libraries), # new feature
                    ('build_ext',     old_build.has_ext_modules),
                    ('build_scripts', old_build.has_scripts),
                   ]
