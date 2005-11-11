# Need to override the build command to include building of fortran libraries
# This class must be used as the entry for the build key in the cmdclass
#    dictionary which is given to the setup command.

__revision__ = "$Id$"

import sys, os
from distutils import util
from distutils.command.build import build as old_build

class build(old_build):

    sub_commands = [('config_fc',     lambda *args: 1), # new feature
                    ('build_src',     old_build.has_ext_modules), # new feature
                    ('build_py',      old_build.has_pure_modules),
                    ('build_clib',    old_build.has_c_libraries),
                    ('build_ext',     old_build.has_ext_modules),
                    ('build_scripts', old_build.has_scripts),
                   ]

    def get_plat_specifier(self):
        """ Return a unique string that identifies this platform.
            The string is used to build path names and contains no
            spaces or control characters. (we hope)
        """        
        plat_specifier = ".%s-%s" % (util.get_platform(), sys.version[0:3])
        
        #--------------------------------------------------------------------
        # get rid of spaces -- added for OS X support.
        # Use '_' like python2.3
        #--------------------------------------------------------------------
        plat_specifier = plat_specifier.replace(' ','_')
        
        #--------------------------------------------------------------------
        # make lower case ?? is this desired'
        #--------------------------------------------------------------------
        #plat_specifier = plat_specifier.lower()
        
        return plat_specifier
        
    def finalize_options (self):

        #--------------------------------------------------------------------
        # This line is re-factored to a function -- everything else in the
        # function is identical to the finalize_options function in the
        # standard distutils build.
        #--------------------------------------------------------------------
        #plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])        
        plat_specifier = self.get_plat_specifier()
        
        # 'build_purelib' and 'build_platlib' just default to 'lib' and
        # 'lib.<plat>' under the base build directory.  We only use one of
        # them for a given distribution, though --
        if self.build_purelib is None:
            self.build_purelib = os.path.join(self.build_base, 'lib')
        if self.build_platlib is None:
            self.build_platlib = os.path.join(self.build_base,
                                              'lib' + plat_specifier)

        # 'build_lib' is the actual directory that we will use for this
        # particular module distribution -- if user didn't supply it, pick
        # one of 'build_purelib' or 'build_platlib'.
        if self.build_lib is None:
            if self.distribution.ext_modules:
                self.build_lib = self.build_platlib
            else:
                self.build_lib = self.build_purelib

        # 'build_temp' -- temporary directory for compiler turds,
        # "build/temp.<plat>"
        if self.build_temp is None:
            self.build_temp = os.path.join(self.build_base,
                                           'temp' + plat_specifier)
        if self.build_scripts is None:
            self.build_scripts = os.path.join(self.build_base,
                                              'scripts-' + sys.version[0:3])

    # finalize_options ()
