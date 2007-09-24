import os
import os.path
from os.path import join as pjoin, dirname as pdirname

#from distutils.core import build_py as old_build_py
from numpy.distutils.command.build_ext import build_ext as old_build_ext
from numpy.distutils.ccompiler import CCompiler

def dist2sconscc(compiler):
    """This converts the name passed to distutils to scons name convention (C
    compiler).

    Example:
        --compiler=intel -> icc."""
    # Geez, why does distutils has no common way to get the compiler name...
    if compiler.compiler_type == 'msvc':
        #print dir(compiler)
        return 'msvc'
    else:
        #print dir(compiler)
        #print compiler.compiler[0]
        return compiler.compiler[0]

class scons(old_build_ext):
    description = "Scons builder"
    #user_options = []
    user_options = [('fcompiler=', None, "specify the Fortran compiler type"),
                    ('compiler=', None, "specify the C compiler type")]

    def initialize_options(self):
        old_build_ext.initialize_options(self)
        pass

    def finalize_options(self):
        old_build_ext.finalize_options(self)
        if self.distribution.has_scons_scripts():
            print "Got it: scons scripts are %s" % self.distribution.scons_scripts
            self.scons_scripts = self.distribution.scons_scripts
        #        build_py = self.get_finalized_command('build_py')
        print "!!!!!!!!!!!!!!!!!!"
        #from distutils.ccompiler import get_default_compiler
        #import sys
        #import os
        #print get_default_compiler(sys.platform)
        #print get_default_compiler(os.name)
        #from numpy.distutils.ccompiler import compiler_class
        #print compiler_class

        # Try to get the same compiler than the ones used
        compiler_type = self.compiler
        # Initialize C compiler:
        from distutils.ccompiler import new_compiler
        #self.compiler = new_compiler(compiler=compiler_type,
        #                             verbose=self.verbose,
        #                             dry_run=self.dry_run,
        #                             force=self.force)
        self.compiler = new_compiler(compiler=compiler_type)

        print "self.compiler is %s, this gives us %s" % (compiler_type, 
                                                         dist2sconscc(self.compiler))
        #print dir(new_compiler(self.compiler))
        #print new_compiler(self.compiler).compiler
        #print self.build_temp
        #print self.build_lib
        #print self.package

    def run(self):
        # XXX: when a scons script is missing, scons only prints warnings, and
        # does not return a failure (status is 0). We have to detect this from
        # distutils (this cannot work for recursive scons builds...)
        for i in self.scons_scripts:
            cmd = "scons -f " + i + ' -I. '
            cmd += ' src_dir=%s ' % pdirname(i)
            cmd += ' distutils_libdir=%s ' % pjoin(self.build_lib, pdirname(i))
            cmd += ' cc_opt=%s ' % dist2sconscc(self.compiler)
            #print cmd
            st = os.system(cmd)
            if st:
                print "status is %d" % st
