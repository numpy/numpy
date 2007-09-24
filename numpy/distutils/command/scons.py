import os
import os.path
from os.path import join as pjoin, dirname as pdirname

#from distutils.core import build_py as old_build_py
from numpy.distutils.command.build_ext import build_ext as old_build_ext
from numpy.distutils.ccompiler import CCompiler
from numpy.distutils.exec_command import find_executable

def dist2sconscc(compiler):
    """This converts the name passed to distutils to scons name convention (C
    compiler). The argument should be a CCompiler instance.

    Example:
        --compiler=intel -> intelc"""
    if compiler.compiler_type == 'msvc':
        return 'msvc'
    elif compiler.compiler_type == 'intel':
        return 'intelc'
    else:
        return compiler.compiler[0]

def get_compiler_executable(compiler):
    """For any give CCompiler instance, this gives us the name of C compiler
    (the actual executable."""
    # Geez, why does distutils has no common way to get the compiler name...
    if compiler.compiler_type == 'msvc':
        return compiler.cc
    else:
        return compiler.compiler[0]

def get_tool_path(compiler):
    """Given a distutils.ccompiler.CCompiler class, returns the path of the
    toolset related to C compilation."""
    fullpath_exec = find_executable(get_compiler_executable(compiler))
    fullpath = pdirname(fullpath_exec)
    return fullpath

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

        # Try to get the same compiler than the ones used by distutils: this is
        # non trivial because distutils and scons have totally different
        # conventions on this one (distutils uses PATH from user's environment,
        # whereas scons uses standard locations). The way we do it is once we
        # got the c compiler used, we use numpy.distutils function to get the
        # full path, and add the path to the env['PATH'] variable in env
        # instance (this is done in numpy.distutils.scons module).
        compiler_type = self.compiler
        from distutils.ccompiler import new_compiler
        self.compiler = new_compiler(compiler=compiler_type,
                                     verbose=self.verbose,
                                     dry_run=self.dry_run,
                                     force=self.force)
        self.compiler.customize(self.distribution)

        #print "++++++++++++++++++++++++++++++++++++++++"
        #print "self.compiler is %s, this gives us scons tool %s" % (compiler_type, 
        #                                                 dist2sconscc(self.compiler))
        #print get_tool_path(self.compiler)
        #print "++++++++++++++++++++++++++++++++++++++++"

    def run(self):
        # XXX: when a scons script is missing, scons only prints warnings, and
        # does not return a failure (status is 0). We have to detect this from
        # distutils (this cannot work for recursive scons builds...)

        # XXX: passing everything at command line may cause some trouble where
        # there is a size limitation ? What is the standard solution in thise
        # case ?
        for i in self.scons_scripts:
            cmd = "PYTHONPATH=$PYTHONPATH "
            cmd += "scons -f " + i + ' -I. '
            cmd += ' src_dir=%s ' % pdirname(i)
            cmd += ' distutils_libdir=%s ' % pjoin(self.build_lib, pdirname(i))
            cmd += ' cc_opt=%s ' % dist2sconscc(self.compiler)
            cmd += ' cc_opt_path=%s ' % get_tool_path(self.compiler)
            print cmd
            st = os.system(cmd)
            if st:
                print "status is %d" % st
