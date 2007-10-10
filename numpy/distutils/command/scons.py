import os
import os.path
from os.path import join as pjoin, dirname as pdirname

#from distutils.core import build_py as old_build_py
from distutils.errors import DistutilsExecError, DistutilsSetupError
from numpy.distutils.command.build_ext import build_ext as old_build_ext
from numpy.distutils.ccompiler import CCompiler
from numpy.distutils.exec_command import find_executable

def get_scons_local_path():
    """This returns the full path where scons.py for scons-local is located."""
    import numpy.distutils
    return pjoin(pdirname(numpy.distutils.__file__), 'scons-local')

def get_python_exec_invoc():
    """This returns the python executable from which this file is invocated."""
    # Do we  need to take into account the PYTHONPATH, in a cross platform way,
    # that is the string returned can be executed directly on supported
    # platforms, and the sys.path of the executed python should be the same
    # than the caller ? This may not be necessary, since os.system is said to
    # take into accound os.environ. This actually also works for my way of
    # using "local python", using the alias facility of bash.
    import sys
    #print dir(sys)
    return sys.executable
    #try:
    #    pypath = os.environ['PYTHONPATH']
    #except KeyError:
    #    pypath = None

def dist2sconscc(compiler):
    """This converts the name passed to distutils to scons name convention (C
    compiler). The argument should be a CCompiler instance.

    Example:
        --compiler=intel -> intelc"""
    if compiler.compiler_type == 'msvc':
        return 'msvc'
    elif compiler.compiler_type == 'intel':
        return 'intelc'
    elif compiler.compiler_type == 'mingw32':
        return 'mingw'
    else:
        return compiler.compiler[0]

def get_compiler_executable(compiler):
    """For any give CCompiler instance, this gives us the name of C compiler
    (the actual executable)."""
    # Geez, why does distutils has no common way to get the compiler name...
    if compiler.compiler_type == 'msvc':
        # this is harcoded in distutils... A bit cleaner way would be to
        # initialize the compiler instance and then get compiler.cc, but this
        # may be costly: we really just want a string.
        # XXX: we need to initialize the compiler anyway, so do not use
        # hardcoded string
        #compiler.initialize()
        #print compiler.cc
        return 'cl.exe' 
    else:
        return compiler.compiler[0]

def get_tool_path(compiler):
    """Given a distutils.ccompiler.CCompiler class, returns the path of the
    toolset related to C compilation."""
    fullpath_exec = find_executable(get_compiler_executable(compiler))
    if fullpath_exec:
        fullpath = pdirname(fullpath_exec)
    else:
        raise DistutilsSetupError("Could not find compiler executable info for scons")
    return fullpath

def protect_path(path):
    """Convert path (given as a string) to something the shell will have no
    problem to understand (space, etc... problems)."""
    # XXX: to this correctly, this is totally bogus for now (does not check for
    # already quoted path, for example.
    return '"' + path + '"'

class scons(old_build_ext):
    # XXX: I really do not like the way distutils add attributes "on the fly".
    # We should eally avoid that and remove all the code which does it before
    # release.
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
        else:
            self.scons_scripts = []

        # Try to get the same compiler than the ones used by distutils: this is
        # non trivial because distutils and scons have totally different
        # conventions on this one (distutils uses PATH from user's environment,
        # whereas scons uses standard locations). The way we do it is once we
        # got the c compiler used, we use numpy.distutils function to get the
        # full path, and add the path to the env['PATH'] variable in env
        # instance (this is done in numpy.distutils.scons module).
        compiler_type = self.compiler
        print compiler_type
        from distutils.ccompiler import new_compiler
        self.compiler = new_compiler(compiler=compiler_type,
                                     verbose=self.verbose,
                                     dry_run=self.dry_run,
                                     force=self.force)
        self.compiler.customize(self.distribution)
		
	# This initialization seems necessary, sometimes, for find_executable to work...
        if hasattr(self.compiler, 'initialize'):
            self.compiler.initialize()
		
        # XXX: debug, remove
        if self.scons_scripts:
            #print "++++++++++++++++++++++++++++++++++++++++"
            #print "self.compiler is %s, this gives us scons tool %s" % (compiler_type, 
            #                                                 dist2sconscc(self.compiler))
            #print get_tool_path(self.compiler)
            #print "++++++++++++++++++++++++++++++++++++++++"

            print "++++++++++++++++++++++++++++++++++++++++"
            print get_scons_local_path()
            print get_python_exec_invoc()
            print "++++++++++++++++++++++++++++++++++++++++"

    def run(self):
        # XXX: when a scons script is missing, scons only prints warnings, and
        # does not return a failure (status is 0). We have to detect this from
        # distutils (this cannot work for recursive scons builds...)

        # XXX: passing everything at command line may cause some trouble where
        # there is a size limitation ? What is the standard solution in thise
        # case ?

        scons_exec = get_python_exec_invoc()
        scons_exec += ' ' + protect_path(pjoin(get_scons_local_path(), 'scons.py'))
        for i in self.scons_scripts:
            cmd = scons_exec + " -f " + i + ' -I. '
            cmd += ' src_dir="%s" ' % pdirname(i)
            cmd += ' distutils_libdir=%s ' % protect_path(pjoin(self.build_lib, pdirname(i)))
            cmd += ' cc_opt=%s ' % dist2sconscc(self.compiler)
            cmd += ' cc_opt_path=%s ' % protect_path(get_tool_path(self.compiler))
            st = os.system(cmd)
            if st:
                print "status is %d" % st
                raise DistutilsExecError("Error while executing scons command %s (see above)" % cmd)
