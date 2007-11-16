import os
import os.path
from os.path import join as pjoin, dirname as pdirname

from distutils.errors import DistutilsPlatformError
from distutils.errors import DistutilsExecError, DistutilsSetupError

from numpy.distutils.command.build_ext import build_ext as old_build_ext
from numpy.distutils.ccompiler import CCompiler
from numpy.distutils.fcompiler import FCompiler
from numpy.distutils.exec_command import find_executable
from numpy.distutils.misc_util import get_scons_build_dir
from numpy.distutils import log
from numpy.distutils.misc_util import get_numpy_include_dirs

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
    return sys.executable

def dirl_to_str(dirlist):
    """Given a list of directories, returns a string where the paths are
    concatenated by the path separator.

    example: ['foo/bar', 'bar/foo'] will return 'foo/bar:bar/foo'."""
    return os.pathsep.join(dirlist)

def dist2sconscc(compiler):
    """This converts the name passed to distutils to scons name convention (C
    compiler). compiler should be a CCompiler instance.

    Example:
        --compiler=intel -> intelc"""
    compiler_type = compiler.compiler_type
    if compiler_type == 'msvc':
        return 'msvc'
    elif compiler_type == 'intel':
        return 'intelc'
    elif compiler_type == 'mingw32':
        return 'mingw'
    else:
        return compiler.compiler[0]

def dist2sconsfc(compiler):
    """This converts the name passed to distutils to scons name convention
    (Fortran compiler). The argument should be a FCompiler instance.

    Example:
        --fcompiler=intel -> ifort on linux, ifl on windows"""
    if compiler.compiler_type == 'intel':
        raise NotImplementedError('FIXME: intel fortran compiler name ?')
        #return 'intelc'
    elif compiler.compiler_type == 'gnu':
        return 'g77'
    elif compiler.compiler_type == 'gnu95':
        return 'gfortran'
    else:
        # XXX: Just give up for now, and use generic fortran compiler
        return 'fortran'

def get_compiler_executable(compiler):
    """For any give CCompiler instance, this gives us the name of C compiler
    (the actual executable).
    
    NOTE: does NOT work with FCompiler instances."""
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

def get_f77_compiler_executable(compiler):
    """For any give FCompiler instance, this gives us the name of F77 compiler
    (the actual executable)."""
    return compiler.compiler_f77[0]

def get_tool_path(compiler):
    """Given a distutils.ccompiler.CCompiler class, returns the path of the
    toolset related to C compilation."""
    fullpath_exec = find_executable(get_compiler_executable(compiler))
    if fullpath_exec:
        fullpath = pdirname(fullpath_exec)
    else:
        raise DistutilsSetupError("Could not find compiler executable info for scons")
    return fullpath

def get_f77_tool_path(compiler):
    """Given a distutils.ccompiler.FCompiler class, returns the path of the
    toolset related to F77 compilation."""
    fullpath_exec = find_executable(get_f77_compiler_executable(compiler))
    if fullpath_exec:
        fullpath = pdirname(fullpath_exec)
    else:
        raise DistutilsSetupError("Could not find F77 compiler executable "\
                "info for scons")
    return fullpath

def protect_path(path):
    """Convert path (given as a string) to something the shell will have no
    problem to understand (space, etc... problems)."""
    # XXX: to this correctly, this is totally bogus for now (does not check for
    # already quoted path, for example).
    return '"' + path + '"'

class scons(old_build_ext):
    # XXX: I really do not like the way distutils overuses monkey patch. We
    # should eally avoid that and remove all the code which does it before
    # release.
    # XXX: add an option to the scons command for configuration (auto/force/cache).
    description = "Scons builder"
    user_options = old_build_ext.user_options + \
            [('jobs=', None, 
              "specify number of worker threads when executing scons"),
             ('silent=', None, 'specify whether scons output should be silent '\
                               '(1), super silent (2) or not (0, default)')]

    def initialize_options(self):
        old_build_ext.initialize_options(self)
        self.jobs = None
        self.silent = 0
        # If true, we bypass distutils to find the c compiler altogether. This
        # is to be used in desperate cases (like incompatible visual studio
        # version).
        self._bypass_distutils_cc = False
        self.scons_compiler = None
        self.scons_compiler_path = None
        self.scons_fcompiler = None

    def finalize_options(self):
        old_build_ext.finalize_options(self)
        if self.distribution.has_scons_scripts():
            self.sconscripts = self.distribution.get_scons_scripts()
            self.pre_hooks = self.distribution.get_scons_pre_hooks()
            self.post_hooks = self.distribution.get_scons_post_hooks()
            self.pkg_names = self.distribution.get_scons_parent_names()
        else:
            self.sconscripts = []
            self.pre_hooks = []
            self.post_hooks = []
            self.pkg_names = []

        # Try to get the same compiler than the ones used by distutils: this is
        # non trivial because distutils and scons have totally different
        # conventions on this one (distutils uses PATH from user's environment,
        # whereas scons uses standard locations). The way we do it is once we
        # got the c compiler used, we use numpy.distutils function to get the
        # full path, and add the path to the env['PATH'] variable in env
        # instance (this is done in numpy.distutils.scons module).

        # XXX: The logic to bypass distutils is ... not so logic.
        compiler_type = self.compiler
        if compiler_type == 'msvc':
            self._bypass_distutils_cc = True
        from numpy.distutils.ccompiler import new_compiler
        try:
            distutils_compiler = new_compiler(compiler=compiler_type,
                                      verbose=self.verbose,
                                      dry_run=self.dry_run,
                                      force=self.force)
            distutils_compiler.customize(self.distribution)
            # This initialization seems necessary, sometimes, for find_executable to work...
            if hasattr(distutils_compiler, 'initialize'):
                distutils_compiler.initialize()
            self.scons_compiler = dist2sconscc(distutils_compiler)
            self.scons_compiler_path = protect_path(get_tool_path(distutils_compiler))
        except DistutilsPlatformError, e:
            if not self._bypass_distutils_cc: 
                raise e
            else:
                self.scons_compiler = compiler_type
 		
        # We do the same for the fortran compiler
        fcompiler_type = self.fcompiler
        from numpy.distutils.fcompiler import new_fcompiler
        self.fcompiler = new_fcompiler(compiler = fcompiler_type,
                                       verbose = self.verbose,
                                       dry_run = self.dry_run,
                                       force = self.force)
        if self.fcompiler is not None:
            self.fcompiler.customize(self.distribution)

    def run(self):
        # XXX: when a scons script is missing, scons only prints warnings, and
        # does not return a failure (status is 0). We have to detect this from
        # distutils (this cannot work for recursive scons builds...)

        # XXX: passing everything at command line may cause some trouble where
        # there is a size limitation ? What is the standard solution in thise
        # case ?

        scons_exec = get_python_exec_invoc()
        scons_exec += ' ' + protect_path(pjoin(get_scons_local_path(), 'scons.py'))

        for sconscript, pre_hook, post_hook, pkg_name in zip(self.sconscripts,
                                                   self.pre_hooks, self.post_hooks,
                                                   self.pkg_names):
            if pre_hook:
                pre_hook()

            cmd = [scons_exec, "-f", sconscript, '-I.']
            if self.jobs:
                cmd.append(" --jobs=%d" % int(self.jobs))
            cmd.append('src_dir="%s"' % pdirname(sconscript))
            cmd.append('pkg_name="%s"' % pkg_name)
            #cmd.append('distutils_libdir=%s' % protect_path(pjoin(self.build_lib,
            #                                                    pdirname(sconscript))))
            cmd.append('distutils_libdir=%s' % protect_path(pjoin(self.build_lib)))

            if not self._bypass_distutils_cc:
                cmd.append('cc_opt=%s' % self.scons_compiler)
                cmd.append('cc_opt_path=%s' % self.scons_compiler_path)
            else:
                cmd.append('cc_opt=%s' % self.scons_compiler)


            if self.fcompiler:
                cmd.append('f77_opt=%s' % dist2sconsfc(self.fcompiler))
                cmd.append('f77_opt_path=%s' % protect_path(get_f77_tool_path(self.fcompiler)))

            cmd.append('include_bootstrap=%s' % dirl_to_str(get_numpy_include_dirs()))
            if self.silent:
                if int(self.silent) == 1:
                    cmd.append('-Q')
                elif int(self.silent) == 2:
                    cmd.append('-s')
            cmdstr = ' '.join(cmd)
            log.info("Executing scons command: %s ", cmdstr)
            st = os.system(cmdstr)
            if st:
                print "status is %d" % st
                raise DistutilsExecError("Error while executing scons command "\
                                         "%s (see above)" % cmdstr)
            if post_hook:
                post_hook()
