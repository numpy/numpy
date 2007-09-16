"""Sun studio compiler (both solaris and linux)."""
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.exec_command import find_executable

class SunCCompiler(UnixCCompiler):
    """ A modified Sun compiler compatible with an gcc built Python. """
    compiler_type = 'sun'
    # Use suncc instead of cc, because it makes it more obvious to follow
    # what's going on when several compilers are available.
    cc_exe = 'suncc'

    def __init__ (self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__ (self, verbose,dry_run, force)
        compiler = self.cc_exe
        self.set_executables(compiler=compiler,
                             compiler_so=compiler,
                             compiler_cxx=compiler,
                             linker_exe=compiler,
                             linker_so=compiler + ' -shared')
