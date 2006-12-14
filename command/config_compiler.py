
import sys
import copy
import distutils.core
from distutils.core import Command
from distutils.errors import DistutilsSetupError
from distutils import log
from numpy.distutils.fcompiler import show_fcompilers, new_fcompiler

#XXX: Implement confic_cc for enhancing C/C++ compiler options.
#XXX: Linker flags

def show_fortran_compilers(_cache=[]):
    # Using cache to prevent infinite recursion
    if _cache:
        return
    _cache.append(1)
    show_fcompilers()

class FCompilerProxy(object):
    """
    A layer of indirection to simplify choosing the correct Fortran compiler.

    If need_f90(), f90(), or fortran(requiref90=True) is called at any time,
    a Fortran 90 compiler is found and used for *all* Fortran sources,
    including Fortran 77 sources.
    """
    #XXX The ability to use a separate F77 compiler is likely not
    # necessary: of all the compilers we support, only the 'gnu'
    # compiler (g77) doesn't support F90, and everything else supports
    # both.

    def __init__(self, compiler_type, distribution):
        self._fcompiler = None
        self._have_f77 = None
        self._have_f90 = None
        self._compiler_type = compiler_type
        self.distribution = distribution

    def _set_fcompiler(self, requiref90=False):
        fc = new_fcompiler(compiler=self._compiler_type,
                           dry_run=self.distribution.dry_run,
                           verbose=self.distribution.verbose,
                           requiref90=requiref90)
        if fc is None:
            raise DistutilsSetupError("could not find a Fortran compiler")
        fc.customize(self.distribution)
        self._fcompiler = fc
        self._have_f77 = fc.compiler_f77 is not None
        if requiref90:
            self._have_f90 = fc.compiler_f90 is not None
        log.info('%s (%s)' % (fc.description, fc.get_version()))

    def need_f77(self):
        if self._fcompiler is None:
            self._set_fcompiler(requiref90=False)
        if not self._have_f77:
            raise DistutilsSetupError("could not find a Fortran 77 compiler")

    def need_f90(self):
        if self._fcompiler is None or self._have_f90 is None:
            self._set_fcompiler(requiref90=True)
        if not self._have_f90:
            raise DistutilsSetupError("could not find a Fortran 90 compiler")

    def f77(self):
        self.need_f77()
        return copy.copy(self._fcompiler)

    def f90(self):
        self.need_f90()
        return copy.copy(self._fcompiler)

    def fortran(self, requiref90=False):
        if requiref90:
            return self.f90()
        else:
            return self.f77()

class config_fc(Command):
    """ Distutils command to hold user specified options
    to Fortran compilers.

    config_fc command is used by the FCompiler.customize() method.
    """

    user_options = [
        ('fcompiler=',None,"specify Fortran compiler type"),
        ('f77exec=', None, "specify F77 compiler command"),
        ('f90exec=', None, "specify F90 compiler command"),
        ('f77flags=',None,"specify F77 compiler flags"),
        ('f90flags=',None,"specify F90 compiler flags"),
        ('ldshared=',None,"shared-library linker command"),
        ('ld=',None,"static library linker command"),
        ('ar=',None,"archiver command (ar)"),
        ('ranlib=',None,"ranlib command"),
        ('opt=',None,"specify optimization flags"),
        ('arch=',None,"specify architecture specific optimization flags"),
        ('debug','g',"compile with debugging information"),
        ('noopt',None,"compile without optimization"),
        ('noarch',None,"compile without arch-dependent optimization"),
        ('fflags=',None,"extra flags for Fortran compiler"),
        ('ldflags=',None,"linker flags"),
        ('arflags=',None,"flags for ar"),
        ]

    help_options = [
        ('help-fcompiler',None, "list available Fortran compilers",
         show_fortran_compilers),
        ]

    boolean_options = ['debug','noopt','noarch']

    def initialize_options(self):
        self.fcompiler = None
        self.f77exec = None
        self.f90exec = None
        self.f77flags = None
        self.f90flags = None
        self.ldshared = None
        self.ld = None
        self.ar = None
        self.ranlib = None
        self.opt = None
        self.arch = None
        self.debug = None
        self.noopt = None
        self.noarch = None
        self.fflags = None
        self.ldflags = None
        self.arflags = None

    def finalize_options(self):
        self.fcompiler = FCompilerProxy(self.fcompiler, self.distribution)

    def run(self):
        pass
