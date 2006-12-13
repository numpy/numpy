
import sys
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
    from numpy.distutils.core import get_distribution
    dist = get_distribution()
    print dist.verbose
    show_fcompilers(dist)

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
        fc = new_fcompiler(compiler=self.fcompiler,
                           verbose=self.distribution.verbose)
        fc.customize(self.distribution)
        self.fcompiler = fc
        if self.fcompiler.compiler_f90 is not None:
            self.f90compiler = fc
        else:
            self.f90compiler = None
        log.info('%s (%s)' % (fc.description, fc.get_version()))

    def run(self):
        pass

    def get_f77_compiler(self):
        if self.fcompiler.compiler_f77 is None:
            raise DistutilsSetupError("could not find a Fortran 77 compiler")
        return self.fcompiler

    def get_f90_compiler(self):
        if self.f90compiler is not None:
            return self.f90compiler
        if self.fcompiler.compiler_f90 is None:
            fc = new_fcompiler(compiler=self.fcompiler,
                               verbose=self.distribution.verbose,
                               requiref90=True)
            if fc is None:
                raise DistutilsSetupError("could not find a Fortran 90 compiler")
            self.f90compiler = fc
            return fc
