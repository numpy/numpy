
import sys
from distutils.core import Command

#XXX: Implement confic_cc for enhancing C/C++ compiler options.

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
        ('opt=',None,"specify optimization flags"),
        ('arch=',None,"specify architecture specific optimization flags"),
        ('debug','g',"compile with debugging information"),
        ('noopt',None,"compile without optimization"),
        ('noarch',None,"compile without arch-dependent optimization"),
        ('help-fcompiler',None,"list available Fortran compilers"),
        ]

    boolean_options = ['debug','noopt','noarch','help-fcompiler']

    def initialize_options(self):
        self.fcompiler = None
        self.f77exec = None
        self.f90exec = None
        self.f77flags = None
        self.f90flags = None
        self.opt = None
        self.arch = None
        self.debug = None
        self.noopt = None
        self.noarch = None
        self.help_fcompiler = None
        return

    def finalize_options(self):
        if self.help_fcompiler:
            from scipy_distutils.fcompiler import show_fcompilers
            show_fcompilers(self.distribution)
            sys.exit()
        return

    def run(self):
        # Do nothing.
        return


