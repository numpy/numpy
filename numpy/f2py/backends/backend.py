from __future__ import annotations

from pathlib import Path
from abc import ABC, abstractmethod

import numpy
import numpy.f2py as f2py


class Backend(ABC):
    """
    Superclass for backend compilation plugins to extend.

    """

    def __init__(
        self,
        module_name,
        fortran_compiler: str,
        c_compiler: str,
        f77exec: Path,
        f90exec: Path,
        f77_flags: list[str],
        f90_flags: list[str],
        include_paths: list[Path],
        include_dirs: list[Path],
        external_resources: list[str],
        linker_libpath: list[Path],
        linker_libname: list[str],
        define_macros: list[tuple[str, str]],
        undef_macros: list[str],
        debug: bool,
        opt_flags: list[str],
        arch_flags: list[str],
        no_opt: bool,
        no_arch: bool,
    ) -> None:
        """
        The class is initialized with f2py compile options.
        The parameters are mappings of f2py compilation flags.

        Parameters
        ----------
        module_name : str
                The name of the module to be compiled. (-m)
        fortran_compiler : str
                Name of the Fortran compiler to use. (--fcompiler)
        c_compiler : str
                Name of the C compiler to use. (--ccompiler)
        f77exec : Pathlike
                Path to the fortran compiler for Fortran 77 files (--f77exec)
        f90exec : Pathlike
                Path to the fortran compiler for Fortran 90 and above files (--f90exec)
        f77_flags : list
                List of flags to pass to the fortran compiler for Fortran 77 files (--f77flags)
        f90_flags : list
                List of flags to pass to the fortran compiler for Fortran 90 and above files (--f90flags)
        include_paths : list
                Search include files from given directories (--include-paths)
        include_dirs : list
                Append directory <dir> to the list of directories searched for include files. (-I<dir>)
        external_resources : list
                Link the extension module with <resource> (--link-<resource>)
        linker_libname : list
                Use the library when linking. (-l<libname>)
        define_macros : list
                Define <macro> to <value> if present else define <macro> to true (-D)
        undef_macros : list
                Undefine <macro> (-U)
        linker_libpath : list
                Add directory to the list of directories to be searched for `-l`. (-L)
        opt_flags : list
                Optimization flags to pass to the compiler. (--opt)
        arch_flags : list
                Architectire specific flags to pass to the compiler (--arch)
        no_opt : bool
                Disable optimization. (--no-opt)
        no_arch : bool
                Disable architecture specific optimizations. (--no-arch)
        debug : bool
                Enable debugging. (--debug)

        """
        self.module_name = module_name
        self.fortran_compiler = fortran_compiler
        self.c_compiler = c_compiler
        self.f77exec = f77exec
        self.f90exec = f90exec
        self.f77_flags = f77_flags
        self.f90_flags = f90_flags
        self.include_paths = include_paths
        self.include_dirs = include_dirs
        self.external_resources = external_resources
        self.linker_libpath = linker_libpath
        self.linker_libname = linker_libname
        self.define_macros = define_macros
        self.undef_macros = undef_macros
        self.debug = debug
        self.opt_flags = opt_flags
        self.arch_flags = arch_flags
        self.no_opt = no_opt
        self.no_arch = no_arch

    def numpy_install_path(self) -> Path:
        """
        Returns the install path for numpy.
        """
        return Path(numpy.__file__).parent

    def numpy_get_include(self) -> Path:
        """
        Returns the include paths for numpy.
        """
        return Path(numpy.get_include())

    def f2py_get_include(self) -> Path:
        """
        Returns the include paths for f2py.
        """
        return Path(f2py.get_include())

    @abstractmethod
    def compile(self, fortran_sources: Path, c_wrapper: Path, build_dir: Path) -> None:
        """Compile the wrapper."""
        pass
