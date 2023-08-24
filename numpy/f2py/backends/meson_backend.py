from __future__ import annotations

import os
import errno
import shutil
import subprocess
from pathlib import Path

from .backend import Backend
from string import Template


class MesonTemplate:
    """Template meson build file generation class."""

    def __init__(
        self,
        module_name: str,
        numpy_install_path,
        numpy_get_include: Path,
        f2py_get_include: Path,
        wrappers: list[Path],
        fortran_sources: list[Path],
        dependencies: list[str],
        include_path: list[Path],
        optimization_flags: list[str],
        architecture_flags: list[str],
        f77_flags: list[str],
        f90_flags: list[str],
        linker_libpath: list[Path],
        linker_libname: list[str],
        define_macros: list[tuple[str, str]],
        undef_macros: list[str],
    ):
        self.module_name = module_name
        self.numpy_install_path = numpy_install_path
        self.build_template_path = (
            numpy_install_path / "f2py" / "backends" / "src" / "meson.build.src"
        )
        self.sources = fortran_sources
        self.numpy_get_include = numpy_get_include
        self.f2py_get_include = f2py_get_include
        self.wrappers = wrappers
        self.dependencies = dependencies
        self.include_directories = include_path
        self.substitutions = {}
        self.optimization_flags = optimization_flags
        self.architecture_flags = architecture_flags
        self.fortran_flags = f77_flags + f90_flags
        self.linker_libpath = linker_libpath
        self.linker_libname = linker_libname
        self.define_macros = define_macros
        self.undef_macros = undef_macros
        self.pipeline = [
            self.initialize_template,
            self.global_flags_substitution,
            self.sources_substitution,
            self.dependencies_substitution,
            self.include_directories_subtitution,
            self.linker_substitution,
            self.macros_substitution,
        ]

    @property
    def meson_build_template(self) -> str:
        if not self.build_template_path.is_file():
            raise FileNotFoundError(
                errno.ENOENT,
                f"Meson build template {self.build_template_path.absolute()} does not exist.",
            )
        return self.build_template_path.read_text()

    def initialize_template(self) -> None:
        """Initialize with module name and external NumPy and F2PY C libraries."""
        self.substitutions["modulename"] = self.module_name
        self.substitutions["numpy_get_include"] = self.numpy_get_include.absolute()
        self.substitutions["f2py_get_include"] = self.f2py_get_include.absolute()

    def sources_substitution(self) -> None:
        self.substitutions["source_list"] = ",".join(
            ["'" + str(source.absolute()) + "'" for source in self.sources]
        )
        self.substitutions["wrappers"] = ",".join(
            ["'" + str(wrapper.absolute()) + "'" for wrapper in self.wrappers]
        )

    def dependencies_substitution(self) -> None:
        self.substitutions["dependencies_list"] = ", ".join(
            [f"dependency('{dependecy}')" for dependecy in self.dependencies]
        )

    def include_directories_subtitution(self) -> None:
        self.substitutions["include_directories_list"] = ", ".join(
            [
                f"include_directories('{include_directory}')"
                for include_directory in self.include_directories
            ]
        )

    def global_flags_substitution(self) -> None:
        fortran_compiler_flags = (
            self.fortran_flags + self.optimization_flags + self.architecture_flags
        )
        c_compiler_flags = self.optimization_flags + self.architecture_flags
        self.substitutions["fortran_global_args"] = fortran_compiler_flags
        self.substitutions["c_global_args"] = c_compiler_flags

    def macros_substitution(self) -> None:
        self.substitutions["macros"] = ""
        if self.define_macros:
            self.substitutions["macros"] = ",".join(
                f"'-D{macro[0]}={macro[1]}'" if macro[1] else f"-D{macro[0]}"
                for macro in self.define_macros
            )
        if self.undef_macros:
            self.substitutions["macros"] += "," + ",".join(
                f"'-U{macro}'" for macro in self.undef_macros
            )

    def linker_substitution(self) -> None:
        self.substitutions["linker_args"] = ""
        if self.linker_libpath:
            linker_libpath_subs = ",".join(
                f"-L{libpath}" for libpath in self.linker_libpath
            )
            self.substitutions["linker_args"] += linker_libpath_subs
        if self.linker_libname:
            linker_libname_subs = ",".join(
                f"-l{libname}" for libname in self.linker_libname
            )
            self.substitutions["linker_args"] += f",{linker_libname_subs}"

    def generate_meson_build(self) -> str:
        for node in self.pipeline:
            node()
        template = Template(self.meson_build_template)
        return template.substitute(self.substitutions)


class MesonBackend(Backend):
    def __init__(
        self,
        module_name: str = "untitled",
        fortran_compiler: str = None,
        c_compiler: str = None,
        f77exec: Path = None,
        f90exec: Path = None,
        f77_flags: list[str] = None,
        f90_flags: list[str] = None,
        include_paths: list[Path] = None,
        include_dirs: list[Path] = None,
        external_resources: list[str] = None,
        linker_libpath: list[Path] = None,
        linker_libname: list[str] = None,
        define_macros: list[tuple[str, str]] = None,
        undef_macros: list[str] = None,
        debug: bool = False,
        opt_flags: list[str] = None,
        arch_flags: list[str] = None,
        no_opt: bool = False,
        no_arch: bool = False,
    ) -> None:
        self.meson_build_dir = "builddir"
        if f77_flags is None:
            f77_flags = []
        if include_paths is None:
            include_paths = []
        if include_dirs is None:
            include_dirs = []
        if external_resources is None:
            external_resources = []
        if linker_libpath is None:
            linker_libpath = []
        if linker_libname is None:
            linker_libname = []
        if define_macros is None:
            define_macros = []
        if undef_macros is None:
            undef_macros = []
        if f77_flags is None:
            f77_flags = []
        if f90_flags is None:
            f90_flags = []
        if opt_flags is None:
            opt_flags = []
        if arch_flags is None:
            arch_flags = []
        super().__init__(
            module_name,
            fortran_compiler,
            c_compiler,
            f77exec,
            f90exec,
            f77_flags,
            f90_flags,
            include_paths,
            include_dirs,
            external_resources,
            linker_libpath,
            linker_libname,
            define_macros,
            undef_macros,
            debug,
            opt_flags,
            arch_flags,
            no_opt,
            no_arch,
        )

        self.wrappers: list[Path] = []
        self.fortran_sources: list[Path] = []
        self.template = Template(self.fortran_sources)

    def _get_optimization_level(self):
        if self.no_arch and not self.no_opt:
            return 2
        elif self.no_opt:
            return 0
        return 3

    def _set_environment_variables(self) -> None:
        if self.fortran_compiler:
            os.putenv("FC", self.fortran_compiler)
        elif self.f77exec:
            os.putenv("FC", self.f77exec)
        elif self.f90exec:
            os.putenv("FC", self.f90exec)
        if self.c_compiler:
            os.putenv("CC", self.c_compiler)

    def _move_exec_to_root(self, build_dir: Path):
        walk_dir = build_dir / self.meson_build_dir
        path_objects = walk_dir.glob(f"{self.module_name}*.so")
        for path_object in path_objects:
            shutil.move(path_object, Path.cwd())

    def _get_build_command(self):
        return [
            "meson",
            "setup",
            self.meson_build_dir,
            "-Ddebug=true" if self.debug else "-Ddebug=false",
            f"-Doptimization={str(self._get_optimization_level())}",
        ]

    def load_wrapper(self, wrappers: list[Path]) -> None:
        self.wrappers = wrappers

    def load_sources(self, fortran_sources: list[Path]) -> None:
        for fortran_source in fortran_sources:
            fortran_source = Path(fortran_source)
            if not fortran_source.is_file():
                raise FileNotFoundError(
                    errno.ENOENT, f"{fortran_source.absolute()} does not exist."
                )
            self.fortran_sources.append(fortran_source)

    def write_meson_build(self, build_dir: Path) -> None:
        """Writes the meson build file at specified location"""
        meson_template = MesonTemplate(
            self.module_name,
            super().numpy_install_path(),
            self.numpy_get_include(),
            self.f2py_get_include(),
            self.wrappers,
            self.fortran_sources,
            self.external_resources,
            self.include_paths + self.include_dirs,
            optimization_flags=self.opt_flags,
            architecture_flags=self.arch_flags,
            f77_flags=self.f77_flags,
            f90_flags=self.f90_flags,
            linker_libpath=self.linker_libpath,
            linker_libname=self.linker_libname,
            define_macros=self.define_macros,
            undef_macros=self.undef_macros,
        )
        src = meson_template.generate_meson_build()
        meson_build_file = build_dir / "meson.build"
        meson_build_file.write_text(src)
        return meson_build_file

    def run_meson(self, build_dir: Path):
        self._set_environment_variables()
        completed_process = subprocess.run(self._get_build_command(), cwd=build_dir)
        if completed_process.returncode != 0:
            raise subprocess.CalledProcessError(
                completed_process.returncode, completed_process.args
            )
        completed_process = subprocess.run(
            ["meson", "compile", "-C", self.meson_build_dir], cwd=build_dir
        )
        if completed_process.returncode != 0:
            raise subprocess.CalledProcessError(
                completed_process.returncode, completed_process.args
            )

    def compile(
        self,
        f77_sources: list[Path],
        f90_sources: list[Path],
        object_files: list[Path],
        wrappers: list[Path],
        build_dir: Path,
    ) -> None:
        self.load_wrapper(wrappers)
        self.load_sources(f77_sources + f90_sources + object_files)
        self.write_meson_build(build_dir)
        self.run_meson(build_dir)
        self._move_exec_to_root(build_dir)
