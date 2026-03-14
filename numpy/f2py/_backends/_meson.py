import errno
import os
import re
import shutil
import subprocess
import sys
from itertools import chain
from pathlib import Path
from string import Template

from ._backend import Backend


class MesonTemplate:
    """Template meson build file generation class."""

    def __init__(
        self,
        modulename: str,
        sources: list[Path],
        deps: list[str],
        libraries: list[str],
        library_dirs: list[Path],
        include_dirs: list[Path],
        object_files: list[Path],
        linker_args: list[str],
        fortran_args: list[str],
        build_type: str,
        python_exe: str,
    ):
        self.modulename = modulename
        self.build_template_path = (
            Path(__file__).parent.absolute() / "meson.build.template"
        )
        self.sources = sources
        self.deps = deps
        self.libraries = libraries
        self.library_dirs = library_dirs
        if include_dirs is not None:
            self.include_dirs = include_dirs
        else:
            self.include_dirs = []
        self.substitutions = {}
        self.objects = object_files
        # Convert args to '' wrapped variant for meson
        self.fortran_args = [
            f"'{x}'" if not (x.startswith("'") and x.endswith("'")) else x
            for x in fortran_args
        ]
        self.has_openmp = self._detect_openmp(fortran_args)
        self.openmp_lib_dir = None
        if self.has_openmp:
            self.openmp_lib_dir = self._find_openmp_library()

        self.pipeline = [
            self.initialize_template,
            self.sources_substitution,
            self.objects_substitution,
            self.deps_substitution,
            self.include_substitution,
            self.libraries_substitution,
            self.fortran_args_substitution,
            self.rpath_substitution,  #Add rpath handling
            self.link_language_substitution,  #For Intel compiler
        ]
        self.build_type = build_type
        self.python_exe = python_exe
        self.indent = " " * 21
    
    def _detect_openmp(self, fortran_args: list[str]) -> bool:
        """Detect if OpenMP flags are present in fortran_args"""
        openmp_flags = ['-fopenmp', '-qopenmp', '-openmp', '/Qopenmp']
        return any(flag in fortran_args for flag in openmp_flags)
    
    def _find_openmp_library(self) -> Path | None:
        """Try to find the OpenMP library directory from the compiler"""
        try:
            # Determine which compiler we're using
            fc = os.environ.get('FC', 'gfortran')
            
            # Try to get library path from compiler
            if 'gfortran' in fc or 'gcc' in fc:
                # For GCC/gfortran, find libgomp
                result = subprocess.run(
                    [fc, '--print-file-name=libgomp.so'],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0 and result.stdout.strip():
                    lib_path = Path(result.stdout.strip())
                    if lib_path.exists():
                        return lib_path.parent
                
                # Try macOS dylib
                result = subprocess.run(
                    [fc, '--print-file-name=libgomp.dylib'],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0 and result.stdout.strip():
                    lib_path = Path(result.stdout.strip())
                    if lib_path.exists():
                        return lib_path.parent
                        
            elif 'ifort' in fc or 'ifx' in fc:
                # For Intel Fortran, find libiomp5
                result = subprocess.run(
                    [fc, '--print-file-name=libiomp5.so'],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0 and result.stdout.strip():
                    lib_path = Path(result.stdout.strip())
                    if lib_path.exists():
                        return lib_path.parent
        except Exception:
            pass
        return None
        

    def meson_build_template(self) -> str:
        if not self.build_template_path.is_file():
            raise FileNotFoundError(
                errno.ENOENT,
                "Meson build template"
                f" {self.build_template_path.absolute()}"
                " does not exist.",
            )
        return self.build_template_path.read_text()

    def initialize_template(self) -> None:
        self.substitutions["modulename"] = self.modulename
        self.substitutions["buildtype"] = self.build_type
        self.substitutions["python"] = self.python_exe

    def sources_substitution(self) -> None:
        self.substitutions["source_list"] = ",\n".join(
            [f"{self.indent}'''{source}'''," for source in self.sources]
        )

    def objects_substitution(self) -> None:
        self.substitutions["obj_list"] = ",\n".join(
            [f"{self.indent}'''{obj}'''," for obj in self.objects]
        )

    def deps_substitution(self) -> None:
        # Handle OpenMP dependency with fallback
        deps_list = []
        openmp_handled = False
        
        for dep in self.deps:
            if dep.lower() == 'openmp':
                deps_list.append(f"{self.indent}openmp_dep,")
                openmp_handled=True
            else:
                deps_list.append(
                    f"{self.indent}dependency('{dep}'),"
                )
        
        # If OpenMP was detected in flags but not explicitly in deps, add it
        if self.has_openmp and not openmp_handled:
            deps_list.append(f"{self.indent}openmp_dep,")
        
        self.substitutions["dep_list"] = f"\n".join(deps_list)
        
        #Add OpenMP dependency declaration with fallback
        if self.has_openmp or openmp_handled:
            openmp_fallback = self._generate_openmp_fallback()
            self.substitutions["openmp_declaration"] = openmp_fallback
        else:
            self.substitutions["openmp_declaration"] = ""

    def _generate_openmp_fallback(self) -> str:
        """Generate OpenMP dependency with fallback for when Meson can't find it"""
        lines = [
            "# OpenMP dependency with fallback",
            "openmp_dep = dependency('openmp', required: false)",
            "if not openmp_dep.found()",
            "  # Fallback: use compiler flags directly",
        ]
        
        if self.openmp_lib_dir:
            lines.append(f"  # Detected OpenMP library at: {self.openmp_lib_dir}")
        
        # Determine the OpenMP library based on compiler/flags
        openmp_lib = 'gomp'  # Default for GCC
        is_intel = any('ifort' in str(arg) or 'ifx' in str(arg) or 'qopenmp' in str(arg).lower() 
               for arg in self.fortran_args)
        if is_intel:
            openmp_lib = 'iomp5'

        lines.append("  openmp_dep = declare_dependency(")

        # Add compile args
        compile_args = []
        for arg in self.fortran_args:
            arg_stripped = arg.strip("'\"")
            if any(omp_flag in arg_stripped for omp_flag in ['-fopenmp', '-qopenmp', '-openmp', '/Qopenmp']):
                compile_args.append(arg)

        if compile_args:
            lines.append(f"    compile_args: [{', '.join(compile_args)}],")

        # Add link args
        link_args = []
        if self.openmp_lib_dir:
            link_args.append(f"'-L{self.openmp_lib_dir}'")
        link_args.append(f"'-l{openmp_lib}'")

        if is_intel and compile_args:
            link_args.extend(compile_args)

        lines.append(f"    link_args: [{', '.join(link_args)}],")
        lines.append("  )")
        lines.append("endif")

        return "\n".join(lines)

    def libraries_substitution(self) -> None:
        self.substitutions["lib_dir_declarations"] = "\n".join(
            [
                f"lib_dir_{i} = declare_dependency(link_args : ['''-L{lib_dir}'''])"
                for i, lib_dir in enumerate(self.library_dirs)
            ]
        )

        self.substitutions["lib_declarations"] = "\n".join(
            [
                f"{lib.replace('.', '_')} = declare_dependency(link_args : ['-l{lib}'])"
                for lib in self.libraries
            ]
        )

        self.substitutions["lib_list"] = f"\n{self.indent}".join(
            [f"{self.indent}{lib.replace('.', '_')}," for lib in self.libraries]
        )
        self.substitutions["lib_dir_list"] = f"\n{self.indent}".join(
            [f"{self.indent}lib_dir_{i}," for i in range(len(self.library_dirs))]
        )

    def include_substitution(self) -> None:
        self.substitutions["inc_list"] = f",\n{self.indent}".join(
            [f"{self.indent}'''{inc}'''," for inc in self.include_dirs]
        )

    def fortran_args_substitution(self) -> None:
        if self.fortran_args:
            self.substitutions["fortran_args"] = (
                f"{self.indent}fortran_args: [{', '.join(list(self.fortran_args))}],"
            )
        else:
            self.substitutions["fortran_args"] = ""

    def rpath_substitution(self) -> None:
        """NEW: Generate rpath configuration for runtime library loading"""
        rpath_dirs = []

        # Add all library directories
        rpath_dirs.extend([str(lib_dir) for lib_dir in self.library_dirs])

        # Add OpenMP library directory if detected
        if self.openmp_lib_dir:
            rpath_dirs.append(str(self.openmp_lib_dir))

        if rpath_dirs:
            # Remove duplicates while preserving order
            unique_rpath_dirs = list(dict.fromkeys(rpath_dirs))
            rpath_list = ", ".join([f"'{path}'" for path in unique_rpath_dirs])
            self.substitutions["rpath"] = f"{self.indent}install_rpath: [{rpath_list}],"
        else:
            self.substitutions["rpath"] = ""

    def link_language_substitution(self) -> None:
        """Force Fortran as link language when OpenMP is used with Intel compilers"""
        # Intel compilers need Fortran linker for OpenMP, not C linker
        is_intel = any('ifort' in str(arg) or 'ifx' in str(arg) or 'qopenmp' in str(arg).lower() 
                       for arg in self.fortran_args)

        if self.has_openmp and is_intel:
            self.substitutions["override_options"] = f"{self.indent}override_options: ['fortran_link_language=fortran'],"
        else:
            self.substitutions["override_options"] = ""

    def generate_meson_build(self):
        for node in self.pipeline:
            node()
        template = Template(self.meson_build_template())
        meson_build = template.substitute(self.substitutions)
        meson_build = meson_build.replace(",,", ",")
        return meson_build

class MesonBackend(Backend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dependencies = self.extra_dat.get("dependencies", [])
        self.meson_build_dir = "bbdir"
        self.build_type = (
            "debug" if any("debug" in flag for flag in self.fc_flags) else "release"
        )
        self.fc_flags = _get_flags(self.fc_flags)

    def _move_exec_to_root(self, build_dir: Path):
        walk_dir = Path(build_dir) / self.meson_build_dir
        path_objects = chain(
            walk_dir.glob(f"{self.modulename}*.so"),
            walk_dir.glob(f"{self.modulename}*.pyd"),
            walk_dir.glob(f"{self.modulename}*.dll"),
        )
        # Same behavior as distutils
        # https://github.com/numpy/numpy/issues/24874#issuecomment-1835632293
        for path_object in path_objects:
            dest_path = Path.cwd() / path_object.name
            if dest_path.exists():
                dest_path.unlink()
            shutil.copy2(path_object, dest_path)
            os.remove(path_object)

    def write_meson_build(self, build_dir: Path) -> None:
        """Writes the meson build file at specified location"""
        meson_template = MesonTemplate(
            self.modulename,
            self.sources,
            self.dependencies,
            self.libraries,
            self.library_dirs,
            self.include_dirs,
            self.extra_objects,
            self.flib_flags,
            self.fc_flags,
            self.build_type,
            sys.executable,
        )
        src = meson_template.generate_meson_build()
        Path(build_dir).mkdir(parents=True, exist_ok=True)
        meson_build_file = Path(build_dir) / "meson.build"
        meson_build_file.write_text(src)
        return meson_build_file

    def _run_subprocess_command(self, command, cwd):
        subprocess.run(command, cwd=cwd, check=True)

    def run_meson(self, build_dir: Path):
        setup_command = ["meson", "setup", self.meson_build_dir]
        self._run_subprocess_command(setup_command, build_dir)
        compile_command = ["meson", "compile", "-C", self.meson_build_dir]
        self._run_subprocess_command(compile_command, build_dir)

    def compile(self) -> None:
        self.sources = _prepare_sources(self.modulename, self.sources, self.build_dir)
        _prepare_objects(self.modulename, self.extra_objects, self.build_dir)
        self.write_meson_build(self.build_dir)
        self.run_meson(self.build_dir)
        self._move_exec_to_root(self.build_dir)


def _prepare_sources(mname, sources, bdir):
    extended_sources = sources.copy()
    Path(bdir).mkdir(parents=True, exist_ok=True)
    # Copy sources
    for source in sources:
        if Path(source).exists() and Path(source).is_file():
            shutil.copy(source, bdir)
    generated_sources = [
        Path(f"{mname}module.c"),
        Path(f"{mname}-f2pywrappers2.f90"),
        Path(f"{mname}-f2pywrappers.f"),
    ]
    bdir = Path(bdir)
    for generated_source in generated_sources:
        if generated_source.exists():
            shutil.copy(generated_source, bdir / generated_source.name)
            extended_sources.append(generated_source.name)
            generated_source.unlink()
    extended_sources = [
        Path(source).name
        for source in extended_sources
        if not Path(source).suffix == ".pyf"
    ]
    return extended_sources

def _prepare_objects(mname, objects, bdir):
    Path(bdir).mkdir(parents=True, exist_ok=True)
    # Copy objects
    for obj in objects:
        if Path(obj).exists() and Path(obj).is_file():
            shutil.copy(obj, bdir)

def _get_flags(fc_flags):
    flag_values = []
    flag_pattern = re.compile(r"--f(77|90)flags=(.*)")
    for flag in fc_flags:
        match_result = flag_pattern.match(flag)
        if match_result:
            values = match_result.group(2).strip().split()
            values = [val.strip("'\"") for val in values]
            flag_values.extend(values)
    # Hacky way to preserve order of flags
    unique_flags = list(dict.fromkeys(flag_values))
    return unique_flags
