import os
import errno
import subprocess
from pathlib import Path

from .backend import Backend
from string import Template

class MesonTemplate:
	"""Template meson build file generation class."""
	def __init__(self, module_name: str, numpy_install_path, numpy_get_include: Path, f2py_get_include: Path, c_wrapper: Path, fortran_sources: list[Path], dependencies: list[str], include_path: list[Path]):
		self.module_name = module_name
		self.numpy_install_path = numpy_install_path
		self.build_template_path = numpy_install_path / "f2py" / "backends" / "src" / "meson.build.src"
		self.sources = fortran_sources
		self.numpy_get_include = numpy_get_include
		self.f2py_get_include = f2py_get_include
		self.c_wrapper = c_wrapper
		self.dependencies = dependencies
		self.include_directories = include_path
		self.substitutions = {}
		self.pipeline = [self.initialize_template,
                   		self.sources_substitution,
                      	self.dependencies_substitution,
                       	self.include_directories_subtitution]

	@property
	def meson_build_template(self) -> str:
		if(not self.build_template_path.is_file()):
			raise FileNotFoundError(errno.ENOENT, f"Meson build template {self.build_template_path.absolute()} does not exist.")
		return self.build_template_path.read_text()

	def initialize_template(self) -> None:
		"""Initialize with module name and external NumPy and F2PY C libraries."""
		self.substitutions['modulename'] = self.module_name
		self.substitutions['numpy_get_include'] = self.numpy_get_include.absolute()
		self.substitutions['f2py_get_include'] = self.f2py_get_include.absolute()
	
	def sources_substitution(self) -> None:
		self.substitutions["source_list"] = ",".join(["\'"+str(source.absolute())+"\'" for source in self.sources])
		self.substitutions["c_wrapper"] = str(self.c_wrapper.absolute()) if self.c_wrapper else ""

	def dependencies_substitution(self) -> None:
		self.substitutions["dependencies_list"] = ", ".join([f"dependency('{dependecy}')" for dependecy in self.dependencies])
		
	def include_directories_subtitution(self) -> None:
		self.substitutions["include_directories_list"] = ", ".join([f"include_directories('{include_directory}')" for include_directory in self.include_directories])

	def generate_meson_build(self) -> str:
		for node in self.pipeline:
			node()
		template = Template(self.meson_build_template)
		return template.substitute(self.substitutions)


class MesonBackend(Backend):

	def __init__(self, module_name: str = 'untitled', fortran_compiler: str = None, c_compiler: str = None, f77exec: Path = None, f90exec: Path = None, f77_flags: list[str] = None, f90_flags: list[str] = None, include_paths: list[Path] = None, include_dirs: list[Path] = None, external_resources: list[str] = None, linker_libpath: list[Path] = None, linker_libname: list[str] = None, define_macros: list[tuple[str, str]] = None, undef_macros: list[str] = None, debug: bool = False, opt_flags: list[str] = None, arch_flags: list[str] = None, no_opt: bool = False, no_arch: bool = False) -> None:
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
		super().__init__(module_name, fortran_compiler, c_compiler, f77exec, f90exec, f77_flags, f90_flags, include_paths, include_dirs, external_resources, linker_libpath, linker_libname, define_macros, undef_macros, debug, opt_flags, arch_flags, no_opt, no_arch)

		self.c_wrapper: Path = None
		self.fortran_sources: list[Path] = []
		self.template = Template(self.fortran_sources)

	def _get_optimization_level(self):
		if self.no_arch and not self.no_opt :
			return 2
		elif self.no_opt :
			return 0
		return 3

	def _set_environment_variables(self) -> None:
		if self.fortran_compiler:
			os.putenv("FC", self.fortran_compiler)
		if self.c_compiler:
			os.putenv("CC", self.c_compiler)
	
	def _get_build_command(self):
		return ["meson", "setup", "builddir", "-Ddebug=true" if self.debug else "-Ddebug=false", f"-Doptimization={str(self._get_optimization_level())}"]
	
	def load_wrapper(self, wrapper_path: Path) -> None:
		wrapper_path: Path = Path(wrapper_path)
		if not wrapper_path.is_file():
			raise FileNotFoundError(errno.ENOENT, f"{wrapper_path.absolute()} does not exist.")
		self.c_wrapper = wrapper_path
	
	def load_sources(self, fortran_sources: list[Path]) -> None:
		for fortran_source in fortran_sources:
			fortran_source = Path(fortran_source)
			if not fortran_source.is_file():
				raise FileNotFoundError(errno.ENOENT, f"{fortran_source.absolute()} does not exist.")
			self.fortran_sources.append(fortran_source)

	def write_meson_build(self, build_dir: Path) -> None:
		"""Writes the meson build file at specified location"""
		meson_template = MesonTemplate(self.module_name, super().numpy_install_path(), self.numpy_get_include(), self.f2py_get_include(), self.c_wrapper, self.fortran_sources, self.external_resources, self.include_path+self.include_dirs)
		src = meson_template.generate_meson_build()
		meson_build_file = build_dir / "meson.build"
		meson_build_file.write_text(src)
		return meson_build_file

	def run_meson(self, build_dir: Path):
		self._set_environment_variables()
		completed_process = subprocess.run(self._get_build_command(), cwd=build_dir)
		if(completed_process.returncode != 0):
			raise subprocess.CalledProcessError(completed_process.returncode, completed_process.args)
		completed_process = subprocess.run(["meson", "compile", "-C", "builddir"], cwd=build_dir)
		if(completed_process.returncode != 0):
			raise subprocess.CalledProcessError(completed_process.returncode, completed_process.args)


	def compile(self, fortran_sources: list[Path], c_wrapper: Path, build_dir: Path) -> None:
		self.load_wrapper(c_wrapper)
		self.load_sources(fortran_sources)
		self.write_meson_build(build_dir)
		self.write_meson_build(build_dir)
		self.run_meson(build_dir)
