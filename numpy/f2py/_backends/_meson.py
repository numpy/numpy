from __future__ import annotations

import errno
import shutil
import subprocess
from pathlib import Path

from ._backend import Backend
from string import Template

import warnings


class MesonTemplate:
    """Template meson build file generation class."""

    def __init__(
        self,
        modulename: str,
        sources: list[Path],
        object_files: list[Path],
        linker_args: list[str],
        c_args: list[str],
    ):
        self.modulename = modulename
        self.build_template_path = (
            Path(__file__).parent.absolute() / "src" / "meson.build.src"
        )
        self.sources = sources
        self.substitutions = {}
        self.objects = object_files
        self.pipeline = [
            self.initialize_template,
            self.sources_substitution,
        ]

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
        """Initialize with module name."""
        self.substitutions["modulename"] = self.modulename

    def sources_substitution(self) -> None:
        self.substitutions["source_list"] = ",\n".join(
            [f"                     '{source}'" for source in self.sources]
        )

    def generate_meson_build(self):
        for node in self.pipeline:
            node()
        template = Template(self.meson_build_template())
        return template.substitute(self.substitutions)


class MesonBackend(Backend):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The MesonBackend is experimental and may change in the future."
            "--build-dir should be used to customize the generated skeleton.",
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
        self.meson_build_dir = "bbdir"

    def _move_exec_to_root(self, build_dir: Path):
        walk_dir = Path(build_dir) / self.meson_build_dir
        path_objects = walk_dir.glob(f"{self.modulename}*.so")
        for path_object in path_objects:
            shutil.move(path_object, Path.cwd())

    def _get_build_command(self):
        return [
            "meson",
            "setup",
            self.meson_build_dir,
        ]

    def write_meson_build(self, build_dir: Path) -> None:
        """Writes the meson build file at specified location"""
        meson_template = MesonTemplate(
            self.modulename,
            self.sources,
            self.extra_objects,
            self.flib_flags,
            self.fc_flags,
        )
        src = meson_template.generate_meson_build()
        Path(build_dir).mkdir(parents=True, exist_ok=True)
        meson_build_file = Path(build_dir) / "meson.build"
        meson_build_file.write_text(src)
        return meson_build_file

    def run_meson(self, build_dir: Path):
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

    def compile(self) -> None:
        self.sources = _prepare_sources(self.modulename, self.sources, self.build_dir)
        self.write_meson_build(self.build_dir)
        self.run_meson(self.build_dir)
        self._move_exec_to_root(self.build_dir)


def _prepare_sources(mname, sources, bdir):
    extended_sources = sources.copy()
    Path(bdir).mkdir(parents=True, exist_ok=True)
    # Copy sources
    for source in sources:
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
