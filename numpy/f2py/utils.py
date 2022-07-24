"""Global f2py utilities."""

import contextlib
import tempfile
import pathlib
import shutil

def get_f2py_dir():
	"""Return the directory where f2py is installed."""
	return pathlib.Path(__file__).resolve().parent

@contextlib.contextmanager
def open_build_dir(build_dir: list or None, compile: bool):
	remove_build_dir: bool = False
	if(type(build_dir) is list and build_dir[0] is not None):
		build_dir = build_dir[0]
	try:
		if build_dir is None:
			if(compile):
				remove_build_dir = True
				build_dir = tempfile.mkdtemp()
			else:
				build_dir = pathlib.Path.cwd()
		else:
			build_dir = pathlib.Path(build_dir).absolute()
		yield build_dir
	finally:
		shutil.rmtree(build_dir) if remove_build_dir else None