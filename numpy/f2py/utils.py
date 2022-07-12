"""Global f2py utilities."""

import os
import contextlib
import tempfile
import pathlib
import shutil

def get_f2py_dir():
	"""Return the directory where f2py is installed."""
	return os.path.dirname(os.path.abspath(__file__))

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