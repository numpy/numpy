"""Global f2py utilities."""

import os

def get_f2py_dir():
	"""Return the directory where f2py is installed."""
	return os.path.dirname(os.path.abspath(__file__))