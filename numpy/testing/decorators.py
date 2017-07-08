"""
Back compatibility decorators module. It will import the appropriate
set of tools

"""
import os

if int(os.getenv('NPY_PYTEST', '0')):
    from .pytest_tools.decorators import *
else:
    from .nose_tools.decorators import *
