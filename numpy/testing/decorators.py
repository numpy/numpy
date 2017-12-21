"""
Back compatibility decorators module. It will import the appropriate
set of tools

"""
from .nose_tools.decorators import (
    SkipTest, absolute_import, assert_warns, collections, deprecated, division,
    knownfailureif, parametrize, print_function, setastest, skipif, slow)
