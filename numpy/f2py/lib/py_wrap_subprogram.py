
__all__ = ['PythonCAPISubProgram']

import sys

from wrapper_base import *

class PythonCAPISubProgram(WrapperBase):
    """
    Fortran subprogram hooks.
    """
    _defined = []
    def __init__(self, parent, block):
        WrapperBase.__init__(self)
        self.name = name = block.name
        if name in self._defined:
            return
        self._defined.append(name)
        self.info('Generating interface for %s: %s' % (block.__class__, name))


        raise NotImplementedError,`name,block.__class__`
        return
