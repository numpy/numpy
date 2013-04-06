from __future__ import division, absolute_import, print_function

import os
import numpy as np

__all__ = ['MathDomainError', 'UnderflowError', 'NumOverflowError',
           'handleError', 'get_numarray_include_dirs']

class MathDomainError(ArithmeticError):
    pass


class UnderflowError(ArithmeticError):
    pass


class NumOverflowError(OverflowError, ArithmeticError):
    pass


def handleError(errorStatus, sourcemsg):
    """Take error status and use error mode to handle it."""
    modes = np.geterr()
    if errorStatus & np.FPE_INVALID:
        if modes['invalid'] == "warn":
            print("Warning: Encountered invalid numeric result(s)", sourcemsg)
        if modes['invalid'] == "raise":
            raise MathDomainError(sourcemsg)
    if errorStatus & np.FPE_DIVIDEBYZERO:
        if modes['dividebyzero'] == "warn":
            print("Warning: Encountered divide by zero(s)", sourcemsg)
        if modes['dividebyzero'] == "raise":
            raise ZeroDivisionError(sourcemsg)
    if errorStatus & np.FPE_OVERFLOW:
        if modes['overflow'] == "warn":
            print("Warning: Encountered overflow(s)", sourcemsg)
        if modes['overflow'] == "raise":
            raise NumOverflowError(sourcemsg)
    if errorStatus & np.FPE_UNDERFLOW:
        if modes['underflow'] == "warn":
            print("Warning: Encountered underflow(s)", sourcemsg)
        if modes['underflow'] == "raise":
            raise UnderflowError(sourcemsg)


def get_numarray_include_dirs():
    base = os.path.dirname(np.__file__)
    newdirs = [os.path.join(base, 'numarray', 'include')]
    return newdirs
