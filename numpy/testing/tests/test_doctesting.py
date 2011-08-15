""" Doctests for NumPy-specific nose/doctest modifications
"""
# try the #random directive on the output line
def check_random_directive():
    '''
    >>> 2+2
    <BadExample object at 0x084D05AC>  #random: may vary on your system
    '''

# check the implicit "import numpy as np"
def check_implicit_np():
    '''
    >>> np.array([1,2,3])
    array([1, 2, 3])
    '''

# there's some extraneous whitespace around the correct responses
def check_whitespace_enabled():
    '''
    # whitespace after the 3
    >>> 1+2
    3

    # whitespace before the 7
    >>> 3+4
     7
    '''


if __name__ == '__main__':
    # Run tests outside numpy test rig
    import nose
    from numpy.testing.noseclasses import NumpyDoctest
    argv = ['', __file__, '--with-numpydoctest']
    nose.core.TestProgram(argv=argv, addplugins=[NumpyDoctest()])
