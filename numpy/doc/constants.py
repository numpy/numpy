"""
=========
Constants
=========

Numpy includes several constants:

%(constant_list)s
"""
import textwrap

# Maintain same format as in numpy.add_newdocs
constants = []
def add_newdoc(module, name, doc):
    constants.append((name, doc))

add_newdoc('numpy', 'Inf',
    """
    """)

add_newdoc('numpy', 'Infinity',
    """
    """)

add_newdoc('numpy', 'NAN',
    """
    """)

add_newdoc('numpy', 'NINF',
    """
    """)

add_newdoc('numpy', 'NZERO',
    """
    """)

add_newdoc('numpy', 'NaN',
    """
    """)

add_newdoc('numpy', 'PINF',
    """
    """)

add_newdoc('numpy', 'PZERO',
    """
    """)

add_newdoc('numpy', 'e',
    """
    """)

add_newdoc('numpy', 'inf',
    """
    """)

add_newdoc('numpy', 'infty',
    """
    """)

add_newdoc('numpy', 'nan',
    """
    """)

add_newdoc('numpy', 'newaxis',
    """
    """)

if __doc__:
    constants_str = []
    constants.sort()
    for name, doc in constants:
        constants_str.append(""".. const:: %s\n    %s""" % (
            name, textwrap.dedent(doc).replace("\n", "\n    ")))
    constants_str = "\n".join(constants_str)

    __doc__ = __doc__ % dict(constant_list=constants_str)
    del constants_str, name, doc

del constants, add_newdoc
