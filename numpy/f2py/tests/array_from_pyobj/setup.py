import os
def configuration(parent_name='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('array_from_pyobj',parent_name,top_path)
    #import numpy.f2py as f2py
    #f2pydir=os.path.dirname(os.path.abspath(f2py.__file__))
    f2pydir=os.path.join(config.local_path,'..','..')
    fobjhsrc = os.path.join(f2pydir,'src','fortranobject.h')
    fobjcsrc = os.path.join(f2pydir,'src','fortranobject.c')
    config.add_extension('wrap',
                         sources = ['wrapmodule.c',fobjcsrc],
                         include_dirs = [os.path.dirname(fobjhsrc)],
                         depends = [fobjhsrc,fobjcsrc],
                         define_macros = [('DEBUG_COPY_ND_ARRAY',1),
                                          #('F2PY_REPORT_ON_ARRAY_COPY',1),
                                          #('F2PY_REPORT_ATEXIT',1)
                                          ]
                         )

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
