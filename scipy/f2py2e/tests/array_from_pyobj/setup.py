
import os
def configuration(parent_name='',top_path=None):
    from scipy.distutils.misc_util import Configuration
    import scipy.f2py as f2py
    f2pydir=os.path.dirname(os.path.abspath(f2py.__file__))
    f2pydir=os.path.abspath('../../')
    fobjhsrc = os.path.join(f2pydir,'src','fortranobject.h')
    fobjcsrc = os.path.join(f2pydir,'src','fortranobject.c')
    config = Configuration('array_from_pyobj',parent_name,top_path)
    config.add_extension('wrap',
                         sources = ['wrapmodule.c',fobjcsrc],
                         include_dirs = [os.path.dirname(fobjhsrc)],
                         depends = [fobjhsrc,fobjcsrc],
                         define_macros = [('DEBUG_COPY_ND_ARRAY',1)]
                         )

    return config

if __name__ == "__main__":
    from scipy.distutils.core import setup
    setup(**configuration(top_path='').todict())
