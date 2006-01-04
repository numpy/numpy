
from os.path import join

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('random',parent_package,top_path)

    # Configure mtrand
    config.add_extension('mtrand',
                         sources=[join('mtrand', x) for x in 
                                  ['mtrand.c', 'randomkit.c', 'initarray.c',
                                   'distributions.c']],
                         libraries=['m'],
                         depends = [join('mtrand','*.h'),
                                    join('mtrand','*.pyx'),
                                    join('mtrand','*.pxi'),
                                    ]
                        )
            
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
