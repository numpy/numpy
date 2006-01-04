
from os.path import join

def configuration(parent_package='',top_path=None):
    from scipy.distutils.misc_util import Configuration
    config = Configuration('corefft',parent_package,top_path)

    config.add_data_dir('tests')

    # Configure fftpack_lite
    config.add_extension('fftpack_lite',
                         sources=['fftpack_litemodule.c', 'fftpack.c']
                         )


    return config

if __name__ == '__main__':
    from scipy.distutils.core import setup
    setup(**configuration(top_path='').todict())
