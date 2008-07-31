
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    config = Configuration('linalg',parent_package,top_path)

    config.add_data_dir('tests')

    config.add_sconscript('SConstruct',
                          source_files = ['lapack_litemodule.c',
                                          'zlapack_lite.c', 'dlapack_lite.c',
                                          'blas_lite.c', 'dlamch.c',
                                          'f2c_lite.c','f2c.h'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
