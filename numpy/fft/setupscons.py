def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('fft', parent_package, top_path)

    config.add_data_dir('tests')

    config.add_sconscript('SConstruct',
                          source_files = ['fftpack_litemodule.c', 'fftpack.c',
                                          'fftpack.h'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
