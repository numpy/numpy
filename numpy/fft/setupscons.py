def configuration(parent_package = '', top_path = None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('fft', parent_package, top_path)

    config.add_data_dir('tests')

    print "!!!!!! %s !!!!!!!!!" % get_numpy_include_dirs()

    # Configure fftpack_lite
    config.add_sconscript('SConstruct')
    config.add_extension('fftpack_lite',
                         sources=['fftpack_litemodule.c', 'fftpack.c']
                         )


    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
