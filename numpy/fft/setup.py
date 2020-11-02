import sys

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('fft', parent_package, top_path)

    config.add_subpackage('tests')

    # AIX needs to be told to use large file support - at all times
    defs = [('_LARGE_FILES', None)] if sys.platform[:3] == "aix" else []
    # Configure pocketfft_internal
    EXTRA_COMPILE_ARGS = []
    EXTRA_LINK_ARGS = []
    if sys.platform == 'OpenVMS':
        # EXTRA_LINK_ARGS = ['/DEBUG']
        EXTRA_COMPILE_ARGS = [
            # '/DEBUG/NOOPTIMIZE/LIST',
            # '/POINTER_SIZE=32',
            '/WARN=DISABLE=BADALIAS',
            ]
    config.add_extension('_pocketfft_internal',
                         sources=['_pocketfft.c'],
                         define_macros=defs,
                         extra_compile_args=EXTRA_COMPILE_ARGS,
                         extra_link_args=EXTRA_LINK_ARGS,
                         )

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
