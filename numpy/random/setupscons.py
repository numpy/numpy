import glob
from os.path import join, split

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_mathlibs
    config = Configuration('random',parent_package,top_path)

    source_files = [join('mtrand', i) for i in ['mtrand.c',
                                                'mtrand.pyx',
                                                'numpy.pxi',
                                                'randomkit.c',
                                                'randomkit.h',
                                                'Python.pxi',
                                                'initarray.c',
                                                'initarray.h',
                                                'distributions.c',
                                                'distributions.h',
                                                ]]
    config.add_sconscript('SConstruct', source_files = source_files)
    config.add_data_files(('.', join('mtrand', 'randomkit.h')))
    config.add_data_dir('tests')

    return config

def testcode_wincrypt():
    return """\
/* check to see if _WIN32 is defined */
int main(int argc, char *argv[])
{
#ifdef _WIN32
    return 0;
#else
    return 1;
#endif
}
"""

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
