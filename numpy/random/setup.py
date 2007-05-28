from os.path import join, split

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_mathlibs
    config = Configuration('random',parent_package,top_path)

    def generate_libraries(ext, build_dir):
        config_cmd = config.get_config_cmd()
        if top_path is None:
            libs = get_mathlibs()
        else:
            path = join(split(build_dir)[0],'core')
            libs = get_mathlibs(path)
        tc = testcode_wincrypt()
        if config_cmd.try_run(tc):
            libs.append('Advapi32')
        ext.libraries.extend(libs)
        return None

    libs = []
    # Configure mtrand
    config.add_extension('mtrand',
                         sources=[join('mtrand', x) for x in
                                  ['mtrand.c', 'randomkit.c', 'initarray.c',
                                   'distributions.c']]+[generate_libraries],
                         libraries=libs,
                         depends = [join('mtrand','*.h'),
                                    join('mtrand','*.pyx'),
                                    join('mtrand','*.pxi'),
                                    ]
                        )

    config.add_data_files(('.', join('mtrand', 'randomkit.h')))

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
