from os.path import join, split, dirname
import os
import sys
from distutils.dep_util import newer
from distutils.msvccompiler import get_build_version as get_msvc_build_version

def needs_mingw_ftime_workaround(config):
    # We need the mingw workaround for _ftime if the msvc runtime version is
    # 7.1 or above and we build with mingw
    if config.compiler.compiler_type == 'mingw32':
        msver = get_msvc_build_version()
        if msver and msver > 7:
            return True

    return False

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_mathlibs
    config = Configuration('random',parent_package,top_path)

    def generate_libraries(ext, build_dir):
        config_cmd = config.get_config_cmd()
        libs = get_mathlibs()
        tc = testcode_wincrypt()
        if config_cmd.try_run(tc):
            libs.append('Advapi32')
        ext.libraries.extend(libs)
        return None

    def generate_config_h(ext, build_dir):
        defs = []
        target = join(build_dir, "mtrand", 'config.h')
        dir = dirname(target)
        if not os.path.exists(dir):
            os.makedirs(dir)

        config_cmd = config.get_config_cmd()
        if needs_mingw_ftime_workaround(config_cmd):
            defs.append("NPY_NEEDS_MINGW_TIME_WORKAROUND")

        if newer(__file__, target):
            target_f = open(target, 'a')
            for d in defs:
                if isinstance(d, str):
                    target_f.write('#define %s\n' % (d))
            target_f.close()

    libs = []
    # Configure mtrand
    config.add_extension('mtrand',
                         sources=[join('mtrand', x) for x in
                                  ['mtrand.c', 'randomkit.c', 'initarray.c',
                                   'distributions.c']]+[generate_libraries]
                                   + [generate_config_h],
                         libraries=libs,
                         depends = [join('mtrand','*.h'),
                                    join('mtrand','*.pyx'),
                                    join('mtrand','*.pxi'),
                                    ]
                        )

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
