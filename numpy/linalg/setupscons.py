
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    config = Configuration('linalg',parent_package,top_path)

    config.add_data_dir('tests')

    print "### Warning:  Using unoptimized lapack ###"

    config.add_sconscript('SConstruct')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
