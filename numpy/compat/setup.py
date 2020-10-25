def configuration(parent_package='',top_path=None):
    from numpy_distutils.misc_util import Configuration

    config = Configuration('compat', parent_package, top_path)
    config.add_subpackage('tests')
    return config

if __name__ == '__main__':
    from numpy_distutils.core import setup
    setup(configuration=configuration)
