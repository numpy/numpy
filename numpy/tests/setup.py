def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('tests', parent_package, top_path)
    config.add_data_dir('pass')
    config.add_data_dir('fail')
    config.add_data_dir('reveal')
    config.add_data_files('mypy.ini')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
