def configuration(parent_package='',top_path=None):
    from scipy.distutils.misc_util import Configuration
    config = Configuration('basic',parent_package,top_path)
    config.add_data_dir('tests')
    return config
