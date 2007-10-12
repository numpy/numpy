
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    return Configuration('oldnumeric',parent_package,top_path)

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
