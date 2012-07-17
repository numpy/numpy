import os
import sys
from distutils import util

from polytemplate import polytemplate

def get_build_dir():
    """Determine build directory."""
    plat = util.get_platform()
    py_ver = '%s.%s' % (sys.version_info[0], sys.version_info[1])

    return os.path.join('build',
                        'lib.%s-%s' % (plat, py_ver), 'numpy', 'polynomial')

def generate_from_template(base_file, name, nick, domain):
    """Generate polynomial class from the template."""
    base_path = os.path.join('numpy', 'polynomial')

    fp = open(os.path.join(base_path, base_file))
    lines = fp.read()
    fp.close()

    build_path = get_build_dir()
    if not os.path.isdir(build_path):
        # Distutils doesn't create the build directory until after
        # this code has been executed.
        os.makedirs(build_path)

    fp = open(os.path.join(build_path, base_file[1:]), 'w')
    fp.write(lines)
    fp.write(polytemplate.substitute(name=name,
                                     nick=nick, domain=domain))
    fp.close()

def configuration(parent_package='',top_path=None):
    generate_from_template('_chebyshev.py',
                           name='Chebyshev', nick='cheb', domain='[-1,1]')
    generate_from_template('_hermite.py',
                           name='Hermite', nick='herm', domain='[-1,1]')
    generate_from_template('_hermite_e.py',
                           name='HermiteE', nick='herme', domain='[-1,1]')
    generate_from_template('_laguerre.py',
                           name='Laguerre', nick='lag', domain='[-1,1]')
    generate_from_template('_legendre.py',
                           name='Legendre', nick='leg', domain='[-1,1]')
    generate_from_template('_polynomial.py',
                           name='Polynomial', nick='poly', domain='[-1,1]')

    from numpy.distutils.misc_util import Configuration
    config = Configuration('polynomial',parent_package,top_path)
    config.add_data_dir('tests')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
