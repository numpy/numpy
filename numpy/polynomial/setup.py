import os

from polytemplate import polytemplate

def generate_from_template(base_file, name, nick, domain):
    """Generate polynomial class from a template."""
    base_path = os.path.join('numpy', 'polynomial')

    fp = open(os.path.join(base_path, base_file))
    lines = fp.readlines()
    fp.close()

    lines.append(polytemplate.substitute(name=name,
                                         nick=nick, domain=domain))

    fp = open(os.path.join(base_path, base_file[1:]), 'w')
    for line in lines:
        fp.write(line)
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
