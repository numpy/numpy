from setuptools import setup
from io import open
from custom_sphinx_theme import __version__

setup(
    name = 'custom_sphinx_theme',
    version =__version__,
    url="https://github.com/shekharrajak/scipy-sphinx-theme-v2",
    description='Custom Sphinx Theme',
    py_modules = ['custom_sphinx_theme'],
    packages = ['custom_sphinx_theme'],
    include_package_data=True,
    zip_safe=False,
    package_data={'custom_sphinx_theme': [
        'theme.conf',
        '*.html',
        'static/css/*.css',
        'static/js/*.js',
        'static/fonts/*.*',
        'static/images/*.*',
        'theme_variables.jinja'
    ]},
    entry_points = {
        'sphinx.html_themes': [
            'custom_sphinx_theme = custom_sphinx_theme',
        ]
    },
    license= 'MIT License',
    classifiers=[
    ],
    install_requires=[
       'sphinx'
    ]
)
