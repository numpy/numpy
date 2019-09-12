from os import path

__version__ = '0.0.1'
__version_full__ = __version__


def get_html_theme_path():
    """Return list of HTML theme paths."""
    cur_dir = path.abspath(path.dirname(path.dirname(__file__)))
    return cur_dir

def setup(app):
    app.add_html_theme('custom_sphinx_theme', path.abspath(path.dirname(__file__)))
