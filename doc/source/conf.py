# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

import sys, os, re

# Check Sphinx version
import sphinx
if sphinx.__version__ < "1.2.1":
    raise RuntimeError("Sphinx 1.2.1 or newer required")

needs_sphinx = '1.0'

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.

sys.path.insert(0, os.path.abspath('../sphinxext'))

extensions = ['sphinx.ext.autodoc', 'numpydoc',
              'sphinx.ext.intersphinx', 'sphinx.ext.coverage',
              'sphinx.ext.doctest', 'sphinx.ext.autosummary',
              'sphinx.ext.graphviz', 'sphinx.ext.ifconfig',
              'matplotlib.sphinxext.plot_directive']

if sphinx.__version__ >= "1.4":
    extensions.append('sphinx.ext.imgmath')
    imgmath_image_format = 'svg'
else:
    extensions.append('sphinx.ext.pngmath')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# General substitutions.
project = 'NumPy'
copyright = '2008-2018, The SciPy community'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
import numpy
# The short X.Y version (including .devXXXX, rcX, b1 suffixes if present)
version = re.sub(r'(\d+\.\d+)\.\d+(.*)', r'\1\2', numpy.__version__)
version = re.sub(r'(\.dev\d+).*?$', r'\1', version)
# The full version, including alpha/beta/rc tags.
release = numpy.__version__
print("%s %s" % (version, release))

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
#unused_docs = []

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# List of directories, relative to source directories, that shouldn't be searched
# for source files.
exclude_dirs = []

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

def setup(app):
    # add a config value for `ifconfig` directives
    app.add_config_value('python_version_major', str(sys.version_info.major), 'env')

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

themedir = os.path.join(os.pardir, 'scipy-sphinx-theme', '_theme')
if not os.path.isdir(themedir):
    raise RuntimeError("Get the scipy-sphinx-theme first, "
                       "via git submodule init && git submodule update")

html_theme = 'scipy'
html_theme_path = [themedir]

if 'scipyorg' in tags:
    # Build for the scipy.org website
    html_theme_options = {
        "edit_link": True,
        "sidebar": "right",
        "scipy_org_logo": True,
        "rootlinks": [("http://scipy.org/", "Scipy.org"),
                      ("http://docs.scipy.org/", "Docs")]
    }
else:
    # Default build
    html_theme_options = {
        "edit_link": False,
        "sidebar": "left",
        "scipy_org_logo": False,
        "rootlinks": []
    }
    html_sidebars = {'index': ['indexsidebar.html', 'searchbox.html']}

html_additional_pages = {
    'index': 'indexcontent.html',
}

html_title = "%s v%s Manual" % (project, version)
html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'

html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = '.html'

htmlhelp_basename = 'numpy'

if 'sphinx.ext.pngmath' in extensions:
    pngmath_use_preview = True
    pngmath_dvipng_args = ['-gamma', '1.5', '-D', '96', '-bg', 'Transparent']

plot_html_show_formats = False
plot_html_show_source_link = False

# -----------------------------------------------------------------------------
# LaTeX output
# -----------------------------------------------------------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
_stdauthor = 'Written by the NumPy community'
latex_documents = [
  ('reference/index', 'numpy-ref.tex', 'NumPy Reference',
   _stdauthor, 'manual'),
  ('user/index', 'numpy-user.tex', 'NumPy User Guide',
   _stdauthor, 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Additional stuff for the LaTeX preamble.
latex_preamble = r'''
\usepackage{amsmath}
\DeclareUnicodeCharacter{00A0}{\nobreakspace}

% In the parameters section, place a newline after the Parameters
% header
\usepackage{expdlist}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}

% Make Examples/etc section headers smaller and more compact
\makeatletter
\titleformat{\paragraph}{\normalsize\py@HeaderFamily}%
            {\py@TitleColor}{0em}{\py@TitleColor}{\py@NormalColor}
\titlespacing*{\paragraph}{0pt}{1ex}{0pt}
\makeatother

% Fix footer/header
\renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{\thechapter.\ #1}}{}}
\renewcommand{\sectionmark}[1]{\markright{\MakeUppercase{\thesection.\ #1}}}
'''

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = False


# -----------------------------------------------------------------------------
# Texinfo output
# -----------------------------------------------------------------------------

texinfo_documents = [
  ("contents", 'numpy', 'NumPy Documentation', _stdauthor, 'NumPy',
   "NumPy: array processing for numbers, strings, records, and objects.",
   'Programming',
   1),
]


# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/dev', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org', None)
}


# -----------------------------------------------------------------------------
# NumPy extensions
# -----------------------------------------------------------------------------

# If we want to do a phantom import from an XML file for all autodocs
phantom_import_file = 'dump.xml'

# Make numpydoc to generate plots for example sections
numpydoc_use_plots = True

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

import glob
autosummary_generate = glob.glob("reference/*.rst")

# -----------------------------------------------------------------------------
# Coverage checker
# -----------------------------------------------------------------------------
coverage_ignore_modules = r"""
    """.split()
coverage_ignore_functions = r"""
    test($|_) (some|all)true bitwise_not cumproduct pkgload
    generic\.
    """.split()
coverage_ignore_classes = r"""
    """.split()

coverage_c_path = []
coverage_c_regexes = {}
coverage_ignore_c_items = {}


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
plot_pre_code = """
import numpy as np
np.random.seed(0)
"""
plot_include_source = True
plot_formats = [('png', 100), 'pdf']

import math
phi = (math.sqrt(5) + 1)/2

plot_rcparams = {
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3*phi, 3),
    'figure.subplot.bottom': 0.2,
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.85,
    'figure.subplot.wspace': 0.4,
    'text.usetex': False,
}


# -----------------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------------

import inspect
from os.path import relpath, dirname

for name in ['sphinx.ext.linkcode', 'numpydoc.linkcode']:
    try:
        __import__(name)
        extensions.append(name)
        break
    except ImportError:
        pass
else:
    print("NOTE: linkcode extension not found -- no links to source generated")

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    fn = relpath(fn, start=dirname(numpy.__file__))

    if 'dev' in numpy.__version__:
        return "http://github.com/numpy/numpy/blob/master/numpy/%s%s" % (
           fn, linespec)
    else:
        return "http://github.com/numpy/numpy/blob/v%s/numpy/%s%s" % (
           numpy.__version__, fn, linespec)
