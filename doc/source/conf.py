# -*- coding: utf-8 -*-

import sys, os, re

# Check Sphinx version
import sphinx
if sphinx.__version__ < "1.0.1":
    raise RuntimeError("Sphinx 1.0.1 or newer required")

needs_sphinx = '1.0'

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.

sys.path.insert(0, os.path.abspath('../sphinxext'))

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.pngmath', 'numpydoc',
              'sphinx.ext.intersphinx', 'sphinx.ext.coverage',
              'sphinx.ext.doctest', 'sphinx.ext.autosummary']

# Determine if the matplotlib has a recent enough version of the
# plot_directive, otherwise use the local fork.
try:
    from matplotlib.sphinxext import plot_directive
except ImportError:
    use_matplotlib_plot_directive = False
else:
    try:
        use_matplotlib_plot_directive = (plot_directive.__version__ >= 2)
    except AttributeError:
        use_matplotlib_plot_directive = False

if use_matplotlib_plot_directive:
    extensions.append('matplotlib.sphinxext.plot_directive')
else:
    extensions.append('plot_directive')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
#master_doc = 'index'

# General substitutions.
project = 'NumPy'
copyright = '2008-2009, The Scipy community'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
import numpy
# The short X.Y version (including .devXXXX, rcX, b1 suffixes if present)
version = re.sub(r'(\d+\.\d+)\.\d+(.*)', r'\1\2', numpy.__version__)
version = re.sub(r'(\.dev\d+).*?$', r'\1', version)
# The full version, including alpha/beta/rc tags.
release = numpy.__version__
print version, release

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


# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
html_style = 'scipy.css'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "%s v%s Manual (DRAFT)" % (project, version)

# The name of an image file (within the static path) to place at the top of
# the sidebar.
html_logo = 'scipyshiny_small.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    'index': 'indexsidebar.html'
}

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {
    'index': 'indexcontent.html',
}

# If false, no module index is generated.
html_use_modindex = True

# If true, the reST sources are included in the HTML build as _sources/<name>.
#html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".html").
#html_file_suffix = '.html'

# Output file base name for HTML help builder.
htmlhelp_basename = 'numpy'

# Pngmath should try to align formulas properly
pngmath_use_preview = True


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
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {'http://docs.python.org/dev': None}


# -----------------------------------------------------------------------------
# Numpy extensions
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

if not use_matplotlib_plot_directive:
    import matplotlib
    matplotlib.rcParams.update(plot_rcparams)

# -----------------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------------

import inspect
from os.path import relpath, dirname

for name in ['sphinx.ext.linkcode', 'linkcode', 'numpydoc.linkcode']:
    try:
        __import__(name)
        extensions.append(name)
        break
    except ImportError:
        pass
else:
    print "NOTE: linkcode extension not found -- no links to source generated"

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
        except:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.findsource(obj)
    except:
        lineno = None

    if lineno:
        linespec = "#L%d" % (lineno + 1)
    else:
        linespec = ""

    fn = relpath(fn, start=dirname(numpy.__file__))

    if 'dev' in numpy.__version__:
        return "http://github.com/numpy/numpy/blob/master/numpy/%s%s" % (
           fn, linespec)
    else:
        return "http://github.com/numpy/numpy/blob/v%s/numpy/%s%s" % (
           numpy.__version__, fn, linespec)
