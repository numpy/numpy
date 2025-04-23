import os
import re
import sys
import importlib
from docutils import nodes
from docutils.parsers.rst import Directive
from datetime import datetime

# Minimum version, enforced by sphinx
needs_sphinx = '4.3'


# This is a nasty hack to use platform-agnostic names for types in the
# documentation.

# must be kept alive to hold the patched names
_name_cache = {}

def replace_scalar_type_names():
    """ Rename numpy types to use the canonical names to make sphinx behave """
    import ctypes

    Py_ssize_t = ctypes.c_int64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_int32

    class PyObject(ctypes.Structure):
        pass

    class PyTypeObject(ctypes.Structure):
        pass

    PyObject._fields_ = [
        ('ob_refcnt', Py_ssize_t),
        ('ob_type', ctypes.POINTER(PyTypeObject)),
    ]

    PyTypeObject._fields_ = [
        # varhead
        ('ob_base', PyObject),
        ('ob_size', Py_ssize_t),
        # declaration
        ('tp_name', ctypes.c_char_p),
    ]

    import numpy

    # change the __name__ of the scalar types
    for name in [
        'byte', 'short', 'intc', 'int_', 'longlong',
        'ubyte', 'ushort', 'uintc', 'uint', 'ulonglong',
        'half', 'single', 'double', 'longdouble',
        'half', 'csingle', 'cdouble', 'clongdouble',
    ]:
        typ = getattr(numpy, name)
        c_typ = PyTypeObject.from_address(id(typ))
        if sys.implementation.name == 'cpython':
            c_typ.tp_name = _name_cache[typ] = b"numpy." + name.encode('utf8')
        else:
            # It is not guarenteed that the c_typ has this model on other
            # implementations
            _name_cache[typ] = b"numpy." + name.encode('utf8')


replace_scalar_type_names()


# As of NumPy 1.25, a deprecation of `str`/`bytes` attributes happens.
# For some reasons, the doc build accesses these, so ignore them.
import warnings
warnings.filterwarnings("ignore", "In the future.*NumPy scalar", FutureWarning)


# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.

sys.path.insert(0, os.path.abspath('../sphinxext'))

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx.ext.imgconverter',
    'jupyterlite_sphinx',
]

skippable_extensions = [
    ('breathe', 'skip generating C/C++ API from comment blocks.'),
]
for ext, warn in skippable_extensions:
    ext_exist = importlib.util.find_spec(ext) is not None
    if ext_exist:
        extensions.append(ext)
    else:
        print(f"Unable to find Sphinx extension '{ext}', {warn}.")

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# General substitutions.
project = 'NumPy'
year = datetime.now().year
copyright = f'2008-{year}, NumPy Developers'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
import numpy
# The short X.Y version (including .devXXXX, rcX, b1 suffixes if present)
version = re.sub(r'(\d+\.\d+)\.\d+(.*)', r'\1\2', numpy.__version__)
version = re.sub(r'(\.dev\d+).*?$', r'\1', version)
# The full version, including alpha/beta/rc tags.
release = numpy.__version__
print(f"{version} {release}")

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

exclude_patterns = []
if sys.version_info[:2] >= (3, 12):
    exclude_patterns += ["reference/distutils.rst"]

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

class LegacyDirective(Directive):
    """
    Adapted from docutils/parsers/rst/directives/admonitions.py

    Uses a default text if the directive does not have contents. If it does,
    the default text is concatenated to the contents.

    See also the same implementation in SciPy's conf.py.
    """
    has_content = True
    node_class = nodes.admonition
    optional_arguments = 1

    def run(self):
        try:
            obj = self.arguments[0]
        except IndexError:
            # Argument is empty; use default text
            obj = "submodule"
        text = (f"This {obj} is considered legacy and will no longer receive "
                "updates. This could also mean it will be removed in future "
                "NumPy versions.")

        try:
            self.content[0] = text + " " + self.content[0]
        except IndexError:
            # Content is empty; use the default text
            source, lineno = self.state_machine.get_source_and_line(
                self.lineno
            )
            self.content.append(
                text,
                source=source,
                offset=lineno
            )
        text = '\n'.join(self.content)
        # Create the admonition node, to be populated by `nested_parse`
        admonition_node = self.node_class(rawsource=text)
        # Set custom title
        title_text = "Legacy"
        textnodes, _ = self.state.inline_text(title_text, self.lineno)
        title = nodes.title(title_text, '', *textnodes)
        # Set up admonition node
        admonition_node += title
        # Select custom class for CSS styling
        admonition_node['classes'] = ['admonition-legacy']
        # Parse the directive contents
        self.state.nested_parse(self.content, self.content_offset,
                                admonition_node)
        return [admonition_node]


def setup(app):
    # add a config value for `ifconfig` directives
    app.add_config_value('python_version_major', str(sys.version_info.major), 'env')
    app.add_lexer('NumPyC', NumPyLexer)
    app.add_directive("legacy", LegacyDirective)


# While these objects do have type `module`, the names are aliases for modules
# elsewhere. Sphinx does not support referring to modules by an aliases name,
# so we make the alias look like a "real" module for it.
# If we deemed it desirable, we could in future make these real modules, which
# would make `from numpy.char import split` work.
sys.modules['numpy.char'] = numpy.char

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_favicon = '_static/favicon/favicon.ico'

# Set up the version switcher.  The versions.json is stored in the doc repo.
if os.environ.get('CIRCLE_JOB') and os.environ['CIRCLE_BRANCH'] != 'main':
    # For PR, name is set to its ref
    switcher_version = os.environ['CIRCLE_BRANCH']
elif ".dev" in version:
    switcher_version = "devdocs"
else:
    switcher_version = f"{version}"

html_theme_options = {
    "logo": {
        "image_light": "_static/numpylogo.svg",
        "image_dark": "_static/numpylogo_dark.svg",
    },
    "github_url": "https://github.com/numpy/numpy",
    "collapse_navigation": True,
    "external_links": [
        {"name": "Learn", "url": "https://numpy.org/numpy-tutorials/"},
        {"name": "NEPs", "url": "https://numpy.org/neps"},
    ],
    "header_links_before_dropdown": 6,
    # Add light/dark mode and documentation version switcher:
    "navbar_end": [
        "search-button",
        "theme-switcher",
        "version-switcher",
        "navbar-icon-links"
    ],
    "navbar_persistent": [],
    "switcher": {
        "version_match": switcher_version,
        "json_url": "https://numpy.org/doc/_static/versions.json",
    },
    "show_version_warning_banner": True,
}

html_title = f"{project} v{version} Manual"
html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'
html_css_files = ["numpy.css"]
html_context = {"default_mode": "light"}
html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = '.html'

htmlhelp_basename = 'numpy'

if 'sphinx.ext.pngmath' in extensions:
    pngmath_use_preview = True
    pngmath_dvipng_args = ['-gamma', '1.5', '-D', '96', '-bg', 'Transparent']

mathjax_path = "scipy-mathjax/MathJax.js?config=scipy-mathjax"

plot_html_show_formats = False
plot_html_show_source_link = False

# sphinx-copybutton configurations
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
# -----------------------------------------------------------------------------
# LaTeX output
# -----------------------------------------------------------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# XeLaTeX for better support of unicode characters
latex_engine = 'xelatex'

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

latex_elements = {
}

# Additional stuff for the LaTeX preamble.
latex_elements['preamble'] = r'''
\newfontfamily\FontForChinese{FandolSong-Regular}[Extension=.otf]
\catcode`琴\active\protected\def琴{{\FontForChinese\string琴}}
\catcode`春\active\protected\def春{{\FontForChinese\string春}}
\catcode`鈴\active\protected\def鈴{{\FontForChinese\string鈴}}
\catcode`猫\active\protected\def猫{{\FontForChinese\string猫}}
\catcode`傅\active\protected\def傅{{\FontForChinese\string傅}}
\catcode`立\active\protected\def立{{\FontForChinese\string立}}
\catcode`业\active\protected\def业{{\FontForChinese\string业}}
\catcode`（\active\protected\def（{{\FontForChinese\string（}}
\catcode`）\active\protected\def）{{\FontForChinese\string）}}

% In the parameters section, place a newline after the Parameters
% header.  This is default with Sphinx 5.0.0+, so no need for
% the old hack then.
% Unfortunately sphinx.sty 5.0.0 did not bump its version date
% so we check rather sphinxpackagefootnote.sty (which exists
% since Sphinx 4.0.0).
\makeatletter
\@ifpackagelater{sphinxpackagefootnote}{2022/02/12}
    {}% Sphinx >= 5.0.0, nothing to do
    {%
\usepackage{expdlist}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}
% but expdlist old LaTeX package requires fixes:
% 1) remove extra space
\usepackage{etoolbox}
\patchcmd\@item{{\@breaklabel} }{{\@breaklabel}}{}{}
% 2) fix bug in expdlist's way of breaking the line after long item label
\def\breaklabel{%
    \def\@breaklabel{%
        \leavevmode\par
        % now a hack because Sphinx inserts \leavevmode after term node
        \def\leavevmode{\def\leavevmode{\unhbox\voidb@x}}%
    }%
}
    }% Sphinx < 5.0.0 (and assumed >= 4.0.0)
\makeatother

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
  ("index", 'numpy', 'NumPy Documentation', _stdauthor, 'NumPy',
   "NumPy: array processing for numbers, strings, records, and objects.",
   'Programming',
   1),
]


# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    'neps': ('https://numpy.org/neps', None),
    'python': ('https://docs.python.org/3', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'imageio': ('https://imageio.readthedocs.io/en/stable', None),
    'skimage': ('https://scikit-image.org/docs/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'scipy-lecture-notes': ('https://scipy-lectures.org', None),
    'pytest': ('https://docs.pytest.org/en/stable', None),
    'numpy-tutorials': ('https://numpy.org/numpy-tutorials', None),
    'numpydoc': ('https://numpydoc.readthedocs.io/en/latest', None),
    'dlpack': ('https://dmlc.github.io/dlpack/latest', None)
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

autosummary_generate = True

# -----------------------------------------------------------------------------
# Coverage checker
# -----------------------------------------------------------------------------
coverage_ignore_modules = []
coverage_ignore_functions = [
    'test($|_)', '(some|all)true', 'bitwise_not', 'cumproduct', 'pkgload', 'generic\\.'
]
coverage_ignore_classes = []

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
phi = (math.sqrt(5) + 1) / 2

plot_rcparams = {
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3 * phi, 3),
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


def _get_c_source_file(obj):
    if issubclass(obj, numpy.generic):
        return r"_core/src/multiarray/scalartypes.c.src"
    elif obj is numpy.ndarray:
        return r"_core/src/multiarray/arrayobject.c"
    else:
        # todo: come up with a better way to generate these
        return None

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

    fn = None
    lineno = None

    # Make a poor effort at linking C extension types
    if isinstance(obj, type) and obj.__module__ == 'numpy':
        fn = _get_c_source_file(obj)

    # This can be removed when removing the decorator set_module. Fix issue #28629
    if hasattr(obj, '_module_file'):
        fn = obj._module_file
        fn = relpath(fn, start=dirname(numpy.__file__))

    if fn is None:
        try:
            fn = inspect.getsourcefile(obj)
        except Exception:
            fn = None
        if not fn:
            return None

        # Ignore re-exports as their source files are not within the numpy repo
        module = inspect.getmodule(obj)
        if module is not None and not module.__name__.startswith("numpy"):
            return None

        try:
            source, lineno = inspect.getsourcelines(obj)
        except Exception:
            lineno = None

        fn = relpath(fn, start=dirname(numpy.__file__))

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    if 'dev' in numpy.__version__:
        return f"https://github.com/numpy/numpy/blob/main/numpy/{fn}{linespec}"
    else:
        return "https://github.com/numpy/numpy/blob/v%s/numpy/%s%s" % (
           numpy.__version__, fn, linespec)


from pygments.lexers import CLexer
from pygments.lexer import inherit
from pygments.token import Comment

class NumPyLexer(CLexer):
    name = 'NUMPYLEXER'

    tokens = {
        'statements': [
            (r'@[a-zA-Z_]*@', Comment.Preproc, 'macro'),
            inherit,
        ],
    }


# -----------------------------------------------------------------------------
# Breathe & Doxygen
# -----------------------------------------------------------------------------
breathe_projects = {'numpy': os.path.join("..", "build", "doxygen", "xml")}
breathe_default_project = "numpy"
breathe_default_members = ("members", "undoc-members", "protected-members")

# See https://github.com/breathe-doc/breathe/issues/696
nitpick_ignore = [
    ('c:identifier', 'FILE'),
    ('c:identifier', 'size_t'),
    ('c:identifier', 'PyHeapTypeObject'),
]

# -----------------------------------------------------------------------------
# Interactive documentation examples via JupyterLite
# -----------------------------------------------------------------------------

global_enable_try_examples = True
try_examples_global_button_text = "Try it in your browser!"
try_examples_global_warning_text = (
    "NumPy's interactive examples are experimental and may not always work"
    " as expected, with high load times especially on low-resource platforms,"
    " and the version of NumPy might not be in sync with the one you are"
    " browsing the documentation for. If you encounter any issues, please"
    " report them on the"
    " [NumPy issue tracker](https://github.com/numpy/numpy/issues)."
)
