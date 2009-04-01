"""
A special directive for generating a matplotlib plot.

.. warning::

   This is a hacked version of plot_directive.py from Matplotlib.
   It's very much subject to change!

Usage
-----

Can be used like this::

    .. plot:: examples/example.py

    .. plot::

       import matplotlib.pyplot as plt
       plt.plot([1,2,3], [4,5,6])

    .. plot::

       A plotting example:

       >>> import matplotlib.pyplot as plt
       >>> plt.plot([1,2,3], [4,5,6])

The content is interpreted as doctest formatted if it has a line starting
with ``>>>``.

The ``plot`` directive supports the options

    format : {'python', 'doctest'}
        Specify the format of the input
    include-source : bool
        Whether to display the source code. Default can be changed in conf.py
    
and the ``image`` directive options ``alt``, ``height``, ``width``,
``scale``, ``align``, ``class``.

Configuration options
---------------------

The plot directive has the following configuration options:

    plot_output_dir
        Directory (relative to config file) where to store plot output.
        Should be inside the static directory. (Default: 'static')

    plot_pre_code
        Code that should be executed before each plot.

    plot_rcparams
        Dictionary of Matplotlib rc-parameter overrides.
        Has 'sane' defaults.

    plot_include_source
        Default value for the include-source option

    plot_formats
        The set of files to generate. Default: ['png', 'pdf', 'hires.png'],
        ie. everything.

TODO
----

* Don't put temp files to _static directory, but do function in the way
  the pngmath directive works, and plot figures only during output writing.

* Refactor Latex output; now it's plain images, but it would be nice
  to make them appear side-by-side, or in floats.

"""

import sys, os, glob, shutil, imp, warnings, cStringIO, re, textwrap

import warnings
warnings.warn("A plot_directive module is also available under "
              "matplotlib.sphinxext; expect this numpydoc.plot_directive "
              "module to be deprecated after relevant features have been "
              "integrated there.",
              FutureWarning, stacklevel=2)

def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir

    static_path = '_static'
    if hasattr(app.config, 'html_static_path') and app.config.html_static_path:
        static_path = app.config.html_static_path[0]

    app.add_config_value('plot_output_dir', static_path, True)
    app.add_config_value('plot_pre_code', '', True)
    app.add_config_value('plot_rcparams', sane_rcparameters, True)
    app.add_config_value('plot_include_source', False, True)
    app.add_config_value('plot_formats', ['png', 'hires.png', 'pdf'], True)

    app.add_directive('plot', plot_directive, True, (0, 1, False),
                      **plot_directive_options)

sane_rcparameters = {
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (4, 3),
}

#------------------------------------------------------------------------------
# Run code and capture figures
#------------------------------------------------------------------------------

import matplotlib
import matplotlib.cbook as cbook
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib import _pylab_helpers

def contains_doctest(text):
    r = re.compile(r'^\s*>>>', re.M)
    m = r.match(text)
    return bool(m)

def unescape_doctest(text):
    """
    Extract code from a piece of text, which contains either Python code
    or doctests.

    """
    if not contains_doctest(text):
        return text

    code = ""
    for line in text.split("\n"):
        m = re.match(r'^\s*(>>>|...) (.*)$', line)
        if m:
            code += m.group(2) + "\n"
        elif line.strip():
            code += "# " + line.strip() + "\n"
        else:
            code += "\n"
    return code

def run_code(code, code_path):
    # Change the working directory to the directory of the example, so
    # it can get at its data files, if any.
    pwd = os.getcwd()
    old_sys_path = list(sys.path)
    if code_path is not None:
        dirname = os.path.abspath(os.path.dirname(code_path))
        os.chdir(dirname)
        sys.path.insert(0, dirname)

    # Redirect stdout
    stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    
    try:
        code = unescape_doctest(code)
        ns = {}
        exec setup.config.plot_pre_code in ns
        exec code in ns
    finally:
        os.chdir(pwd)
        sys.path[:] = old_sys_path
        sys.stdout = stdout
    return ns


#------------------------------------------------------------------------------
# Generating figures
#------------------------------------------------------------------------------

def out_of_date(original, derived):
    """
    Returns True if derivative is out-of-date wrt original,
    both of which are full file paths.
    """
    return (not os.path.exists(derived)
            or os.stat(derived).st_mtime < os.stat(original).st_mtime)


def makefig(code, code_path, output_dir, output_base, config):
    """
    run a pyplot script and save the low and high res PNGs and a PDF in _static

    """

    included_formats = config.plot_formats
    if type(included_formats) is str:
        included_formats = eval(included_formats)
    
    formats = [x for x in [('png', 80), ('hires.png', 200), ('pdf', 50)]
               if x[0] in config.plot_formats]

    all_exists = True

    # Look for single-figure output files first
    for format, dpi in formats:
        output_path = os.path.join(output_dir, '%s.%s' % (output_base, format))
        if out_of_date(code_path, output_path):
            all_exists = False
            break

    if all_exists:
        return [output_base]

    # Then look for multi-figure output files
    image_names = []
    for i in xrange(1000):
        image_names.append('%s_%02d' % (output_base, i))
        for format, dpi in formats:
            output_path = os.path.join(output_dir,
                                       '%s.%s' % (image_names[-1], format))
            if out_of_date(code_path, output_path):
                all_exists = False
                break
        if not all_exists:
            # assume that if we have one, we have them all
            all_exists = (i > 0)
            break

    if all_exists:
        return image_names

    # We didn't find the files, so build them
    print "-- Plotting figures %s" % output_base

    # Clear between runs
    plt.close('all')

    # Reset figure parameters
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(config.plot_rcparams)

    # Run code
    run_code(code, code_path)

    # Collect images
    image_names = []

    fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
    for i, figman in enumerate(fig_managers):
        if len(fig_managers) == 1:
            name = output_base
        else:
            name = "%s_%02d" % (output_base, i)
        image_names.append(name)
        for format, dpi in formats:
            path = os.path.join(output_dir, '%s.%s' % (name, format))
            figman.canvas.figure.savefig(path, dpi=dpi)

    return image_names

#------------------------------------------------------------------------------
# Generating output
#------------------------------------------------------------------------------

from docutils import nodes, utils
import jinja

TEMPLATE = """
{{source_code}}

.. htmlonly::

   {% if source_code %}
       (`Source code <{{source_link}}>`__)
   {% endif %}

   .. admonition:: Output
      :class: plot-output

      {% for name in image_names %}
      .. figure:: {{link_dir}}/{{name}}.png
         {%- for option in options %}
         {{option}}
         {% endfor %}

         (
         {%- if not source_code %}`Source code <{{source_link}}>`__, {% endif -%}
         `PNG <{{link_dir}}/{{name}}.hires.png>`__,
         `PDF <{{link_dir}}/{{name}}.pdf>`__)
      {% endfor %}

.. latexonly::

   {% for name in image_names %}
   .. image:: {{link_dir}}/{{name}}.pdf
   {% endfor %}

"""

def run(arguments, content, options, state_machine, state, lineno):
    if arguments and content:
        raise RuntimeError("plot:: directive can't have both args and content")

    document = state_machine.document
    config = document.settings.env.config

    options.setdefault('include-source', config.plot_include_source)
    if options['include-source'] is None:
        options['include-source'] = config.plot_include_source

    # determine input
    rst_file = document.attributes['source']
    rst_dir = os.path.dirname(rst_file)
    
    if arguments:
        file_name = os.path.join(rst_dir, directives.uri(arguments[0]))
        code = open(file_name, 'r').read()
        output_base = os.path.basename(file_name)
    else:
        file_name = rst_file
        code = textwrap.dedent("\n".join(map(str, content)))
        counter = document.attributes.get('_plot_counter', 0) + 1
        document.attributes['_plot_counter'] = counter
        output_base = '%d-%s' % (counter, os.path.basename(file_name))

    rel_name = relpath(file_name, setup.confdir)

    base, ext = os.path.splitext(output_base)
    if ext in ('.py', '.rst', '.txt'):
        output_base = base

    # is it in doctest format?
    is_doctest = contains_doctest(code)
    if options.has_key('format'):
        if options['format'] == 'python':
            is_doctest = False
        else:
            is_doctest = True

    # determine output
    file_rel_dir = os.path.dirname(rel_name)
    while file_rel_dir.startswith(os.path.sep):
        file_rel_dir = file_rel_dir[1:]

    output_dir = os.path.join(setup.confdir, setup.config.plot_output_dir,
                              file_rel_dir)

    if not os.path.exists(output_dir):
        cbook.mkdirs(output_dir)

    # copy script
    target_name = os.path.join(output_dir, output_base)
    f = open(target_name, 'w')
    f.write(unescape_doctest(code))
    f.close()

    source_link = relpath(target_name, rst_dir)

    # determine relative reference
    link_dir = relpath(output_dir, rst_dir)

    # make figures
    try:
        image_names = makefig(code, file_name, output_dir, output_base, config)
    except RuntimeError, err:
        reporter = state.memo.reporter
        sm = reporter.system_message(3, "Exception occurred rendering plot",
                                     line=lineno)
        return [sm]

    # generate output
    if options['include-source']:
        if is_doctest:
            lines = ['']
        else:
            lines = ['.. code-block:: python', '']
        lines += ['    %s' % row.rstrip() for row in code.split('\n')]
        source_code = "\n".join(lines)
    else:
        source_code = ""

    opts = [':%s: %s' % (key, val) for key, val in options.items()
            if key in ('alt', 'height', 'width', 'scale', 'align', 'class')]

    result = jinja.from_string(TEMPLATE).render(
        link_dir=link_dir.replace(os.path.sep, '/'),
        source_link=source_link,
        options=opts,
        image_names=image_names,
        source_code=source_code)

    lines = result.split("\n")
    if len(lines):
        state_machine.insert_input(
            lines, state_machine.input_lines.source(0))

    return []


if hasattr(os.path, 'relpath'):
    relpath = os.path.relpath
else:
    def relpath(target, base=os.curdir):
        """
        Return a relative path to the target from either the current
        dir or an optional base dir.  Base can be a directory
        specified either as absolute or relative to current dir.
        """

        if not os.path.exists(target):
            raise OSError, 'Target does not exist: '+target

        if not os.path.isdir(base):
            raise OSError, 'Base is not a directory or does not exist: '+base

        base_list = (os.path.abspath(base)).split(os.sep)
        target_list = (os.path.abspath(target)).split(os.sep)

        # On the windows platform the target may be on a completely
        # different drive from the base.
        if os.name in ['nt','dos','os2'] and base_list[0] <> target_list[0]:
            raise OSError, 'Target is on a different drive to base. Target: '+target_list[0].upper()+', base: '+base_list[0].upper()

        # Starting from the filepath root, work out how much of the
        # filepath is shared by base and target.
        for i in range(min(len(base_list), len(target_list))):
            if base_list[i] <> target_list[i]: break
        else:
            # If we broke out of the loop, i is pointing to the first
            # differing path elements.  If we didn't break out of the
            # loop, i is pointing to identical path elements.
            # Increment i so that in all cases it points to the first
            # differing path elements.
            i+=1

        rel_list = [os.pardir] * (len(base_list)-i) + target_list[i:]
        return os.path.join(*rel_list)

#------------------------------------------------------------------------------
# plot:: directive registration etc.
#------------------------------------------------------------------------------

from docutils.parsers.rst import directives
try:
    # docutils 0.4
    from docutils.parsers.rst.directives.images import align
except ImportError:
    # docutils 0.5
    from docutils.parsers.rst.directives.images import Image
    align = Image.align

def plot_directive(name, arguments, options, content, lineno,
                   content_offset, block_text, state, state_machine):
    return run(arguments, content, options, state_machine, state, lineno)

plot_directive.__doc__ = __doc__

def _option_boolean(arg):
    if not arg or not arg.strip():
        return None
    elif arg.strip().lower() in ('no', '0', 'false'):
        return False
    elif arg.strip().lower() in ('yes', '1', 'true'):
        return True
    else:
        raise ValueError('"%s" unknown boolean' % arg)

def _option_format(arg):
    return directives.choice(arg, ('python', 'lisp'))

plot_directive_options = {'alt': directives.unchanged,
                          'height': directives.length_or_unitless,
                          'width': directives.length_or_percentage_or_unitless,
                          'scale': directives.nonnegative_int,
                          'align': align,
                          'class': directives.class_option,
                          'include-source': _option_boolean,
                          'format': _option_format,
                          }
