# plot_directive.py from matplotlib.sf.net
"""A special directive for including a matplotlib plot.

Given a path to a .py file, it includes the source code inline, then:

- On HTML, will include a .png with a link to a high-res .png.

- On LaTeX, will include a .pdf

This directive supports all of the options of the `image` directive,
except for `target` (since plot will add its own target).

Additionally, if the :include-source: option is provided, the literal
source will be included inline, as well as a link to the source.

.. warning::

   This is a hacked version of plot_directive.py from Matplotlib.
   It's very much subject to change!

"""

import sys, os, glob, shutil, imp, warnings, cStringIO, re
from docutils.parsers.rst import directives
try:
    # docutils 0.4
    from docutils.parsers.rst.directives.images import align
except ImportError:
    # docutils 0.5
    from docutils.parsers.rst.directives.images import Image
    align = Image.align

import matplotlib
import matplotlib.cbook as cbook
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib import _pylab_helpers

def runfile(fullpath, is_doctest=False):
    # Change the working directory to the directory of the example, so
    # it can get at its data files, if any.
    pwd = os.getcwd()
    path, fname = os.path.split(fullpath)
    os.chdir(path)
    stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    try:
        code = ""
        if is_doctest:
            fd = cStringIO.StringIO()
            for line in open(fname):
                m = re.match(r'^\s*(>>>|...) (.*)$', line)
                if m:
                    code += m.group(2) + "\n"
        else:
            code = open(fname).read()

        ns = {}
        exec setup.config.plot_pre_code in ns
        exec code in ns
    finally:
        os.chdir(pwd)
        sys.stdout = stdout
    return ns

options = {'alt': directives.unchanged,
           'height': directives.length_or_unitless,
           'width': directives.length_or_percentage_or_unitless,
           'scale': directives.nonnegative_int,
           'align': align,
           'class': directives.class_option,
           'include-source': directives.flag,
           'doctest-format': directives.flag
           }

template = """
.. htmlonly::

   [`source code <%(linkdir)s/%(sourcename)s>`__,
   `png <%(linkdir)s/%(outname)s.hires.png>`__,
   `pdf <%(linkdir)s/%(outname)s.pdf>`__]

   .. image:: %(linkdir)s/%(outname)s.png
%(options)s

.. latexonly::
   .. image:: %(linkdir)s/%(outname)s.pdf
%(options)s

"""

exception_template = """
.. htmlonly::

   [`source code <%(linkdir)s/%(sourcename)s>`__]

Exception occurred rendering plot.

"""


def out_of_date(original, derived):
    """
    Returns True if derivative is out-of-date wrt original,
    both of which are full file paths.
    """
    return (not os.path.exists(derived)
            or os.stat(derived).st_mtime < os.stat(original).st_mtime)

def makefig(fullpath, outdir, is_doctest=False):
    """
    run a pyplot script and save the low and high res PNGs and a PDF in _static

    """

    fullpath = str(fullpath)  # todo, why is unicode breaking this

    print '    makefig: fullpath=%s, outdir=%s'%( fullpath, outdir)
    formats = [('png', 80),
               ('hires.png', 200),
               ('pdf', 50),
               ]

    basedir, fname = os.path.split(fullpath)
    basename, ext = os.path.splitext(fname)
    if ext != '.py':
        basename = fname
    sourcename = fname
    all_exists = True

    if basedir != outdir:
        shutil.copyfile(fullpath, os.path.join(outdir, fname))

    # Look for single-figure output files first
    for format, dpi in formats:
        outname = os.path.join(outdir, '%s.%s' % (basename, format))
        if out_of_date(fullpath, outname):
            all_exists = False
            break

    if all_exists:
        print '    already have %s'%fullpath
        return 1

    # Then look for multi-figure output files, assuming
    # if we have some we have all...
    i = 0
    while True:
        all_exists = True
        for format, dpi in formats:
            outname = os.path.join(outdir, '%s_%02d.%s' % (basename, i, format))
            if out_of_date(fullpath, outname):
                all_exists = False
                break
        if all_exists:
            i += 1
        else:
            break

    if i != 0:
        print '    already have %d figures for %s' % (i, fullpath)
        return i

    # We didn't find the files, so build them

    print '    building %s'%fullpath
    plt.close('all')    # we need to clear between runs
    matplotlib.rcdefaults()
    # Set a figure size that doesn't overflow typical browser windows
    matplotlib.rcParams['figure.figsize'] = (5.5, 4.5)

    try:
        runfile(fullpath, is_doctest=is_doctest)
    except:
	s = cbook.exception_to_str("Exception running plot %s" % fullpath)
        warnings.warn(s)
        return 0

    fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
    for i, figman in enumerate(fig_managers):
        for format, dpi in formats:
            if len(fig_managers) == 1:
                outname = basename
            else:
                outname = "%s_%02d" % (basename, i)
            outpath = os.path.join(outdir, '%s.%s' % (outname, format))
            try:
                figman.canvas.figure.savefig(outpath, dpi=dpi)
            except:
                s = cbook.exception_to_str("Exception running plot %s" % fullpath)
                warnings.warn(s)
                return 0

    return len(fig_managers)

def run(arguments, options, state_machine, lineno):
    reference = directives.uri(arguments[0])
    basedir, fname = os.path.split(reference)
    basename, ext = os.path.splitext(fname)
    if ext != '.py':
        basename = fname
    sourcename = fname
    #print 'plotdir', reference, basename, ext

    # get the directory of the rst file
    rstdir, rstfile = os.path.split(state_machine.document.attributes['source'])
    reldir = rstdir[len(setup.confdir)+1:]
    relparts = [p for p in os.path.split(reldir) if p.strip()]
    nparts = len(relparts)
    #print '    rstdir=%s, reldir=%s, relparts=%s, nparts=%d'%(rstdir, reldir, relparts, nparts)
    #print 'RUN', rstdir, reldir
    outdir = os.path.join(setup.confdir, setup.config.plot_output_dir, basedir)
    if not os.path.exists(outdir):
        cbook.mkdirs(outdir)

    linkdir = ('../' * nparts) + setup.config.plot_output_dir.replace(os.path.sep, '/') + '/' + basedir
    #linkdir = os.path.join('..', outdir)
    num_figs = makefig(reference, outdir,
                       is_doctest=('doctest-format' in options))
    #print '    reference="%s", basedir="%s", linkdir="%s", outdir="%s"'%(reference, basedir, linkdir, outdir)

    if options.has_key('include-source'):
        contents = open(reference, 'r').read()
        if 'doctest-format' in options:
            lines = ['']
        else:
            lines = ['.. code-block:: python', '']
        lines += ['    %s'%row.rstrip() for row in contents.split('\n')]
        del options['include-source']
    else:
        lines = []

    if 'doctest-format' in options:
        del options['doctest-format']
    
    if num_figs > 0:
        options = ['      :%s: %s' % (key, val) for key, val in
                   options.items()]
        options = "\n".join(options)

        for i in range(num_figs):
            if num_figs == 1:
                outname = basename
            else:
                outname = "%s_%02d" % (basename, i)
            lines.extend((template % locals()).split('\n'))
    else:
        lines.extend((exception_template % locals()).split('\n'))

    if len(lines):
        state_machine.insert_input(
            lines, state_machine.input_lines.source(0))
    return []



try:
    from docutils.parsers.rst import Directive
except ImportError:
    from docutils.parsers.rst.directives import _directives

    def plot_directive(name, arguments, options, content, lineno,
                       content_offset, block_text, state, state_machine):
        return run(arguments, options, state_machine, lineno)
    plot_directive.__doc__ = __doc__
    plot_directive.arguments = (1, 0, 1)
    plot_directive.options = options

    _directives['plot'] = plot_directive
else:
    class plot_directive(Directive):
        required_arguments = 1
        optional_arguments = 0
        final_argument_whitespace = True
        option_spec = options
        def run(self):
            return run(self.arguments, self.options,
                       self.state_machine, self.lineno)
    plot_directive.__doc__ = __doc__

    directives.register_directive('plot', plot_directive)

def setup(app):
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir

    app.add_config_value('plot_output_dir', '_static', True)
    app.add_config_value('plot_pre_code', '', True)

plot_directive.__doc__ = __doc__

directives.register_directive('plot', plot_directive)

