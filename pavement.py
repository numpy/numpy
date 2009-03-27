import os
import sys
import subprocess
try:
    from hash import md5
except ImportError:
    import md5

import sphinx

import distutils
import numpy.distutils

try:
    from paver.tasks import VERSION as _PVER
    if not _PVER >= '1.0':
        raise RuntimeError("paver version >= 1.0 required (was %s)" % _PVER)
except ImportError, e:
    raise RuntimeError("paver version >= 1.0 required")

import paver
import paver.doctools
import paver.path
from paver.easy import options, Bunch, task, needs, dry, sh, call_task
from paver.setuputils import setup

PDF_DESTDIR = paver.path.path('build') / 'pdf'
HTML_DESTDIR = paver.path.path('build') / 'html'

RELEASE = 'doc/release/1.3.0-notes.rst'
LOG_START = 'tags/1.2.0'
LOG_END = 'master'
BOOTSTRAP_DIR = "bootstrap"
BOOTSTRAP_PYEXEC = "%s/bin/python" % BOOTSTRAP_DIR
BOOTSTRAP_SCRIPT = "%s/bootstrap.py" % BOOTSTRAP_DIR

options(sphinx=Bunch(builddir="build", sourcedir="source", docroot='doc'),
        virtualenv=Bunch(script_name=BOOTSTRAP_SCRIPT))

# Bootstrap stuff
@task
def bootstrap():
    """create virtualenv in ./install"""
    install = paver.path.path(BOOTSTRAP_DIR)
    if not install.exists():
        install.mkdir()
    call_task('paver.virtual.bootstrap')
    sh('cd %s; %s bootstrap.py' % (BOOTSTRAP_DIR, sys.executable))

@task
def clean():
    """Remove build, dist, egg-info garbage."""
    d = ['build', 'dist']
    for i in d:
        paver.path.path(i).rmtree()

    (paver.path.path('doc') / options.sphinx.builddir).rmtree()

@task
def clean_bootstrap():
    paver.path.path('bootstrap').rmtree()

# NOTES/Changelog stuff
def compute_md5():
    released = paver.path.path('installers').listdir()
    checksums = []
    for f in released:
        m = md5.md5(open(f, 'r').read())
        checksums.append('%s  %s' % (m.hexdigest(), f))

    return checksums

def write_release_task(filename='NOTES.txt'):
    source = paver.path.path(RELEASE)
    target = paver.path.path(filename)
    if target.exists():
        target.remove()
    source.copy(target)
    ftarget = open(str(target), 'a')
    ftarget.writelines("""
Checksums
=========

""")
    ftarget.writelines(['%s\n' % c for c in compute_md5()])

def write_log_task(filename='Changelog'):
    st = subprocess.Popen(
            ['git', 'svn', 'log',  '%s..%s' % (LOG_START, LOG_END)],
            stdout=subprocess.PIPE)

    out = st.communicate()[0]
    a = open(filename, 'w')
    a.writelines(out)
    a.close()

@task
def write_release():
    write_release_task()

@task
def write_log():
    write_log_task()

# Doc stuff
@task
@needs('paver.doctools.html')
def html(options):
    """Build numpy documentation and put it into build/docs"""
    builtdocs = paver.path.path("doc") / options.sphinx.builddir / "html"
    HTML_DESTDIR.rmtree()
    builtdocs.copytree(HTML_DESTDIR)
