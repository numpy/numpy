"""
Building a fancy dmg from scratch
=================================

Clone the numpy-macosx-installer git repo from on github into the source tree
(numpy-macosx-installer should be in the same directory as setup.py). Then, do
as follows::

    paver clean clean_bootstrap
    paver bootstrap && source boostrap/bin/activate
    python setupegg.py install
    paver dmg

Building a simple (no-superpack) windows installer from wine
============================================================

It assumes that blas/lapack are in c:\local\lib inside drive_c. Build python
2.5 and python 2.6 installers.

    paver clean
    paver bdist_wininst

Building changelog + notes
==========================

Assumes you have git and the binaries/tarballs in installers/::

    paver write_release
    paver write_note

This automatically put the checksum into NOTES.txt, and write the Changelog
which can be uploaded to sourceforge.
"""
import os
import sys
import subprocess
import re
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

setup_py = __import__("setup")
FULLVERSION = setup_py.FULLVERSION

# Wine config for win32 builds
WINE_SITE_CFG = ""
if sys.platform == "darwin":
	WINE_PY25 = "/Applications/Darwine/Wine.bundle/Contents/bin/wine /Users/david/.wine/drive_c/Python25/python.exe"
	WINE_PY26 = "/Applications/Darwine/Wine.bundle/Contents/bin/wine /Users/david/.wine/drive_c/Python26/python.exe"
else:
	WINE_PY25 = "/home/david/.wine/drive_c/Python25/python.exe"
	WINE_PY26 = "/home/david/.wine/drive_c/Python26/python.exe"
WINE_PYS = {'2.6' : WINE_PY26, '2.5': WINE_PY25}
SUPERPACK_BUILD = 'build-superpack'
SUPERPACK_BINDIR = os.path.join(SUPERPACK_BUILD, 'binaries')

PDF_DESTDIR = paver.path.path('build') / 'pdf'
HTML_DESTDIR = paver.path.path('build') / 'html'

RELEASE = 'doc/release/1.3.0-notes.rst'
LOG_START = 'tags/1.2.0'
LOG_END = 'master'
BOOTSTRAP_DIR = "bootstrap"
BOOTSTRAP_PYEXEC = "%s/bin/python" % BOOTSTRAP_DIR
BOOTSTRAP_SCRIPT = "%s/bootstrap.py" % BOOTSTRAP_DIR

DMG_CONTENT = paver.path.path('numpy-macosx-installer') / 'content'

INSTALLERS_DIR = 'installers'

options(sphinx=Bunch(builddir="build", sourcedir="source", docroot='doc'),
        virtualenv=Bunch(script_name=BOOTSTRAP_SCRIPT),
        wininst=Bunch(pyver="2.5", scratch=True))

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
    d = ['build', 'dist', 'numpy.egg-info']
    for i in d:
        paver.path.path(i).rmtree()

    (paver.path.path('doc') / options.sphinx.builddir).rmtree()

@task
def clean_bootstrap():
    paver.path.path('bootstrap').rmtree()

@task
@needs('clean', 'clean_bootstrap')
def nuke():
    """Remove everything: build dir, installers, bootstrap dirs, etc..."""
    d = [SUPERPACK_BUILD, INSTALLERS_DIR]
    for i in d:
        paver.path.path(i).rmtree()

# NOTES/Changelog stuff
def compute_md5():
    released = paver.path.path(INSTALLERS_DIR).listdir()
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

def _latex_paths():
    """look up the options that determine where all of the files are."""
    opts = options
    docroot = paver.path.path(opts.get('docroot', 'docs'))
    if not docroot.exists():
        raise BuildFailure("Sphinx documentation root (%s) does not exist."
                % docroot)
    builddir = docroot / opts.get("builddir", ".build")
    builddir.mkdir()
    srcdir = docroot / opts.get("sourcedir", "")
    if not srcdir.exists():
        raise BuildFailure("Sphinx source file dir (%s) does not exist"
                % srcdir)
    latexdir = builddir / "latex"
    latexdir.mkdir()
    return Bunch(locals())

@task
def latex():
    """Build samplerate's documentation and install it into
    scikits/samplerate/docs"""
    paths = _latex_paths()
    sphinxopts = ['', '-b', 'latex', paths.srcdir, paths.latexdir]
    #dry("sphinx-build %s" % (" ".join(sphinxopts),), sphinx.main, sphinxopts)
    subprocess.check_call(["make", "latex"], cwd="doc")

@task
@needs('latex')
def pdf():
    paths = _latex_paths()
    def build_latex():
        subprocess.check_call(["make", "all-pdf"], cwd=paths.latexdir)
    dry("Build pdf doc", build_latex)

    PDF_DESTDIR.rmtree()
    PDF_DESTDIR.makedirs()

    user = paths.latexdir / "numpy-user.pdf"
    user.copy(PDF_DESTDIR / "userguide.pdf")
    ref = paths.latexdir / "numpy-ref.pdf"
    ref.copy(PDF_DESTDIR / "reference.pdf")

@task
def sdist():
    # To be sure to bypass paver when building sdist... paver + numpy.distutils
    # do not play well together.
    sh('python setup.py sdist --formats=gztar,zip')

#------------------
# Wine-based builds
#------------------
_SSE3_CFG = r"""[atlas]
library_dirs = C:\local\lib\yop\sse3"""
_SSE2_CFG = r"""[atlas]
library_dirs = C:\local\lib\yop\sse2"""
_NOSSE_CFG = r"""[DEFAULT]
library_dirs = C:\local\lib\yop\nosse"""

SITECFG = {"sse2" : _SSE2_CFG, "sse3" : _SSE3_CFG, "nosse" : _NOSSE_CFG}

def internal_wininst_name(arch, ismsi=False):
    """Return the name of the wininst as it will be inside the superpack (i.e.
    with the arch encoded."""
    if ismsi:
        ext = '.msi'
    else:
        ext = '.exe'
    return "numpy-%s-%s%s" % (FULLVERSION, arch, ext)

def wininst_name(pyver, ismsi=False):
    """Return the name of the installer built by wininst command."""
    # Yeah, the name logic is harcoded in distutils. We have to reproduce it
    # here
    if ismsi:
        ext = '.msi'
    else:
        ext = '.exe'
    name = "numpy-%s.win32-py%s%s" % (FULLVERSION, pyver, ext)
    return name

def bdist_wininst_arch(pyver, arch, scratch=True):
    """Arch specific wininst build."""
    if scratch:
        paver.path.path('build').rmtree()

    if not os.path.exists(SUPERPACK_BINDIR):
        os.makedirs(SUPERPACK_BINDIR)
    _bdist_wininst(pyver, SITECFG[arch])
    source = os.path.join('dist', wininst_name(pyver))
    target = os.path.join(SUPERPACK_BINDIR, internal_wininst_name(arch))
    if os.path.exists(target):
        os.remove(target)
    os.rename(source, target)

def prepare_nsis_script(pyver, numver):
    if not os.path.exists(SUPERPACK_BUILD):
        os.makedirs(SUPERPACK_BUILD)

    tpl = os.path.join('tools/win32build/nsis_scripts', 'numpy-superinstaller.nsi.in')
    source = open(tpl, 'r')
    target = open(os.path.join(SUPERPACK_BUILD, 'numpy-superinstaller.nsi'), 'w')

    installer_name = 'numpy-%s-win32-superpack-python%s.exe' % (numver, pyver)
    cnt = "".join(source.readlines())
    cnt = cnt.replace('@NUMPY_INSTALLER_NAME@', installer_name)
    for arch in ['nosse', 'sse2', 'sse3']:
        cnt = cnt.replace('@%s_BINARY@' % arch.upper(),
                          internal_wininst_name(arch))

    target.write(cnt)

@task
def bdist_wininst_nosse(options):
    """Build the nosse wininst installer."""
    bdist_wininst_arch(options.wininst.pyver, 'nosse', scratch=options.wininst.scratch)

@task
def bdist_wininst_sse2(options):
    """Build the sse2 wininst installer."""
    bdist_wininst_arch(options.wininst.pyver, 'sse2', scratch=options.wininst.scratch)

@task
def bdist_wininst_sse3(options):
    """Build the sse3 wininst installer."""
    bdist_wininst_arch(options.wininst.pyver, 'sse3', scratch=options.wininst.scratch)

@task
#@needs('bdist_wininst_nosse', 'bdist_wininst_sse2', 'bdist_wininst_sse3')
def bdist_superpack(options):
    """Build all arch specific wininst installers."""
    prepare_nsis_script(options.wininst.pyver, FULLVERSION)
    subprocess.check_call(['makensis', 'numpy-superinstaller.nsi'],
            cwd=SUPERPACK_BUILD)

@task
@needs('clean', 'bdist_wininst')
def bdist_wininst_simple():
    """Simple wininst-based installer."""
    _bdist_wininst(pyver=options.wininst.pyver)

def _bdist_wininst(pyver, cfgstr=WINE_SITE_CFG):
    site = paver.path.path('site.cfg')
    exists = site.exists()
    try:
        if exists:
            site.move('site.cfg.bak')
        a = open(str(site), 'w')
        a.writelines(cfgstr)
        a.close()
        sh('%s setup.py build -c mingw32 bdist_wininst' % WINE_PYS[pyver])
    finally:
        site.remove()
        if exists:
            paver.path.path('site.cfg.bak').move(site)

#-------------------
# Mac OS X installer
#-------------------
def macosx_version():
    if not sys.platform == 'darwin':
        raise ValueError("Not darwin ??")
    st = subprocess.Popen(["sw_vers"], stdout=subprocess.PIPE)
    out = st.stdout.readlines()
    ver = re.compile("ProductVersion:\s+([0-9]+)\.([0-9]+)\.([0-9]+)")
    for i in out:
        m = ver.match(i)
        if m:
            return m.groups()

def mpkg_name():
    maj, min = macosx_version()[:2]
    pyver = ".".join([str(i) for i in sys.version_info[:2]])
    return "numpy-%s-py%s-macosx%s.%s.mpkg" % \
            (FULLVERSION, pyver, maj, min)

def dmg_name():
    maj, min = macosx_version()[:2]
    pyver = ".".join([str(i) for i in sys.version_info[:2]])
    return "numpy-%s-py%s-macosx%s.%s.dmg" % \
            (FULLVERSION, pyver, maj, min)

@task
def bdist_mpkg():
	sh("python setupegg.py bdist_mpkg")

@task
@needs("bdist_mpkg", "pdf")
def dmg():
    pyver = ".".join([str(i) for i in sys.version_info[:2]])

    dmg_n = dmg_name()
    dmg = paver.path.path('numpy-macosx-installer') / dmg_n
    if dmg.exists():
        dmg.remove()

	# Clean the image source
    content = DMG_CONTENT
    content.rmtree()
    content.mkdir()

    # Copy mpkg into image source
    mpkg_n = mpkg_name()
    mpkg_tn = "numpy-%s-py%s.mpkg" % (FULLVERSION, pyver)
    mpkg_source = paver.path.path("dist") / mpkg_n
    mpkg_target = content / mpkg_tn
    mpkg_source.copytree(content / mpkg_tn)

    # Copy docs into image source

    #html_docs = HTML_DESTDIR
    #html_docs.copytree(content / "Documentation" / "html")

    pdf_docs = DMG_CONTENT / "Documentation"
    pdf_docs.rmtree()
    pdf_docs.makedirs()

    user = PDF_DESTDIR / "userguide.pdf"
    user.copy(pdf_docs / "userguide.pdf")
    ref = PDF_DESTDIR / "reference.pdf"
    ref.copy(pdf_docs / "reference.pdf")

    # Build the dmg
    cmd = ["./create-dmg", "--window-size", "500", "500", "--background",
        "art/dmgbackground.png", "--icon-size", "128", "--icon", mpkg_tn, 
        "125", "320", "--icon", "Documentation", "375", "320", "--volname", "numpy",
        dmg_n, "./content"]
    subprocess.check_call(cmd, cwd="numpy-macosx-installer")
    
@task
def simple_dmg():
    # Build the dmg
    image_name = "numpy-%s.dmg" % FULLVERSION
    image = paver.path.path(image_name)
    image.remove()
    cmd = ["hdiutil", "create", image_name, "-srcdir", str(builddir)]
    sh(" ".join(cmd))
