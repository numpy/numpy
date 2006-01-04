# Makefile for f2py2e
#
# Use GNU make for making.
# $Revision: 1.46 $
# $Date: 2005/01/30 17:22:55 $
# Pearu Peterson <pearu@ioc.ee>

PYTHON=python
MAJOR=2
F2PY2E_CVSROOT=:pserver:anonymous@cens.ioc.ee:/home/cvs
SCIPY_CVSROOT=:pserver:anonymous@numpy.org:/home/cvsroot

UPLOADCMD = scp -r
UPLOADDIR = pearu@kev.ioc.ee:/net/cens/home/www/unsecure/projects/f2py2e/

REV=`python -c 'from __version__ import *;print version'`
SCIPY_DISTUTILS_REV=`cd numpy_distutils && $(PYTHON) -c 'from numpy_distutils_version import *;print numpy_distutils_version' && cd ..`

SRC_FILES = F2PY-$(MAJOR)-latest.tar.gz numpy_distutils-latest.tar.gz F2PY-$(MAJOR)-latest.win32.exe numpy_distutils-latest.win32.exe

HTML_FILES = index.html FAQ.html HISTORY.html THANKS.html TESTING.html OLDNEWS.html
FAQ_DEPS = simple.f pytest.py pyforttest.pyf simple_session.dat
README_DEPS = hello.f
UG_FILES = index.html f2py_usersguide.pdf
UG_FILES_DEP = $(shell cd docs/usersguide && ls *.{f,f90,dat,pyf,py})

WWW_SRC_FILES = $(SRC_FILES:%=upload/www/$(MAJOR).x/%)
WWW_WEB_FILES = $(HTML_FILES:%=upload/www/%) $(README_DEPS:%=upload/www/%)
WWW_UG_FILES = $(UG_FILES:%=upload/www/usersguide/%) $(UG_FILES_DEP:%=upload/www/usersguide/%)

TMP_WEB_FILES = $(HTML_FILES:%=upload/tmp/%) $(README_DEPS:%=upload/tmp/%)

##############################################################################

all:
	@echo "Use 'make install' to install f2py"
	@echo "Use 'make generate' to build f2py docs to upload/tmp"
install:
	$(PYTHON) setup.py install
test:
	cd tests && $(PYTHON) run_all.py

##############################################################################
# Create F2PY tar-balls
##############################################################################
f2py2e:
	test -d f2py2e && (cd f2py2e && cvs -d $(F2PY2E_CVSROOT) -z7 update -Pd && cd -) || cvs -d $(F2PY2E_CVSROOT) checkout f2py2e

upload/tmp/$(MAJOR).x/F2PY-$(MAJOR)-latest.tar.gz: f2py2e
	cd f2py2e && python setup.py sdist -f
	mkdir -p upload/tmp/$(MAJOR).x
	cp f2py2e/dist/F2PY-$(REV).tar.gz upload/tmp/$(MAJOR).x
	ln -sf F2PY-$(REV).tar.gz F2PY-$(MAJOR)-latest.tar.gz
	mv F2PY-$(MAJOR)-latest.tar.gz upload/tmp/$(MAJOR).x
upload/tmp/$(MAJOR).x/F2PY-$(MAJOR)-latest.win32.exe: f2py2e
	cd f2py2e && python setup.py bdist_wininst
	mkdir -p upload/tmp/$(MAJOR).x
	cp f2py2e/dist/F2PY-$(REV).win32.exe upload/tmp/$(MAJOR).x
	ln -sf F2PY-$(REV).win32.exe F2PY-$(MAJOR)-latest.win32.exe
	mv F2PY-$(MAJOR)-latest.win32.exe upload/tmp/$(MAJOR).x
f2py2e_latest: upload/tmp/$(MAJOR).x/F2PY-$(MAJOR)-latest.tar.gz upload/tmp/$(MAJOR).x/F2PY-$(MAJOR)-latest.win32.exe

##############################################################################
# Create Scipy_distutils tar-balls
##############################################################################

numpy_distutils:
	test -d numpy_distutils && (cd numpy_distutils && cvs -d $(SCIPY_CVSROOT) -z7 update -Pd && cd -) || cvs -d $(SCIPY_CVSROOT) checkout numpy_distutils

upload/tmp/$(MAJOR).x/numpy_distutils-latest.tar.gz: numpy_distutils
	cd numpy_distutils && python setup.py sdist -f
	mkdir -p upload/tmp/$(MAJOR).x
	cp numpy_distutils/dist/numpy_distutils-$(SCIPY_DISTUTILS_REV).tar.gz upload/tmp/$(MAJOR).x
	ln -sf numpy_distutils-$(SCIPY_DISTUTILS_REV).tar.gz numpy_distutils-latest.tar.gz
	mv numpy_distutils-latest.tar.gz upload/tmp/$(MAJOR).x
upload/tmp/$(MAJOR).x/numpy_distutils-latest.win32.exe: numpy_distutils
	cd numpy_distutils && python setup.py bdist_wininst
	mkdir -p upload/tmp/$(MAJOR).x
	cp numpy_distutils/dist/numpy_distutils-$(SCIPY_DISTUTILS_REV).win32.exe upload/tmp/$(MAJOR).x
	ln -sf numpy_distutils-$(SCIPY_DISTUTILS_REV).win32.exe numpy_distutils-latest.win32.exe
	mv numpy_distutils-latest.win32.exe upload/tmp/$(MAJOR).x

numpy_distutils_latest: upload/tmp/$(MAJOR).x/numpy_distutils-latest.tar.gz upload/tmp/$(MAJOR).x/numpy_distutils-latest.win32.exe

latest: f2py2e_latest numpy_distutils_latest

##############################################################################
# Upload files.
##############################################################################

upload/www/$(MAJOR).x/F2PY-$(MAJOR)-latest.tar.gz: upload/tmp/$(MAJOR).x/F2PY-$(MAJOR)-latest.tar.gz
	-mkdir -p `dirname $@`
	cp -P upload/tmp/$(MAJOR).x/F2PY-{$(MAJOR)-latest,$(REV)}.tar.gz upload/www/$(MAJOR).x
	$(UPLOADCMD) upload/tmp/$(MAJOR).x/F2PY-{$(MAJOR)-latest,$(REV)}.tar.gz $(UPLOADDIR)/$(MAJOR).x/
upload/www/$(MAJOR).x/numpy_distutils-latest.tar.gz: upload/tmp/$(MAJOR).x/numpy_distutils-latest.tar.gz
	-mkdir -p `dirname $@`
	cp -P upload/tmp/$(MAJOR).x/numpy_distutils-{latest,$(SCIPY_DISTUTILS_REV)}.tar.gz upload/www/$(MAJOR).x/
	$(UPLOADCMD) upload/tmp/$(MAJOR).x/numpy_distutils-{latest,$(SCIPY_DISTUTILS_REV)}.tar.gz $(UPLOADDIR)/$(MAJOR).x
upload/www/$(MAJOR).x/F2PY-$(MAJOR)-latest.win32.exe: upload/tmp/$(MAJOR).x/F2PY-$(MAJOR)-latest.win32.exe
	-mkdir -p `dirname $@`
	cp -P upload/tmp/$(MAJOR).x/F2PY-{$(MAJOR)-latest,$(REV)}.win32.exe upload/www/$(MAJOR).x
	$(UPLOADCMD) upload/tmp/$(MAJOR).x/F2PY-{$(MAJOR)-latest,$(REV)}.win32.exe $(UPLOADDIR)/$(MAJOR).x/
upload/www/$(MAJOR).x/numpy_distutils-latest.win32.exe: upload/tmp/$(MAJOR).x/numpy_distutils-latest.win32.exe
	-mkdir -p `dirname $@`
	cp -P upload/tmp/$(MAJOR).x/numpy_distutils-{latest,$(SCIPY_DISTUTILS_REV)}.win32.exe upload/www/$(MAJOR).x
	$(UPLOADCMD) upload/tmp/$(MAJOR).x/numpy_distutils-{latest,$(SCIPY_DISTUTILS_REV)}.win32.exe $(UPLOADDIR)/$(MAJOR).x/

upload/tmp/usersguide/index.html: docs/usersguide/index.txt $(UG_FILES_DEP:%=upload/www/usersguide/%)
	-mkdir -p upload/tmp/usersguide
	rest2html $< $@
upload/tmp/usersguide/f2py_usersguide.tex: docs/usersguide/index.txt $(UG_FILES_DEP:%=upload/www/usersguide/%)
	-mkdir -p upload/tmp/usersguide
	rest2latex $< $@
upload/tmp/usersguide/f2py_usersguide.pdf: upload/tmp/usersguide/f2py_usersguide.tex
	cd `dirname $@` && pdflatex `basename $<`
upload/tmp/usersguide/%.f: docs/usersguide/%.f
	-mkdir -p upload/tmp/usersguide
	cp $< $@
upload/tmp/usersguide/%.f90: docs/usersguide/%.f90
	-mkdir -p upload/tmp/usersguide
	cp $< $@
upload/tmp/usersguide/%.dat: docs/usersguide/%.dat
	-mkdir -p upload/tmp/usersguide
	cp $< $@
upload/tmp/usersguide/%.pyf: docs/usersguide/%.pyf
	-mkdir -p upload/tmp/usersguide
	cp $< $@
upload/tmp/usersguide/%.py: docs/usersguide/%.py
	-mkdir -p upload/tmp/usersguide
	cp $< $@
upload/www/usersguide/%: upload/tmp/usersguide/%
	-mkdir -p `dirname $@`
	cp -P $< $@
	$(UPLOADCMD) $@ $(UPLOADDIR)/usersguide

upload/tmp/FAQ.html: docs/FAQ.txt $(FAQ_DEPS:%=docs/%)
	-mkdir -p upload/tmp
	rest2html $< $@
upload/tmp/index.html: docs/README.txt $(README_DEPS:%=docs/%)
	-mkdir -p upload/tmp
	rest2html $< $@
upload/tmp/%.f: docs/%.f
	-mkdir -p upload/tmp
	cp $< $@
upload/tmp/%.html: docs/%.txt
	-mkdir -p upload/tmp
	rest2html $< $@
upload/www/%: upload/tmp/%
	-mkdir -p `dirname $@`
	cp -P $< $@
	$(UPLOADCMD) $@ $(UPLOADDIR)/

upload_web: $(WWW_WEB_FILES)
upload_ug: $(WWW_UG_FILES)
upload_src: $(WWW_SRC_FILES)
upload: upload_src upload_ug upload_web

generate_web: $(TMP_WEB_FILES)
generate: generate_web

##############################################################################
# Clean up
##############################################################################
clean:
	rm -f {tests/,tests/{f77,f90,mixed}/,docs/,docs/usersguide/,}*.{o,a,so,sl,pyc}
	rm -f {tests/,tests/{f77,f90,mixed}/,docs/,docs/usersguide/,}*~
distclean: clean
	rm -f {tests/,src/,}*~
	rm -f tests/*.{f,f90}
	rm -rf dist {docs/,docs/usersguide/,}build f2py2e numpy_distutils upload
	rm -f MANIFEST f2py?.? f2py

.PHONY: install test
