# -*- makefile -*-
# File: Makefile-foo
# Usage:
#     make -f Makefile-foo [MODE=opt|debug]
# Notes:
# 1) You must use GNU make; try `gmake ..' if `make' fails.
# 2) This file is auto-generated with f2py (version 2.264).
#    f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
#    written by Pearu Peterson <pearu@ioc.ee>.
#    See http://cens.ioc.ee/projects/f2py2e/
#    Generation date: Wed Sep 13 16:22:55 2000
#    $Revision: 1.2 $
#    $Date: 2000/09/17 16:10:27 $

# Recommendation notes produced by f2py2e/buildmakefile.py:
# *** 

PYINC =  -I/numeric/include/python1.5/Numeric -I/numeric/include/python1.5
INCLUDES = -I..
LIBS =  -L$(shell gcc -v 2>&1 | grep specs | sed -e 's/Reading specs from //g' | sed -e 's/\/specs//g') -lg2c 
LIBS=-L$$ABSOFT/lib -lfio -lf77math -lf90math
LIBS=-L/numeric/bin -lvast90 -L/usr/lib/gcc-lib/i586-mandrake-linux/2.95.2 -lg2c

# Wrapper generator:
F2PY = /home/pearu/bin/f2py-cvs

# Fortran compiler: Absoft f95 
FC = f95
FC = f90
FOPT = 
FDEBUG = 
FFLAGS = -B108 -YCFRL=1 -YCOM_NAMES=LCS -YCOM_PFX -YCOM_SFX=_ -YEXT_PFX -YEXT_NAMES=LCS
FFLAGS = 
# C compiler: cc ('gcc 2.x.x' 2.95.2)
CC = cc
COPT = 
CDEBUG = 
CFLAGS =  -fpic 

# Linker: ld ('GNU ld' 2.9.5)
LD = ld
LDFLAGS = -shared -s 
SO = .so

ifeq '$(MODE)' 'debug'
FFLAGS += $(FDEBUG)
CFLAGS += $(CDEBUG)
endif
ifeq '$(MODE)' 'opt'
FFLAGS += $(FOPT)
CFLAGS += $(COPT)
endif
FFLAGS += $(INCLUDES)
CFLAGS += $(PYINC) $(INCLUDES)

SRCC = ../fortranobject.c
SRCF = mod.f90 bar.f foo90.f90 wrap.f
SRCS = $(SRCC) $(SRCF)
OBJC = $(filter %.o,$(SRCC:.c=.o) $(SRCC:.cc=.o) $(SRCC:.C=.o))
OBJF = $(filter %.o,$(SRCF:.f90=.o) $(SRCF:.f=.o) $(SRCF:.F=.o) $(SRCF:.for=.o))
OBJS = $(OBJC) $(OBJF)

INSTALLNAME = f2py2e-apps
INSTALLDIRECTORY = /numeric/lib/python1.5/site-packages/$(INSTALLNAME)
INSTALLDIR = install -d -c
INSTALLEXEC = install -m 755 -c

all: foo

foo: foomodule$(SO)
foomodule$(SO) : foomodule.o $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $< $(OBJS) $(LIBS)

foomodule.o: foomodule.c


$(OBJS) : $(SRCS)
%.o : %.f ; $(FC) -c $(FFLAGS) $< 
%.o : %.f90 ; $(FC) -c $(FFLAGS) $< 

test: foomodule$(SO)
	python -c 'import foo;print foo.__doc__'
install: foomodule$(SO)
	$(INSTALLDIR) $(INSTALLDIRECTORY)
	$(INSTALLEXEC) foomodule$(SO) $(INSTALLDIRECTORY)
	cd $(INSTALLDIRECTORY) && echo "$(INSTALLNAME)" > ../$(INSTALLNAME).pth

.PHONY: clean distclean debug test install foo
debug:
	echo "OBJS=$(OBJS)"
	echo "SRCS=$(SRCS)"
clean:
	$(RM) *.o *.mod core foomodule.{dvi,log} $(OBJS)
distclean: clean
	$(RM) *.so *.sl foomodule.{tex,so}
	$(RM) .f2py_get_compiler_*
