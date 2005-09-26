#
# Written by Patrick Guio <patrick.guio@fys.uio.no>
#

SUBDIRS = array generate meta

blitzdir = $(includedir)/blitz

blitz_HEADERS = applics.h array-impl.h array-old.h array.h bench.cc bench.h \
benchext.cc benchext.h blitz.h bzdebug.h compiler.h config.h \
etbase.h extremum.h funcs.h indexexpr.h limits-hack.h listinit.h matbops.h \
matdiag.h matexpr.h matgen.h mathf2.h mathfunc.h matltri.h matref.h matrix.cc \
matrix.h matsymm.h mattoep.h matuops.h matutri.h memblock.cc memblock.h \
minmax.h mstruct.h numinquire.h numtrait.h ops.h prettyprint.h promote-old.h \
promote.h rand-dunif.h rand-mt.h rand-normal.h rand-tt800.h rand-uniform.h \
random.h randref.h range.h reduce.h shapecheck.h tau.h timer.h tiny.h \
tinymat.h tinymatexpr.h tinymatio.cc tinyvec-et.h tinyvec.cc tinyvec.h \
tinyvecio.cc tinyveciter.h traversal.cc traversal.h tuning.h tvcross.h \
tvecglobs.h update.h vecaccum.cc vecall.cc vecany.cc vecbfn.cc vecbops.cc \
veccount.cc vecdelta.cc vecdot.cc vecexpr.h vecexprwrap.h vecglobs.cc \
vecglobs.h vecio.cc veciter.h vecmax.cc vecmin.cc vecnorm.cc vecnorm1.cc \
vecpick.cc vecpick.h vecpickio.cc vecpickiter.h vecsum.cc vector-et.h \
vector.cc vector.h vecuops.cc vecwhere.cc vecwhere.h wrap-climits.h zero.cc \
zero.h

EXTRA_DIST = MAT-NOTE.gz config-ICL.h config-KCC.h config-SC4.0.h \
config-SGi.h config-g++2.7.2.h config-mwerks.h config-xlC.h

clean-local:
	-rm -rf config.h 
