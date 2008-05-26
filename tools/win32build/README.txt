This directory contains various scripts and code to build installers for
windows
	- cpuid: contains a mini lib to detect SSE.
	- cpucaps: nsis plugin to add the ability to detect SSE for installers.
	- *nsi scripts: actual nsis scripts to build the installer
	- build.py: script to build various versions of python binaries
	  (several archs, several python versions)

To build the binaries, you need blas/lapack/atlas for all architectures.
