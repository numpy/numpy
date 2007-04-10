This is a stripped down version of libcprops.  It only has the files in it necessary for building the thread pool library.

./configure doesn't work because it is expecting some other files that aren't around.

To build use:

make -f ThreadMakefile

I've edited the Makefile to only build the needed library files as well as a simple main file that tests it.

To test, run:

time ./main

Todo:

	* How much of the config.h file is really needed?
	* If we can get rid of log.c, we also get rid of str.c.  These guys use a lot of the unix specific stuff.
	  How much of log.c do we use?
	* Is it possible to get setup.py to do all the configuration we need?
	
