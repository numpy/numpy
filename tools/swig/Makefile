# List all of the subdirectories here for recursive make
SUBDIRS = test

# Default target
.PHONY : default
default:
	@echo "There is no default make target for this Makefile"
	@echo "Valid make targets are:"
	@echo "    test           - Compile and run tests of numpy.i"
	@echo "    doc            - Generate numpy.i documentation"
	@echo "    all            - make test + doc"
	@echo "    clean          - Remove generated files recursively"

# Target all
.PHONY : all
all: $(SUBDIRS)

# Target test
.PHONY : test
test:
	cd $@ && make $@

# Target clean
.PHONY : clean
clean:
	@for dir in $(SUBDIRS); do                \
	    echo ;                                \
	    echo Running \'make clean\' in $$dir; \
	    cd $$dir && make clean && cd ..;      \
	done;                                     \
	echo
