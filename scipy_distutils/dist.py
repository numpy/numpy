from distutils.dist import *
from distutils.dist import Distribution as OldDistribution
from distutils.errors import DistutilsSetupError

from types import *

class Distribution (OldDistribution):
    def __init__ (self, attrs=None):
        self.fortran_libraries = None
        OldDistribution.__init__(self, attrs)

    def has_f2py_sources (self):
        if self.has_ext_modules():
            for ext in self.ext_modules:
                for source in ext.sources:
                    (base, file_ext) = os.path.splitext(source)
                    if file_ext == ".pyf":       # f2py interface file
                        return 1
        return 0
    
    def has_f_libraries(self):
        if self.fortran_libraries and len(self.fortran_libraries) > 0:
            return 1
        return self.has_f2py_sources() # f2py might generate fortran sources.

        if hasattr(self,'_been_here_has_f_libraries'):
            return 0
        if self.has_ext_modules():
            # extension module sources may contain fortran files,
            # extract them to fortran_libraries.
            for ext in self.ext_modules:
                self.fortran_sources_to_flib(ext)
        self._been_here_has_f_libraries = None
        return self.fortran_libraries and len(self.fortran_libraries) > 0

    def fortran_sources_to_flib(self, ext):
        """
        Extract fortran files from ext.sources and append them to
        fortran_libraries item having the same name as ext.
        """
        sources = []
        f_files = []
        match = re.compile(r'.*[.](f90|f95|f77|for|ftn|f)\Z',re.I).match
        for file in ext.sources:
            if match(file):
                f_files.append(file)
            else:
                sources.append(file)
        if not f_files:
            return

        ext.sources = sources

        if self.fortran_libraries is None:
            self.fortran_libraries = []

        name = ext.name
        flib = None
        for n,d in self.fortran_libraries:
            if n == name:
                flib = d
                break
        if flib is None:
            flib = {'sources':[]}
            self.fortran_libraries.append((name,flib))

        flib['sources'].extend(f_files)

    def check_data_file_list(self):
        """Ensure that the list of data_files (presumably provided as a
           command option 'data_files') is valid, i.e. it is a list of
           2-tuples, where the tuples are (name, list_of_libraries).
           Raise DistutilsSetupError if the structure is invalid anywhere;
           just returns otherwise."""
        print 'check_data_file_list'
        if type(self.data_files) is not ListType:
            raise DistutilsSetupError, \
                  "'data_files' option must be a list of tuples"

        for lib in self.data_files:
            if type(lib) is not TupleType and len(lib) != 2:
                raise DistutilsSetupError, \
                      "each element of 'data_files' must a 2-tuple"

            if type(lib[0]) is not StringType:
                raise DistutilsSetupError, \
                      "first element of each tuple in 'data_files' " + \
                      "must be a string (the package with the data_file)"

            if type(lib[1]) is not ListType:
                raise DistutilsSetupError, \
                      "second element of each tuple in 'data_files' " + \
                      "must be a list of files."
        # for lib

    # check_data_file_list ()
   
    def get_data_files (self):
        print 'get_data_files'
        self.check_data_file_list()
        filenames = []
        
        # Gets data files specified
        for ext in self.data_files:
            filenames.extend(ext[1])

        return filenames
