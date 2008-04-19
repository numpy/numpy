"""A file interface for handling local and remote data files.
The goal of datasource is to abstract some of the file system operations when
dealing with data files so the researcher doesn't have to know all the
low-level details.  Through datasource, a researcher can obtain and use a
file with one function call, regardless of location of the file.

DataSource is meant to augment standard python libraries, not replace them.
It should work seemlessly with standard file IO operations and the os module.

DataSource files can originate locally or remotely:

- local files : '/home/guido/src/local/data.txt'
- URLs (http, ftp, ...) : 'http://www.scipy.org/not/real/data.txt'

DataSource files can also be compressed or uncompressed.  Currently only gzip
and bz2 are supported.

Example:

    >>> # Create a DataSource, use os.curdir (default) for local storage.
    >>> ds = datasource.DataSource()
    >>>
    >>> # Open a remote file.
    >>> # DataSource downloads the file, stores it locally in:
    >>> #     './www.google.com/index.html'
    >>> # opens the file and returns a file object.
    >>> fp = ds.open('http://www.google.com/index.html')
    >>>
    >>> # Use the file as you normally would
    >>> fp.read()
    >>> fp.close()

"""

__docformat__ = "restructuredtext en"

import os
import tempfile
from shutil import rmtree
from urllib2 import urlopen, URLError
from urlparse import urlparse

# TODO: .zip support, .tar support?
_file_openers = {None: open}
try:
    import bz2
    _file_openers[".bz2"] = bz2.BZ2File
except ImportError:
    pass
try:
    import gzip
    _file_openers[".gz"] = gzip.open
except ImportError:
    pass


def open(path, mode='r', destpath=os.curdir):
    """Open ``path`` with ``mode`` and return the file object.

    If ``path`` is an URL, it will be downloaded, stored in the DataSource
    directory and opened from there.

    *Parameters*:

        path : {string}

        mode : {string}, optional

        destpath : {string}, optional
            Destination directory where URLs will be downloaded and stored.

    *Returns*:

        file object

    """

    ds = DataSource(destpath)
    return ds.open(path, mode)


class DataSource (object):
    """A generic data source file (file, http, ftp, ...).

    DataSources could be local files or remote files/URLs.  The files may
    also be compressed or uncompressed.  DataSource hides some of the low-level
    details of downloading the file, allowing you to simply pass in a valid
    file path (or URL) and obtain a file object.

    *Methods*:

        - exists : test if the file exists locally or remotely
        - abspath : get absolute path of the file in the DataSource directory
        - open : open the file

    *Example URL DataSource*::

        # Initialize DataSource with a local directory, default is os.curdir.
        ds = DataSource('/home/guido')

        # Open remote file.
        # File will be downloaded and opened from here:
        #     /home/guido/site/xyz.txt
        ds.open('http://fake.xyz.web/site/xyz.txt')

    *Example using DataSource for temporary files*::

        # Initialize DataSource with 'None' for the local directory.
        ds = DataSource(None)

        # Open local file.
        # Opened file exists in a temporary directory like:
        #     /tmp/tmpUnhcvM/foobar.txt
        # Temporary directories are deleted when the DataSource is deleted.
        ds.open('/home/guido/foobar.txt')

    *Notes*:
        BUG : URLs require a scheme string ('http://') to be used.
              www.google.com will fail.

              >>> repos.exists('www.google.com/index.html')
              False

              >>> repos.exists('http://www.google.com/index.html')
              True

    """

    def __init__(self, destpath=os.curdir):
        """Create a DataSource with a local path at destpath."""
        if destpath:
            self._destpath = os.path.abspath(destpath)
            self._istmpdest = False
        else:
            self._destpath = tempfile.mkdtemp()
            self._istmpdest = True

    def __del__(self):
        # Remove temp directories
        if self._istmpdest:
            rmtree(self._destpath)

    def _iszip(self, filename):
        """Test if the filename is a zip file by looking at the file extension.
        """
        fname, ext = os.path.splitext(filename)
        return ext in _file_openers.keys()

    def _iswritemode(self, mode):
        """Test if the given mode will open a file for writing."""

        # Currently only used to test the bz2 files.
        _writemodes = ("w", "+")
        for c in mode:
            if c in _writemodes:
                return True
        return False

    def _splitzipext(self, filename):
        """Split zip extension from filename and return filename.

        *Returns*:
            base, zip_ext : {tuple}

        """

        if self._iszip(filename):
            return os.path.splitext(filename)
        else:
            return filename, None

    def _possible_names(self, filename):
        """Return a tuple containing compressed filename variations."""
        names = [filename]
        if not self._iszip(filename):
            for zipext in _file_openers.keys():
                if zipext:
                    names.append(filename+zipext)
        return names

    def _isurl(self, path):
        """Test if path is a net location.  Tests the scheme and netloc."""

        # BUG : URLs require a scheme string ('http://') to be used.
        #       www.google.com will fail.
        #       Should we prepend the scheme for those that don't have it and
        #       test that also?  Similar to the way we append .gz and test for
        #       for compressed versions of files.

        scheme, netloc, upath, uparams, uquery, ufrag = urlparse(path)
        return bool(scheme and netloc)

    def _cache(self, path):
        """Cache the file specified by path.

        Creates a copy of the file in the datasource cache.

        """

        upath = self.abspath(path)

        # ensure directory exists
        if not os.path.exists(os.path.dirname(upath)):
            os.makedirs(os.path.dirname(upath))

        # TODO: Doesn't handle compressed files!
        if self._isurl(path):
            try:
                openedurl = urlopen(path)
                file(upath, 'w').write(openedurl.read())
            except URLError:
                raise URLError("URL not found: ", path)
        else:
            try:
                # TODO: Why not just copy the file with shutils.copyfile?
                fp = file(path, 'r')
                file(upath, 'w').write(fp.read())
            except IOError:
                raise IOError("File not found: ", path)
        return upath

    def _findfile(self, path):
        """Searches for ``path`` and returns full path if found.

        If path is an URL, _findfile will cache a local copy and return
        the path to the cached file.
        If path is a local file, _findfile will return a path to that local
        file.

        The search will include possible compressed versions of the file and
        return the first occurence found.

        """

        # Build list of possible local file paths
        if not self._isurl(path):
            # Valid local paths
            filelist = self._possible_names(path)
            # Paths in self._destpath
            filelist += self._possible_names(self.abspath(path))
        else:
            # Cached URLs in self._destpath
            filelist = self._possible_names(self.abspath(path))
            # Remote URLs
            filelist = filelist + self._possible_names(path)

        for name in filelist:
            if self.exists(name):
                if self._isurl(name):
                    name = self._cache(name)
                return name
        return None

    def abspath(self, path):
        """Return absolute path of ``path`` in the DataSource directory.

        If ``path`` is an URL, the ``abspath`` will be either the location
        the file exists locally or the location it would exist when opened
        using the ``open`` method.

        The functionality is idential to os.path.abspath.

        *Parameters*:

            path : {string}
                Can be a local file or a remote URL.

        *Returns*:

            Complete path, rooted in the DataSource destination directory.

        *See Also*:

            `open` : Method that downloads and opens files.

        """

        # TODO:  This should be more robust.  Handles case where path includes
        #        the destpath, but not other sub-paths. Failing case:
        #        path = /home/guido/datafile.txt
        #        destpath = /home/alex/
        #        upath = self.abspath(path)
        #        upath == '/home/alex/home/guido/datafile.txt'

        # handle case where path includes self._destpath
        splitpath = path.split(self._destpath, 2)
        if len(splitpath) > 1:
            path = splitpath[1]
        scheme, netloc, upath, uparams, uquery, ufrag = urlparse(path)
        netloc = self._sanitize_relative_path(netloc)
        upath = self._sanitize_relative_path(upath)
        return os.path.join(self._destpath, netloc, upath)

    def _sanitize_relative_path(self, path):
        """Return a sanitised relative path for which
        os.path.abspath(os.path.join(base, path)).startswith(base)
        """
        last = None
        path = os.path.normpath(path)
        while path != last:
            last = path
            # Note: os.path.join treats '/' as os.sep
            path = path.lstrip(os.sep).lstrip('/')
            path = path.lstrip(os.pardir).lstrip('..')
        return path

    def exists(self, path):
        """Test if ``path`` exists.

        Test if ``path`` exists as (and in this order):

        - a local file.
        - a remote URL that have been downloaded and stored locally in the
          DataSource directory.
        - a remote URL that has not been downloaded, but is valid and
          accessible.

        *Parameters*:

            path : {string}
                Can be a local file or a remote URL.

        *Returns*:

            boolean

        *See Also*:

            `abspath`

        *Notes*

            When ``path`` is an URL, ``exist`` will return True if it's either
            stored locally in the DataSource directory, or is a valid remote
            URL.  DataSource does not discriminate between to two, the file
            is accessible if it exists in either location.

        """

        # Test local path
        if os.path.exists(path):
            return True

        # Test cached url
        upath = self.abspath(path)
        if os.path.exists(upath):
            return True

        # Test remote url
        if self._isurl(path):
            try:
                netfile = urlopen(path)
                del(netfile)
                return True
            except URLError:
                return False
        return False

    def open(self, path, mode='r'):
        """Open ``path`` with ``mode`` and return the file object.

        If ``path`` is an URL, it will be downloaded, stored in the DataSource
        directory and opened from there.

        *Parameters*:

            path : {string}

            mode : {string}, optional


        *Returns*:

            file object

        """

        # TODO: There is no support for opening a file for writing which
        #       doesn't exist yet (creating a file).  Should there be?

        # TODO: Add a ``subdir`` parameter for specifying the subdirectory
        #       used to store URLs in self._destpath.

        if self._isurl(path) and self._iswritemode(mode):
            raise ValueError("URLs are not writeable")

        # NOTE: _findfile will fail on a new file opened for writing.
        found = self._findfile(path)
        if found:
            _fname, ext = self._splitzipext(found)
            if ext == 'bz2':
                mode.replace("+", "")
            return _file_openers[ext](found, mode=mode)
        else:
            raise IOError("%s not found." % path)


class Repository (DataSource):
    """A data Repository where multiple DataSource's share a base URL/directory.

    Repository extends DataSource by prepending a base URL (or directory) to
    all the files it handles. Use a Repository when you will be working with
    multiple files from one base URL.  Initialize the Respository with the
    base URL, then refer to each file by it's filename only.

    *Methods*:

        - exists : test if the file exists locally or remotely
        - abspath : get absolute path of the file in the DataSource directory
        - open : open the file

    *Toy example*::

        # Analyze all files in the repository.
        repos = Repository('/home/user/data/dir/')
        for filename in filelist:
            fp = repos.open(filename)
            fp.analyze()
            fp.close()

        # Similarly you could use a URL for a repository.
        repos = Repository('http://www.xyz.edu/data')

    """

    def __init__(self, baseurl, destpath=os.curdir):
        """Create a Repository with a shared url or directory of baseurl."""
        DataSource.__init__(self, destpath=destpath)
        self._baseurl = baseurl

    def __del__(self):
        DataSource.__del__(self)

    def _fullpath(self, path):
        """Return complete path for path.  Prepends baseurl if necessary."""
        splitpath = path.split(self._baseurl, 2)
        if len(splitpath) == 1:
            result = os.path.join(self._baseurl, path)
        else:
            result = path    # path contains baseurl already
        return result

    def _findfile(self, path):
        """Extend DataSource method to prepend baseurl to ``path``."""
        return DataSource._findfile(self, self._fullpath(path))

    def abspath(self, path):
        """Extend DataSource method to prepend baseurl to ``path``."""
        return DataSource.abspath(self, self._fullpath(path))

    def exists(self, path):
        """Extend DataSource method to prepend baseurl to ``path``."""
        return DataSource.exists(self, self._fullpath(path))

    def open(self, path, mode='r'):
        """Extend DataSource method to prepend baseurl to ``path``."""
        return DataSource.open(self, self._fullpath(path), mode)

    def listdir(self):
        '''List files in the source Repository.'''
        if self._isurl(self._baseurl):
            raise NotImplementedError, \
                  "Directory listing of URLs, not supported yet."
        else:
            return os.listdir(self._baseurl)
