
import os
import sys

from distutils.command.sdist import *
from distutils.command.sdist import sdist as old_sdist
from scipy_distutils import log
from scipy_distutils import line_endings

class sdist(old_sdist):
    def add_defaults (self):
        old_sdist.add_defaults(self)

        if self.distribution.has_data_files():
            self.filelist.extend(self.distribution.get_data_files())

    def make_release_tree (self, base_dir, files):
        """Create the directory tree that will become the source
        distribution archive.  All directories implied by the filenames in
        'files' are created under 'base_dir', and then we hard link or copy
        (if hard linking is unavailable) those files into place.
        Essentially, this duplicates the developer's source tree, but in a
        directory named after the distribution, containing only the files
        to be distributed.
        """
        # Create all the directories under 'base_dir' necessary to
        # put 'files' there; the 'mkpath()' is just so we don't die
        # if the manifest happens to be empty.
        
        dest_files = remove_common_base(files)
        self.mkpath(base_dir)
        dir_util.create_tree(base_dir, dest_files,
                             verbose=self.verbose, dry_run=self.dry_run)

        # And walk over the list of files, either making a hard link (if
        # os.link exists) to each one that doesn't already exist in its
        # corresponding location under 'base_dir', or copying each file
        # that's out-of-date in 'base_dir'.  (Usually, all files will be
        # out-of-date, because by default we blow away 'base_dir' when
        # we're done making the distribution archives.)

        
    
        if 0 and hasattr(os, 'link'):        # can make hard links on this system
            link = 'hard'
            msg = "making hard links in %s..." % base_dir
        else:                           # nope, have to copy
            link = None
            msg = "copying files to %s..." % base_dir
        self._use_hard_link = not not link

        if not files:
            log.warn("no files to distribute -- empty manifest?")
        else:
            log.info(msg)
        
        dest_files = [os.path.join(base_dir,file) for file in dest_files]
        file_pairs = zip(files,dest_files)
        for file,dest in file_pairs:
            if not os.path.isfile(file):
                log.warn("'%s' not a regular file -- skipping", file)
            else:
                #ej: here is the only change -- made to handle
                # absolute paths to files as well as relative
                #par,file_name = os.path.split(file)
                #dest = os.path.join(base_dir, file_name)
                # end of changes
                
                # old code
                #dest = os.path.join(base_dir, file)
                #end old code
                self.copy_file(file, dest, link=link)

        self.distribution.metadata.write_pkg_info(base_dir)
        #raise ValueError
    # make_release_tree ()

    def make_distribution (self):
        """ Overridden to force a build of zip files to have Windows line 
            endings and tar balls to have Unix line endings.
            
            Create the source distribution(s).  First, we create the release
            tree with 'make_release_tree()'; then, we create all required
            archive files (according to 'self.formats') from the release tree.
            Finally, we clean up by blowing away the release tree (unless
            'self.keep_temp' is true).  The list of archive files created is
            stored so it can be retrieved later by 'get_archive_files()'.
        """
        # Don't warn about missing meta-data here -- should be (and is!)
        # done elsewhere.
        base_dir = self.distribution.get_fullname()
        base_name = os.path.join(self.dist_dir, base_dir)
        files = map(os.path.abspath, self.filelist.files)
        self.make_release_tree(base_dir, files)
        archive_files = []              # remember names of files we create
        for fmt in self.formats:
            modified_files,restore_func = self.convert_line_endings(base_dir,fmt)
            file = self.make_archive(base_name, fmt, base_dir=base_dir)
            archive_files.append(file)
            if self._use_hard_link:
                map(restore_func,modified_files)

        self.archive_files = archive_files

        if not self.keep_temp:
            dir_util.remove_tree(base_dir, self.verbose, self.dry_run)

    def convert_line_endings(self,base_dir,fmt):
        """ Convert all text files in a tree to have correct line endings.
            
            gztar --> \n   (Unix style)
            zip   --> \r\n (Windows style)
        """
        if fmt == 'gztar':
            return line_endings.dos2unix_dir(base_dir),line_endings.unix2dos
        elif fmt == 'zip':
            return line_endings.unix2dos_dir(base_dir),line_endings.dos2unix
        return [],lambda a:None

def remove_common_base(files):
    """ Remove the greatest common base directory from all the
        absolute file paths in the list of files.  files in the
        list without a parent directory are not affected.
    """
    rel_files = filter(lambda x: not os.path.dirname(x),files)
    abs_files = filter(os.path.dirname,files)
    base = find_common_base(abs_files)
    # will leave files with local path unaffected
    # and maintains original file order
    results = [file[len(base):] for file in files]
    return results

def find_common_base(files):
    """ Find the "greatest common base directory" of a list of files
    """
    if not files:
        return ''
    result = ''
    d,f = os.path.split(files[0])
    keep_looking = 1    
    while(keep_looking and d):
        keep_looking = 0
        for file in files:
            if string.find('start'+file,'start'+d) == -1:
                keep_looking = 1
                break
        if keep_looking:
            d,f = os.path.split(d)
        else:
            result = d
            
    if d: 
        d = os.path.join(d,'')
    return d        
