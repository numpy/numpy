from distutils.command.sdist import *
from distutils.command.sdist import sdist as old_sdist

class sdist(old_sdist):
    def add_defaults (self):
        old_sdist.add_defaults(self)
        if self.distribution.has_f_libraries():
            build_flib = self.get_finalized_command('build_flib')
            self.filelist.extend(build_flib.get_source_files())

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
        print 'common_base_dir:', find_common_base(files)
        #files = [os.path.abspath(x) for x in files]
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
    
        if hasattr(os, 'link'):        # can make hard links on this system
            link = 'hard'
            msg = "making hard links in %s..." % base_dir
        else:                           # nope, have to copy
            link = None
            msg = "copying files to %s..." % base_dir

        if not files:
            self.warn("no files to distribute -- empty manifest?")
        else:
            self.announce(msg)
        
        print 'base:',base_dir, dest_files[0]
        dest_files = [os.path.join(base_dir,file) for file in dest_files]
        file_pairs = zip(files,dest_files)    
        for file,dest in file_pairs:
            if not os.path.isfile(file):
                self.warn("'%s' not a regular file -- skipping" % file)
            else:
                #ej: here is the only change -- made to handle
                # absolute paths to files as well as relative
                #par,file_name = os.path.split(file)
                #dest = os.path.join(base_dir, file_name)
                # end of changes
                
                # old code
                #dest = os.path.join(base_dir, file)
                #end old code
                print dest
                self.copy_file(file, dest, link=link)

        self.distribution.metadata.write_pkg_info(base_dir)
        #raise ValueError
    # make_release_tree ()

def remove_common_base(files):
    """ Remove the greatest common base directory from all the
        absolute file paths in the list of files.  files in the
        list without a parent directory are not affected.
    """
    rel_files = filter(lambda x: not os.path.dirname(x),files)
    abs_files = filter(os.path.dirname,files)
    print 're files:', rel_files
    base = find_common_base(abs_files)
    # will leave files with local path unaffected
    # and maintains original file order
    results = [string.replace(file,base,'') for file in files]
    return results

def find_common_base(files):
    """ Find the "greatest common base directory" of a list of files
    """
    if not files:
        return files
    result = ''
    d,f = os.path.split(files[0])
    keep_looking = 1    
    while(keep_looking and d):
        keep_looking = 0
        print d
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