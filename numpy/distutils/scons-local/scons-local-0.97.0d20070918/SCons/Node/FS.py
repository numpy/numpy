"""scons.Node.FS

File system nodes.

These Nodes represent the canonical external objects that people think
of when they think of building software: files and directories.

This holds a "default_fs" variable that should be initialized with an FS
that can be used by scripts or modules looking for the canonical default.

"""

#
# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007 The SCons Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

__revision__ = "src/engine/SCons/Node/FS.py 2446 2007/09/18 11:41:57 knight"

import os
import os.path
import shutil
import stat
import string
import sys
import time
import cStringIO

import SCons.Action
from SCons.Debug import logInstanceCreation
import SCons.Errors
import SCons.Memoize
import SCons.Node
import SCons.Node.Alias
import SCons.Subst
import SCons.Util
import SCons.Warnings

from SCons.Debug import Trace

# The max_drift value:  by default, use a cached signature value for
# any file that's been untouched for more than two days.
default_max_drift = 2*24*60*60

#
# We stringify these file system Nodes a lot.  Turning a file system Node
# into a string is non-trivial, because the final string representation
# can depend on a lot of factors:  whether it's a derived target or not,
# whether it's linked to a repository or source directory, and whether
# there's duplication going on.  The normal technique for optimizing
# calculations like this is to memoize (cache) the string value, so you
# only have to do the calculation once.
#
# A number of the above factors, however, can be set after we've already
# been asked to return a string for a Node, because a Repository() or
# BuildDir() call or the like may not occur until later in SConscript
# files.  So this variable controls whether we bother trying to save
# string values for Nodes.  The wrapper interface can set this whenever
# they're done mucking with Repository and BuildDir and the other stuff,
# to let this module know it can start returning saved string values
# for Nodes.
#
Save_Strings = None

def save_strings(val):
    global Save_Strings
    Save_Strings = val

#
# SCons.Action objects for interacting with the outside world.
#
# The Node.FS methods in this module should use these actions to
# create and/or remove files and directories; they should *not* use
# os.{link,symlink,unlink,mkdir}(), etc., directly.
#
# Using these SCons.Action objects ensures that descriptions of these
# external activities are properly displayed, that the displays are
# suppressed when the -s (silent) option is used, and (most importantly)
# the actions are disabled when the the -n option is used, in which case
# there should be *no* changes to the external file system(s)...
#

if hasattr(os, 'link'):
    def _hardlink_func(fs, src, dst):
        # If the source is a symlink, we can't just hard-link to it
        # because a relative symlink may point somewhere completely
        # different.  We must disambiguate the symlink and then
        # hard-link the final destination file.
        while fs.islink(src):
            link = fs.readlink(src)
            if not os.path.isabs(link):
                src = link
            else:
                src = os.path.join(os.path.dirname(src), link)
        fs.link(src, dst)
else:
    _hardlink_func = None

if hasattr(os, 'symlink'):
    def _softlink_func(fs, src, dst):
        fs.symlink(src, dst)
else:
    _softlink_func = None

def _copy_func(fs, src, dest):
    shutil.copy2(src, dest)
    st = fs.stat(src)
    fs.chmod(dest, stat.S_IMODE(st[stat.ST_MODE]) | stat.S_IWRITE)


Valid_Duplicates = ['hard-soft-copy', 'soft-hard-copy',
                    'hard-copy', 'soft-copy', 'copy']

Link_Funcs = [] # contains the callables of the specified duplication style

def set_duplicate(duplicate):
    # Fill in the Link_Funcs list according to the argument
    # (discarding those not available on the platform).

    # Set up the dictionary that maps the argument names to the
    # underlying implementations.  We do this inside this function,
    # not in the top-level module code, so that we can remap os.link
    # and os.symlink for testing purposes.
    link_dict = {
        'hard' : _hardlink_func,
        'soft' : _softlink_func,
        'copy' : _copy_func
    }

    if not duplicate in Valid_Duplicates:
        raise SCons.Errors.InternalError, ("The argument of set_duplicate "
                                           "should be in Valid_Duplicates")
    global Link_Funcs
    Link_Funcs = []
    for func in string.split(duplicate,'-'):
        if link_dict[func]:
            Link_Funcs.append(link_dict[func])

def LinkFunc(target, source, env):
    # Relative paths cause problems with symbolic links, so
    # we use absolute paths, which may be a problem for people
    # who want to move their soft-linked src-trees around. Those
    # people should use the 'hard-copy' mode, softlinks cannot be
    # used for that; at least I have no idea how ...
    src = source[0].abspath
    dest = target[0].abspath
    dir, file = os.path.split(dest)
    if dir and not target[0].fs.isdir(dir):
        os.makedirs(dir)
    if not Link_Funcs:
        # Set a default order of link functions.
        set_duplicate('hard-soft-copy')
    fs = source[0].fs
    # Now link the files with the previously specified order.
    for func in Link_Funcs:
        try:
            func(fs, src, dest)
            break
        except (IOError, OSError):
            # An OSError indicates something happened like a permissions
            # problem or an attempt to symlink across file-system
            # boundaries.  An IOError indicates something like the file
            # not existing.  In either case, keeping trying additional
            # functions in the list and only raise an error if the last
            # one failed.
            if func == Link_Funcs[-1]:
                # exception of the last link method (copy) are fatal
                raise
            else:
                pass
    return 0

Link = SCons.Action.Action(LinkFunc, None)
def LocalString(target, source, env):
    return 'Local copy of %s from %s' % (target[0], source[0])

LocalCopy = SCons.Action.Action(LinkFunc, LocalString)

def UnlinkFunc(target, source, env):
    t = target[0]
    t.fs.unlink(t.abspath)
    return 0

Unlink = SCons.Action.Action(UnlinkFunc, None)

def MkdirFunc(target, source, env):
    t = target[0]
    if not t.exists():
        t.fs.mkdir(t.abspath)
    return 0

Mkdir = SCons.Action.Action(MkdirFunc, None, presub=None)

MkdirBuilder = None

def get_MkdirBuilder():
    global MkdirBuilder
    if MkdirBuilder is None:
        import SCons.Builder
        import SCons.Defaults
        # "env" will get filled in by Executor.get_build_env()
        # calling SCons.Defaults.DefaultEnvironment() when necessary.
        MkdirBuilder = SCons.Builder.Builder(action = Mkdir,
                                             env = None,
                                             explain = None,
                                             is_explicit = None,
                                             target_scanner = SCons.Defaults.DirEntryScanner,
                                             name = "MkdirBuilder")
    return MkdirBuilder

class _Null:
    pass

_null = _Null()

DefaultSCCSBuilder = None
DefaultRCSBuilder = None

def get_DefaultSCCSBuilder():
    global DefaultSCCSBuilder
    if DefaultSCCSBuilder is None:
        import SCons.Builder
        # "env" will get filled in by Executor.get_build_env()
        # calling SCons.Defaults.DefaultEnvironment() when necessary.
        act = SCons.Action.Action('$SCCSCOM', '$SCCSCOMSTR')
        DefaultSCCSBuilder = SCons.Builder.Builder(action = act,
                                                   env = None,
                                                   name = "DefaultSCCSBuilder")
    return DefaultSCCSBuilder

def get_DefaultRCSBuilder():
    global DefaultRCSBuilder
    if DefaultRCSBuilder is None:
        import SCons.Builder
        # "env" will get filled in by Executor.get_build_env()
        # calling SCons.Defaults.DefaultEnvironment() when necessary.
        act = SCons.Action.Action('$RCS_COCOM', '$RCS_COCOMSTR')
        DefaultRCSBuilder = SCons.Builder.Builder(action = act,
                                                  env = None,
                                                  name = "DefaultRCSBuilder")
    return DefaultRCSBuilder

# Cygwin's os.path.normcase pretends it's on a case-sensitive filesystem.
_is_cygwin = sys.platform == "cygwin"
if os.path.normcase("TeSt") == os.path.normpath("TeSt") and not _is_cygwin:
    def _my_normcase(x):
        return x
else:
    def _my_normcase(x):
        return string.upper(x)



class DiskChecker:
    def __init__(self, type, do, ignore):
        self.type = type
        self.do = do
        self.ignore = ignore
        self.set_do()
    def set_do(self):
        self.__call__ = self.do
    def set_ignore(self):
        self.__call__ = self.ignore
    def set(self, list):
        if self.type in list:
            self.set_do()
        else:
            self.set_ignore()

def do_diskcheck_match(node, predicate, errorfmt):
    result = predicate()
    try:
        # If calling the predicate() cached a None value from stat(),
        # remove it so it doesn't interfere with later attempts to
        # build this Node as we walk the DAG.  (This isn't a great way
        # to do this, we're reaching into an interface that doesn't
        # really belong to us, but it's all about performance, so
        # for now we'll just document the dependency...)
        if node._memo['stat'] is None:
            del node._memo['stat']
    except (AttributeError, KeyError):
        pass
    if result:
        raise TypeError, errorfmt % node.abspath

def ignore_diskcheck_match(node, predicate, errorfmt):
    pass

def do_diskcheck_rcs(node, name):
    try:
        rcs_dir = node.rcs_dir
    except AttributeError:
        if node.entry_exists_on_disk('RCS'):
            rcs_dir = node.Dir('RCS')
        else:
            rcs_dir = None
        node.rcs_dir = rcs_dir
    if rcs_dir:
        return rcs_dir.entry_exists_on_disk(name+',v')
    return None

def ignore_diskcheck_rcs(node, name):
    return None

def do_diskcheck_sccs(node, name):
    try:
        sccs_dir = node.sccs_dir
    except AttributeError:
        if node.entry_exists_on_disk('SCCS'):
            sccs_dir = node.Dir('SCCS')
        else:
            sccs_dir = None
        node.sccs_dir = sccs_dir
    if sccs_dir:
        return sccs_dir.entry_exists_on_disk('s.'+name)
    return None

def ignore_diskcheck_sccs(node, name):
    return None

diskcheck_match = DiskChecker('match', do_diskcheck_match, ignore_diskcheck_match)
diskcheck_rcs = DiskChecker('rcs', do_diskcheck_rcs, ignore_diskcheck_rcs)
diskcheck_sccs = DiskChecker('sccs', do_diskcheck_sccs, ignore_diskcheck_sccs)

diskcheckers = [
    diskcheck_match,
    diskcheck_rcs,
    diskcheck_sccs,
]

def set_diskcheck(list):
    for dc in diskcheckers:
        dc.set(list)

def diskcheck_types():
    return map(lambda dc: dc.type, diskcheckers)



class EntryProxy(SCons.Util.Proxy):
    def __get_abspath(self):
        entry = self.get()
        return SCons.Subst.SpecialAttrWrapper(entry.get_abspath(),
                                             entry.name + "_abspath")

    def __get_filebase(self):
        name = self.get().name
        return SCons.Subst.SpecialAttrWrapper(SCons.Util.splitext(name)[0],
                                             name + "_filebase")

    def __get_suffix(self):
        name = self.get().name
        return SCons.Subst.SpecialAttrWrapper(SCons.Util.splitext(name)[1],
                                             name + "_suffix")

    def __get_file(self):
        name = self.get().name
        return SCons.Subst.SpecialAttrWrapper(name, name + "_file")

    def __get_base_path(self):
        """Return the file's directory and file name, with the
        suffix stripped."""
        entry = self.get()
        return SCons.Subst.SpecialAttrWrapper(SCons.Util.splitext(entry.get_path())[0],
                                             entry.name + "_base")

    def __get_posix_path(self):
        """Return the path with / as the path separator,
        regardless of platform."""
        if os.sep == '/':
            return self
        else:
            entry = self.get()
            r = string.replace(entry.get_path(), os.sep, '/')
            return SCons.Subst.SpecialAttrWrapper(r, entry.name + "_posix")

    def __get_windows_path(self):
        """Return the path with \ as the path separator,
        regardless of platform."""
        if os.sep == '\\':
            return self
        else:
            entry = self.get()
            r = string.replace(entry.get_path(), os.sep, '\\')
            return SCons.Subst.SpecialAttrWrapper(r, entry.name + "_windows")

    def __get_srcnode(self):
        return EntryProxy(self.get().srcnode())

    def __get_srcdir(self):
        """Returns the directory containing the source node linked to this
        node via BuildDir(), or the directory of this node if not linked."""
        return EntryProxy(self.get().srcnode().dir)

    def __get_rsrcnode(self):
        return EntryProxy(self.get().srcnode().rfile())

    def __get_rsrcdir(self):
        """Returns the directory containing the source node linked to this
        node via BuildDir(), or the directory of this node if not linked."""
        return EntryProxy(self.get().srcnode().rfile().dir)

    def __get_dir(self):
        return EntryProxy(self.get().dir)

    dictSpecialAttrs = { "base"     : __get_base_path,
                         "posix"    : __get_posix_path,
                         "windows"  : __get_windows_path,
                         "win32"    : __get_windows_path,
                         "srcpath"  : __get_srcnode,
                         "srcdir"   : __get_srcdir,
                         "dir"      : __get_dir,
                         "abspath"  : __get_abspath,
                         "filebase" : __get_filebase,
                         "suffix"   : __get_suffix,
                         "file"     : __get_file,
                         "rsrcpath" : __get_rsrcnode,
                         "rsrcdir"  : __get_rsrcdir,
                       }

    def __getattr__(self, name):
        # This is how we implement the "special" attributes
        # such as base, posix, srcdir, etc.
        try:
            attr_function = self.dictSpecialAttrs[name]
        except KeyError:
            try:
                attr = SCons.Util.Proxy.__getattr__(self, name)
            except AttributeError:
                entry = self.get()
                classname = string.split(str(entry.__class__), '.')[-1]
                if classname[-2:] == "'>":
                    # new-style classes report their name as:
                    #   "<class 'something'>"
                    # instead of the classic classes:
                    #   "something"
                    classname = classname[:-2]
                raise AttributeError, "%s instance '%s' has no attribute '%s'" % (classname, entry.name, name)
            return attr
        else:
            return attr_function(self)

class Base(SCons.Node.Node):
    """A generic class for file system entries.  This class is for
    when we don't know yet whether the entry being looked up is a file
    or a directory.  Instances of this class can morph into either
    Dir or File objects by a later, more precise lookup.

    Note: this class does not define __cmp__ and __hash__ for
    efficiency reasons.  SCons does a lot of comparing of
    Node.FS.{Base,Entry,File,Dir} objects, so those operations must be
    as fast as possible, which means we want to use Python's built-in
    object identity comparisons.
    """

    memoizer_counters = []

    def __init__(self, name, directory, fs):
        """Initialize a generic Node.FS.Base object.

        Call the superclass initialization, take care of setting up
        our relative and absolute paths, identify our parent
        directory, and indicate that this node should use
        signatures."""
        if __debug__: logInstanceCreation(self, 'Node.FS.Base')
        SCons.Node.Node.__init__(self)

        self.name = name
        self.suffix = SCons.Util.splitext(name)[1]
        self.fs = fs

        assert directory, "A directory must be provided"

        self.abspath = directory.entry_abspath(name)
        self.labspath = directory.entry_labspath(name)
        if directory.path == '.':
            self.path = name
        else:
            self.path = directory.entry_path(name)
        if directory.tpath == '.':
            self.tpath = name
        else:
            self.tpath = directory.entry_tpath(name)
        self.path_elements = directory.path_elements + [self]

        self.dir = directory
        self.cwd = None # will hold the SConscript directory for target nodes
        self.duplicate = directory.duplicate

    def must_be_same(self, klass):
        """
        This node, which already existed, is being looked up as the
        specified klass.  Raise an exception if it isn't.
        """
        if self.__class__ is klass or klass is Entry:
            return
        raise TypeError, "Tried to lookup %s '%s' as a %s." %\
              (self.__class__.__name__, self.path, klass.__name__)

    def get_dir(self):
        return self.dir

    def get_suffix(self):
        return self.suffix

    def rfile(self):
        return self

    def __str__(self):
        """A Node.FS.Base object's string representation is its path
        name."""
        global Save_Strings
        if Save_Strings:
            return self._save_str()
        return self._get_str()

    memoizer_counters.append(SCons.Memoize.CountValue('_save_str'))

    def _save_str(self):
        try:
            return self._memo['_save_str']
        except KeyError:
            pass
        result = self._get_str()
        self._memo['_save_str'] = result
        return result

    def _get_str(self):
        if self.duplicate or self.is_derived():
            return self.get_path()
        return self.srcnode().get_path()

    rstr = __str__

    memoizer_counters.append(SCons.Memoize.CountValue('stat'))

    def stat(self):
        try: return self._memo['stat']
        except KeyError: pass
        try: result = self.fs.stat(self.abspath)
        except os.error: result = None
        self._memo['stat'] = result
        return result

    def exists(self):
        return not self.stat() is None

    def rexists(self):
        return self.rfile().exists()

    def getmtime(self):
        st = self.stat()
        if st: return st[stat.ST_MTIME]
        else: return None

    def getsize(self):
        st = self.stat()
        if st: return st[stat.ST_SIZE]
        else: return None

    def isdir(self):
        st = self.stat()
        return not st is None and stat.S_ISDIR(st[stat.ST_MODE])

    def isfile(self):
        st = self.stat()
        return not st is None and stat.S_ISREG(st[stat.ST_MODE])

    if hasattr(os, 'symlink'):
        def islink(self):
            try: st = self.fs.lstat(self.abspath)
            except os.error: return 0
            return stat.S_ISLNK(st[stat.ST_MODE])
    else:
        def islink(self):
            return 0                    # no symlinks

    def is_under(self, dir):
        if self is dir:
            return 1
        else:
            return self.dir.is_under(dir)

    def set_local(self):
        self._local = 1

    def srcnode(self):
        """If this node is in a build path, return the node
        corresponding to its source file.  Otherwise, return
        ourself.
        """
        dir=self.dir
        name=self.name
        while dir:
            if dir.srcdir:
                srcnode = dir.srcdir.Entry(name)
                srcnode.must_be_same(self.__class__)
                return srcnode
            name = dir.name + os.sep + name
            dir = dir.up()
        return self

    def get_path(self, dir=None):
        """Return path relative to the current working directory of the
        Node.FS.Base object that owns us."""
        if not dir:
            dir = self.fs.getcwd()
        if self == dir:
            return '.'
        path_elems = self.path_elements
        try: i = path_elems.index(dir)
        except ValueError: pass
        else: path_elems = path_elems[i+1:]
        path_elems = map(lambda n: n.name, path_elems)
        return string.join(path_elems, os.sep)

    def set_src_builder(self, builder):
        """Set the source code builder for this node."""
        self.sbuilder = builder
        if not self.has_builder():
            self.builder_set(builder)

    def src_builder(self):
        """Fetch the source code builder for this node.

        If there isn't one, we cache the source code builder specified
        for the directory (which in turn will cache the value from its
        parent directory, and so on up to the file system root).
        """
        try:
            scb = self.sbuilder
        except AttributeError:
            scb = self.dir.src_builder()
            self.sbuilder = scb
        return scb

    def get_abspath(self):
        """Get the absolute path of the file."""
        return self.abspath

    def for_signature(self):
        # Return just our name.  Even an absolute path would not work,
        # because that can change thanks to symlinks or remapped network
        # paths.
        return self.name

    def get_subst_proxy(self):
        try:
            return self._proxy
        except AttributeError:
            ret = EntryProxy(self)
            self._proxy = ret
            return ret

    def target_from_source(self, prefix, suffix, splitext=SCons.Util.splitext):
        """

	Generates a target entry that corresponds to this entry (usually
        a source file) with the specified prefix and suffix.

        Note that this method can be overridden dynamically for generated
        files that need different behavior.  See Tool/swig.py for
        an example.
        """
        return self.dir.Entry(prefix + splitext(self.name)[0] + suffix)

    def _Rfindalldirs_key(self, pathlist):
        return pathlist

    memoizer_counters.append(SCons.Memoize.CountDict('Rfindalldirs', _Rfindalldirs_key))

    def Rfindalldirs(self, pathlist):
        """
        Return all of the directories for a given path list, including
        corresponding "backing" directories in any repositories.

        The Node lookups are relative to this Node (typically a
        directory), so memoizing result saves cycles from looking
        up the same path for each target in a given directory.
        """
        try:
            memo_dict = self._memo['Rfindalldirs']
        except KeyError:
            memo_dict = {}
            self._memo['Rfindalldirs'] = memo_dict
        else:
            try:
                return memo_dict[pathlist]
            except KeyError:
                pass

        create_dir_relative_to_self = self.Dir
        result = []
        for path in pathlist:
            if isinstance(path, SCons.Node.Node):
                result.append(path)
            else:
                dir = create_dir_relative_to_self(path)
                result.extend(dir.get_all_rdirs())

        memo_dict[pathlist] = result

        return result

    def RDirs(self, pathlist):
        """Search for a list of directories in the Repository list."""
        cwd = self.cwd or self.fs._cwd
        return cwd.Rfindalldirs(pathlist)

    memoizer_counters.append(SCons.Memoize.CountValue('rentry'))

    def rentry(self):
        try:
            return self._memo['rentry']
        except KeyError:
            pass
        result = self
        if not self.exists():
            norm_name = _my_normcase(self.name)
            for dir in self.dir.get_all_rdirs():
                try:
                    node = dir.entries[norm_name]
                except KeyError:
                    if dir.entry_exists_on_disk(self.name):
                        result = dir.Entry(self.name)
                        break
        self._memo['rentry'] = result
        return result

class Entry(Base):
    """This is the class for generic Node.FS entries--that is, things
    that could be a File or a Dir, but we're just not sure yet.
    Consequently, the methods in this class really exist just to
    transform their associated object into the right class when the
    time comes, and then call the same-named method in the transformed
    class."""

    def diskcheck_match(self):
        pass

    def disambiguate(self, must_exist=None):
        """
        """
        if self.isdir():
            self.__class__ = Dir
            self._morph()
        elif self.isfile():
            self.__class__ = File
            self._morph()
            self.clear()
        else:
            # There was nothing on-disk at this location, so look in
            # the src directory.
            #
            # We can't just use self.srcnode() straight away because
            # that would create an actual Node for this file in the src
            # directory, and there might not be one.  Instead, use the
            # dir_on_disk() method to see if there's something on-disk
            # with that name, in which case we can go ahead and call
            # self.srcnode() to create the right type of entry.
            srcdir = self.dir.srcnode()
            if srcdir != self.dir and \
               srcdir.entry_exists_on_disk(self.name) and \
               self.srcnode().isdir():
                self.__class__ = Dir
                self._morph()
            elif must_exist:
                msg = "No such file or directory: '%s'" % self.abspath
                raise SCons.Errors.UserError, msg
            else:
                self.__class__ = File
                self._morph()
                self.clear()
        return self

    def rfile(self):
        """We're a generic Entry, but the caller is actually looking for
        a File at this point, so morph into one."""
        self.__class__ = File
        self._morph()
        self.clear()
        return File.rfile(self)

    def scanner_key(self):
        return self.get_suffix()

    def get_contents(self):
        """Fetch the contents of the entry.

        Since this should return the real contents from the file
        system, we check to see into what sort of subclass we should
        morph this Entry."""
        try:
            self = self.disambiguate(must_exist=1)
        except SCons.Errors.UserError:
            # There was nothing on disk with which to disambiguate
            # this entry.  Leave it as an Entry, but return a null
            # string so calls to get_contents() in emitters and the
            # like (e.g. in qt.py) don't have to disambiguate by hand
            # or catch the exception.
            return ''
        else:
            return self.get_contents()

    def must_be_same(self, klass):
        """Called to make sure a Node is a Dir.  Since we're an
        Entry, we can morph into one."""
        if not self.__class__ is klass:
            self.__class__ = klass
            self._morph()
            self.clear

    # The following methods can get called before the Taskmaster has
    # had a chance to call disambiguate() directly to see if this Entry
    # should really be a Dir or a File.  We therefore use these to call
    # disambiguate() transparently (from our caller's point of view).
    #
    # Right now, this minimal set of methods has been derived by just
    # looking at some of the methods that will obviously be called early
    # in any of the various Taskmasters' calling sequences, and then
    # empirically figuring out which additional methods are necessary
    # to make various tests pass.

    def exists(self):
        """Return if the Entry exists.  Check the file system to see
        what we should turn into first.  Assume a file if there's no
        directory."""
        return self.disambiguate().exists()

#    def rel_path(self, other):
#        d = self.disambiguate()
#        if d.__class__ == Entry:
#            raise "rel_path() could not disambiguate File/Dir"
#        return d.rel_path(other)

    def new_ninfo(self):
        return self.disambiguate().new_ninfo()

    def changed_since_last_build(self, target, prev_ni):
        return self.disambiguate().changed_since_last_build(target, prev_ni)

# This is for later so we can differentiate between Entry the class and Entry
# the method of the FS class.
_classEntry = Entry


class LocalFS:

    if SCons.Memoize.use_memoizer:
        __metaclass__ = SCons.Memoize.Memoized_Metaclass

    # This class implements an abstraction layer for operations involving
    # a local file system.  Essentially, this wraps any function in
    # the os, os.path or shutil modules that we use to actually go do
    # anything with or to the local file system.
    #
    # Note that there's a very good chance we'll refactor this part of
    # the architecture in some way as we really implement the interface(s)
    # for remote file system Nodes.  For example, the right architecture
    # might be to have this be a subclass instead of a base class.
    # Nevertheless, we're using this as a first step in that direction.
    #
    # We're not using chdir() yet because the calling subclass method
    # needs to use os.chdir() directly to avoid recursion.  Will we
    # really need this one?
    #def chdir(self, path):
    #    return os.chdir(path)
    def chmod(self, path, mode):
        return os.chmod(path, mode)
    def copy2(self, src, dst):
        return shutil.copy2(src, dst)
    def exists(self, path):
        return os.path.exists(path)
    def getmtime(self, path):
        return os.path.getmtime(path)
    def getsize(self, path):
        return os.path.getsize(path)
    def isdir(self, path):
        return os.path.isdir(path)
    def isfile(self, path):
        return os.path.isfile(path)
    def link(self, src, dst):
        return os.link(src, dst)
    def lstat(self, path):
        return os.lstat(path)
    def listdir(self, path):
        return os.listdir(path)
    def makedirs(self, path):
        return os.makedirs(path)
    def mkdir(self, path):
        return os.mkdir(path)
    def rename(self, old, new):
        return os.rename(old, new)
    def stat(self, path):
        return os.stat(path)
    def symlink(self, src, dst):
        return os.symlink(src, dst)
    def open(self, path):
        return open(path)
    def unlink(self, path):
        return os.unlink(path)

    if hasattr(os, 'symlink'):
        def islink(self, path):
            return os.path.islink(path)
    else:
        def islink(self, path):
            return 0                    # no symlinks

    if hasattr(os, 'readlink'):
        def readlink(self, file):
            return os.readlink(file)
    else:
        def readlink(self, file):
            return ''


#class RemoteFS:
#    # Skeleton for the obvious methods we might need from the
#    # abstraction layer for a remote filesystem.
#    def upload(self, local_src, remote_dst):
#        pass
#    def download(self, remote_src, local_dst):
#        pass


class FS(LocalFS):

    memoizer_counters = []

    def __init__(self, path = None):
        """Initialize the Node.FS subsystem.

        The supplied path is the top of the source tree, where we
        expect to find the top-level build file.  If no path is
        supplied, the current directory is the default.

        The path argument must be a valid absolute path.
        """
        if __debug__: logInstanceCreation(self, 'Node.FS')

        self._memo = {}

        self.Root = {}
        self.SConstruct_dir = None
        self.max_drift = default_max_drift

        self.Top = None
        if path is None:
            self.pathTop = os.getcwd()
        else:
            self.pathTop = path
        self.defaultDrive = _my_normcase(os.path.splitdrive(self.pathTop)[0])

        self.Top = self.Dir(self.pathTop)
        self.Top.path = '.'
        self.Top.tpath = '.'
        self._cwd = self.Top

        DirNodeInfo.top = self.Top
        FileNodeInfo.top = self.Top
    
    def set_SConstruct_dir(self, dir):
        self.SConstruct_dir = dir

    def get_max_drift(self):
        return self.max_drift

    def set_max_drift(self, max_drift):
        self.max_drift = max_drift

    def getcwd(self):
        return self._cwd

    def chdir(self, dir, change_os_dir=0):
        """Change the current working directory for lookups.
        If change_os_dir is true, we will also change the "real" cwd
        to match.
        """
        curr=self._cwd
        try:
            if not dir is None:
                self._cwd = dir
                if change_os_dir:
                    os.chdir(dir.abspath)
        except OSError:
            self._cwd = curr
            raise

    def get_root(self, drive):
        """
        Returns the root directory for the specified drive, creating
        it if necessary.
        """
        drive = _my_normcase(drive)
        try:
            return self.Root[drive]
        except KeyError:
            root = RootDir(drive, self)
            self.Root[drive] = root
            if not drive:
                self.Root[self.defaultDrive] = root
            elif drive == self.defaultDrive:
                self.Root[''] = root
            return root

    def _lookup(self, p, directory, fsclass, create=1):
        """
        The generic entry point for Node lookup with user-supplied data.

        This translates arbitrary input into a canonical Node.FS object
        of the specified fsclass.  The general approach for strings is
        to turn it into a normalized absolute path and then call the
        root directory's lookup_abs() method for the heavy lifting.

        If the path name begins with '#', it is unconditionally
        interpreted relative to the top-level directory of this FS.

        If the path name is relative, then the path is looked up relative
        to the specified directory, or the current directory (self._cwd,
        typically the SConscript directory) if the specified directory
        is None.
        """
        if isinstance(p, Base):
            # It's already a Node.FS object.  Make sure it's the right
            # class and return.
            p.must_be_same(fsclass)
            return p
        # str(p) in case it's something like a proxy object
        p = str(p)
        drive, p = os.path.splitdrive(p)
        if drive and not p:
            # A drive letter without a path...
            p = os.sep
            root = self.get_root(drive)
        elif os.path.isabs(p):
            # An absolute path...
            p = os.path.normpath(p)
            root = self.get_root(drive)
        else:
            if p[0:1] == '#':
                # A top-relative path...
                directory = self.Top
                offset = 1
                if p[1:2] in(os.sep, '/'):
                    offset = 2
                p = p[offset:]
            else:
                # A relative path...
                if not directory:
                    # ...to the current (SConscript) directory.
                    directory = self._cwd
                elif not isinstance(directory, Dir):
                    # ...to the specified directory.
                    directory = self.Dir(directory)
            p = os.path.normpath(directory.labspath + '/' + p)
            root = directory.root
        if os.sep != '/':
            p = string.replace(p, os.sep, '/')
        return root._lookup_abs(p, fsclass, create)

    def Entry(self, name, directory = None, create = 1):
        """Lookup or create a generic Entry node with the specified name.
        If the name is a relative path (begins with ./, ../, or a file
        name), then it is looked up relative to the supplied directory
        node, or to the top level directory of the FS (supplied at
        construction time) if no directory is supplied.
        """
        return self._lookup(name, directory, Entry, create)

    def File(self, name, directory = None, create = 1):
        """Lookup or create a File node with the specified name.  If
        the name is a relative path (begins with ./, ../, or a file name),
        then it is looked up relative to the supplied directory node,
        or to the top level directory of the FS (supplied at construction
        time) if no directory is supplied.

        This method will raise TypeError if a directory is found at the
        specified path.
        """
        return self._lookup(name, directory, File, create)

    def Dir(self, name, directory = None, create = 1):
        """Lookup or create a Dir node with the specified name.  If
        the name is a relative path (begins with ./, ../, or a file name),
        then it is looked up relative to the supplied directory node,
        or to the top level directory of the FS (supplied at construction
        time) if no directory is supplied.

        This method will raise TypeError if a normal file is found at the
        specified path.
        """
        return self._lookup(name, directory, Dir, create)

    def BuildDir(self, build_dir, src_dir, duplicate=1):
        """Link the supplied build directory to the source directory
        for purposes of building files."""

        if not isinstance(src_dir, SCons.Node.Node):
            src_dir = self.Dir(src_dir)
        if not isinstance(build_dir, SCons.Node.Node):
            build_dir = self.Dir(build_dir)
        if src_dir.is_under(build_dir):
            raise SCons.Errors.UserError, "Source directory cannot be under build directory."
        if build_dir.srcdir:
            if build_dir.srcdir == src_dir:
                return # We already did this.
            raise SCons.Errors.UserError, "'%s' already has a source directory: '%s'."%(build_dir, build_dir.srcdir)
        build_dir.link(src_dir, duplicate)

    def Repository(self, *dirs):
        """Specify Repository directories to search."""
        for d in dirs:
            if not isinstance(d, SCons.Node.Node):
                d = self.Dir(d)
            self.Top.addRepository(d)

    def build_dir_target_climb(self, orig, dir, tail):
        """Create targets in corresponding build directories

        Climb the directory tree, and look up path names
        relative to any linked build directories we find.

        Even though this loops and walks up the tree, we don't memoize
        the return value because this is really only used to process
        the command-line targets.
        """
        targets = []
        message = None
        fmt = "building associated BuildDir targets: %s"
        start_dir = dir
        while dir:
            for bd in dir.build_dirs:
                if start_dir.is_under(bd):
                    # If already in the build-dir location, don't reflect
                    return [orig], fmt % str(orig)
                p = apply(os.path.join, [bd.path] + tail)
                targets.append(self.Entry(p))
            tail = [dir.name] + tail
            dir = dir.up()
        if targets:
            message = fmt % string.join(map(str, targets))
        return targets, message

class DirNodeInfo(SCons.Node.NodeInfoBase):
    # This should get reset by the FS initialization.
    current_version_id = 1

    top = None

    def str_to_node(self, s):
        top = self.top
        if os.path.isabs(s):
            n = top.fs._lookup(s, top, Entry)
        else:
            s = top.labspath + '/' + s
            n = top.root._lookup_abs(s, Entry)
        return n

class DirBuildInfo(SCons.Node.BuildInfoBase):
    current_version_id = 1

class Dir(Base):
    """A class for directories in a file system.
    """

    memoizer_counters = []

    NodeInfo = DirNodeInfo
    BuildInfo = DirBuildInfo

    def __init__(self, name, directory, fs):
        if __debug__: logInstanceCreation(self, 'Node.FS.Dir')
        Base.__init__(self, name, directory, fs)
        self._morph()

    def _morph(self):
        """Turn a file system Node (either a freshly initialized directory
        object or a separate Entry object) into a proper directory object.

        Set up this directory's entries and hook it into the file
        system tree.  Specify that directories (this Node) don't use
        signatures for calculating whether they're current.
        """

        self.repositories = []
        self.srcdir = None

        self.entries = {}
        self.entries['.'] = self
        self.entries['..'] = self.dir
        self.cwd = self
        self.searched = 0
        self._sconsign = None
        self.build_dirs = []
        self.root = self.dir.root

        # Don't just reset the executor, replace its action list,
        # because it might have some pre-or post-actions that need to
        # be preserved.
        self.builder = get_MkdirBuilder()
        self.get_executor().set_action_list(self.builder.action)

    def diskcheck_match(self):
        diskcheck_match(self, self.isfile,
                        "File %s found where directory expected.")

    def __clearRepositoryCache(self, duplicate=None):
        """Called when we change the repository(ies) for a directory.
        This clears any cached information that is invalidated by changing
        the repository."""

        for node in self.entries.values():
            if node != self.dir:
                if node != self and isinstance(node, Dir):
                    node.__clearRepositoryCache(duplicate)
                else:
                    node.clear()
                    try:
                        del node._srcreps
                    except AttributeError:
                        pass
                    if duplicate != None:
                        node.duplicate=duplicate

    def __resetDuplicate(self, node):
        if node != self:
            node.duplicate = node.get_dir().duplicate

    def Entry(self, name):
        """
        Looks up or creates an entry node named 'name' relative to
        this directory.
        """
        return self.fs.Entry(name, self)

    def Dir(self, name):
        """
        Looks up or creates a directory node named 'name' relative to
        this directory.
        """
        dir = self.fs.Dir(name, self)
        return dir

    def File(self, name):
        """
        Looks up or creates a file node named 'name' relative to
        this directory.
        """
        return self.fs.File(name, self)

    def _lookup_rel(self, name, klass, create=1):
        """
        Looks up a *normalized* relative path name, relative to this
        directory.

        This method is intended for use by internal lookups with
        already-normalized path data.  For general-purpose lookups,
        use the Entry(), Dir() and File() methods above.

        This method does *no* input checking and will die or give
        incorrect results if it's passed a non-normalized path name (e.g.,
        a path containing '..'), an absolute path name, a top-relative
        ('#foo') path name, or any kind of object.
        """
        name = self.labspath + '/' + name
        return self.root._lookup_abs(name, klass, create)

    def link(self, srcdir, duplicate):
        """Set this directory as the build directory for the
        supplied source directory."""
        self.srcdir = srcdir
        self.duplicate = duplicate
        self.__clearRepositoryCache(duplicate)
        srcdir.build_dirs.append(self)

    def getRepositories(self):
        """Returns a list of repositories for this directory.
        """
        if self.srcdir and not self.duplicate:
            return self.srcdir.get_all_rdirs() + self.repositories
        return self.repositories

    memoizer_counters.append(SCons.Memoize.CountValue('get_all_rdirs'))

    def get_all_rdirs(self):
        try:
            return self._memo['get_all_rdirs']
        except KeyError:
            pass

        result = [self]
        fname = '.'
        dir = self
        while dir:
            for rep in dir.getRepositories():
                result.append(rep.Dir(fname))
            fname = dir.name + os.sep + fname
            dir = dir.up()

        self._memo['get_all_rdirs'] = result

        return result

    def addRepository(self, dir):
        if dir != self and not dir in self.repositories:
            self.repositories.append(dir)
            dir.tpath = '.'
            self.__clearRepositoryCache()

    def up(self):
        return self.entries['..']

# This complicated method, which constructs relative paths between
# arbitrary Node.FS objects, is no longer used.  It was introduced to
# store dependency paths in .sconsign files relative to the target, but
# that ended up being significantly inefficient.  We're leaving the code
# here, commented out, because it would be too easy for someone to decide
# to re-invent this wheel in the future (if it becomes necessary) because
# they didn't know this code was buried in some source-code change from
# the distant past...
#
#    def _rel_path_key(self, other):
#        return str(other)
#
#    memoizer_counters.append(SCons.Memoize.CountDict('rel_path', _rel_path_key))
#
#    def rel_path(self, other):
#        """Return a path to "other" relative to this directory.
#        """
#        try:
#            memo_dict = self._memo['rel_path']
#        except KeyError:
#            memo_dict = {}
#            self._memo['rel_path'] = memo_dict
#        else:
#            try:
#                return memo_dict[other]
#            except KeyError:
#                pass
#
#        if self is other:
#
#            result = '.'
#
#        elif not other in self.path_elements:
#
#            try:
#                other_dir = other.get_dir()
#            except AttributeError:
#                result = str(other)
#            else:
#                if other_dir is None:
#                    result = other.name
#                else:
#                    dir_rel_path = self.rel_path(other_dir)
#                    if dir_rel_path == '.':
#                        result = other.name
#                    else:
#                        result = dir_rel_path + os.sep + other.name
#
#        else:
#
#            i = self.path_elements.index(other) + 1
#
#            path_elems = ['..'] * (len(self.path_elements) - i) \
#                         + map(lambda n: n.name, other.path_elements[i:])
#             
#            result = string.join(path_elems, os.sep)
#
#        memo_dict[other] = result
#
#        return result

    def get_env_scanner(self, env, kw={}):
        import SCons.Defaults
        return SCons.Defaults.DirEntryScanner

    def get_target_scanner(self):
        import SCons.Defaults
        return SCons.Defaults.DirEntryScanner

    def get_found_includes(self, env, scanner, path):
        """Return this directory's implicit dependencies.

        We don't bother caching the results because the scan typically
        shouldn't be requested more than once (as opposed to scanning
        .h file contents, which can be requested as many times as the
        files is #included by other files).
        """
        if not scanner:
            return []
        # Clear cached info for this Dir.  If we already visited this
        # directory on our walk down the tree (because we didn't know at
        # that point it was being used as the source for another Node)
        # then we may have calculated build signature before realizing
        # we had to scan the disk.  Now that we have to, though, we need
        # to invalidate the old calculated signature so that any node
        # dependent on our directory structure gets one that includes
        # info about everything on disk.
        self.clear()
        return scanner(self, env, path)

    #
    # Taskmaster interface subsystem
    #

    def prepare(self):
        pass

    def build(self, **kw):
        """A null "builder" for directories."""
        global MkdirBuilder
        if not self.builder is MkdirBuilder:
            apply(SCons.Node.Node.build, [self,], kw)

    #
    #
    #

    def _create(self):
        """Create this directory, silently and without worrying about
        whether the builder is the default or not."""
        listDirs = []
        parent = self
        while parent:
            if parent.exists():
                break
            listDirs.append(parent)
            p = parent.up()
            if p is None:
                raise SCons.Errors.StopError, parent.path
            parent = p
        listDirs.reverse()
        for dirnode in listDirs:
            try:
                # Don't call dirnode.build(), call the base Node method
                # directly because we definitely *must* create this
                # directory.  The dirnode.build() method will suppress
                # the build if it's the default builder.
                SCons.Node.Node.build(dirnode)
                dirnode.get_executor().nullify()
                # The build() action may or may not have actually
                # created the directory, depending on whether the -n
                # option was used or not.  Delete the _exists and
                # _rexists attributes so they can be reevaluated.
                dirnode.clear()
            except OSError:
                pass

    def multiple_side_effect_has_builder(self):
        global MkdirBuilder
        return not self.builder is MkdirBuilder and self.has_builder()

    def alter_targets(self):
        """Return any corresponding targets in a build directory.
        """
        return self.fs.build_dir_target_climb(self, self, [])

    def scanner_key(self):
        """A directory does not get scanned."""
        return None

    def get_contents(self):
        """Return aggregate contents of all our children."""
        contents = map(lambda n: n.get_contents(), self.children())
        return  string.join(contents, '')

    def do_duplicate(self, src):
        pass

    changed_since_last_build = SCons.Node.Node.state_has_changed

    def is_up_to_date(self):
        """If any child is not up-to-date, then this directory isn't,
        either."""
        if not self.builder is MkdirBuilder and not self.exists():
            return 0
        up_to_date = SCons.Node.up_to_date
        for kid in self.children():
            if kid.get_state() > up_to_date:
                return 0
        return 1

    def rdir(self):
        if not self.exists():
            norm_name = _my_normcase(self.name)
            for dir in self.dir.get_all_rdirs():
                try: node = dir.entries[norm_name]
                except KeyError: node = dir.dir_on_disk(self.name)
                if node and node.exists() and \
                    (isinstance(dir, Dir) or isinstance(dir, Entry)):
                        return node
        return self

    def sconsign(self):
        """Return the .sconsign file info for this directory,
        creating it first if necessary."""
        if not self._sconsign:
            import SCons.SConsign
            self._sconsign = SCons.SConsign.ForDirectory(self)
        return self._sconsign

    def srcnode(self):
        """Dir has a special need for srcnode()...if we
        have a srcdir attribute set, then that *is* our srcnode."""
        if self.srcdir:
            return self.srcdir
        return Base.srcnode(self)

    def get_timestamp(self):
        """Return the latest timestamp from among our children"""
        stamp = 0
        for kid in self.children():
            if kid.get_timestamp() > stamp:
                stamp = kid.get_timestamp()
        return stamp

    def entry_abspath(self, name):
        return self.abspath + os.sep + name

    def entry_labspath(self, name):
        return self.labspath + '/' + name

    def entry_path(self, name):
        return self.path + os.sep + name

    def entry_tpath(self, name):
        return self.tpath + os.sep + name

    def entry_exists_on_disk(self, name):
        try:
            d = self.on_disk_entries
        except AttributeError:
            d = {}
            try:
                entries = os.listdir(self.abspath)
            except OSError:
                pass
            else:
                for entry in map(_my_normcase, entries):
                    d[entry] = 1
            self.on_disk_entries = d
        return d.has_key(_my_normcase(name))

    memoizer_counters.append(SCons.Memoize.CountValue('srcdir_list'))

    def srcdir_list(self):
        try:
            return self._memo['srcdir_list']
        except KeyError:
            pass

        result = []

        dirname = '.'
        dir = self
        while dir:
            if dir.srcdir:
                d = dir.srcdir.Dir(dirname)
                if d.is_under(dir):
                    # Shouldn't source from something in the build path:
                    # build_dir is probably under src_dir, in which case
                    # we are reflecting.
                    break
                result.append(d)
            dirname = dir.name + os.sep + dirname
            dir = dir.up()

        self._memo['srcdir_list'] = result

        return result

    def srcdir_duplicate(self, name):
        for dir in self.srcdir_list():
            if dir.entry_exists_on_disk(name):
                srcnode = dir.Entry(name).disambiguate()
                if self.duplicate:
                    node = self.Entry(name).disambiguate()
                    node.do_duplicate(srcnode)
                    return node
                else:
                    return srcnode
        return None

    def _srcdir_find_file_key(self, filename):
        return filename

    memoizer_counters.append(SCons.Memoize.CountDict('srcdir_find_file', _srcdir_find_file_key))

    def srcdir_find_file(self, filename):
        try:
            memo_dict = self._memo['srcdir_find_file']
        except KeyError:
            memo_dict = {}
            self._memo['srcdir_find_file'] = memo_dict
        else:
            try:
                return memo_dict[filename]
            except KeyError:
                pass

        def func(node):
            if (isinstance(node, File) or isinstance(node, Entry)) and \
               (node.is_derived() or node.exists()):
                    return node
            return None

        norm_name = _my_normcase(filename)

        for rdir in self.get_all_rdirs():
            try: node = rdir.entries[norm_name]
            except KeyError: node = rdir.file_on_disk(filename)
            else: node = func(node)
            if node:
                result = (node, self)
                memo_dict[filename] = result
                return result

        for srcdir in self.srcdir_list():
            for rdir in srcdir.get_all_rdirs():
                try: node = rdir.entries[norm_name]
                except KeyError: node = rdir.file_on_disk(filename)
                else: node = func(node)
                if node:
                    result = (File(filename, self, self.fs), srcdir)
                    memo_dict[filename] = result
                    return result

        result = (None, None)
        memo_dict[filename] = result
        return result

    def dir_on_disk(self, name):
        if self.entry_exists_on_disk(name):
            try: return self.Dir(name)
            except TypeError: pass
        return None

    def file_on_disk(self, name):
        if self.entry_exists_on_disk(name) or \
           diskcheck_rcs(self, name) or \
           diskcheck_sccs(self, name):
            try: return self.File(name)
            except TypeError: pass
        node = self.srcdir_duplicate(name)
        if isinstance(node, Dir):
            node = None
        return node

    def walk(self, func, arg):
        """
        Walk this directory tree by calling the specified function
        for each directory in the tree.

        This behaves like the os.path.walk() function, but for in-memory
        Node.FS.Dir objects.  The function takes the same arguments as
        the functions passed to os.path.walk():

                func(arg, dirname, fnames)

        Except that "dirname" will actually be the directory *Node*,
        not the string.  The '.' and '..' entries are excluded from
        fnames.  The fnames list may be modified in-place to filter the
        subdirectories visited or otherwise impose a specific order.
        The "arg" argument is always passed to func() and may be used
        in any way (or ignored, passing None is common).
        """
        entries = self.entries
        names = entries.keys()
        names.remove('.')
        names.remove('..')
        func(arg, self, names)
        select_dirs = lambda n, e=entries: isinstance(e[n], Dir)
        for dirname in filter(select_dirs, names):
            entries[dirname].walk(func, arg)

class RootDir(Dir):
    """A class for the root directory of a file system.

    This is the same as a Dir class, except that the path separator
    ('/' or '\\') is actually part of the name, so we don't need to
    add a separator when creating the path names of entries within
    this directory.
    """
    def __init__(self, name, fs):
        if __debug__: logInstanceCreation(self, 'Node.FS.RootDir')
        # We're going to be our own parent directory (".." entry and .dir
        # attribute) so we have to set up some values so Base.__init__()
        # won't gag won't it calls some of our methods.
        self.abspath = ''
        self.labspath = ''
        self.path = ''
        self.tpath = ''
        self.path_elements = []
        self.duplicate = 0
        self.root = self
        Base.__init__(self, name, self, fs)

        # Now set our paths to what we really want them to be: the
        # initial drive letter (the name) plus the directory separator,
        # except for the "lookup abspath," which does not have the
        # drive letter.
        self.abspath = name + os.sep
        self.labspath = '/'
        self.path = name + os.sep
        self.tpath = name + os.sep
        self._morph()

        self._lookupDict = {}

        # The // and os.sep + os.sep entries are necessary because
        # os.path.normpath() seems to preserve double slashes at the
        # beginning of a path (presumably for UNC path names), but
        # collapses triple slashes to a single slash.
        self._lookupDict['/'] = self
        self._lookupDict['//'] = self
        self._lookupDict[os.sep] = self
        self._lookupDict[os.sep + os.sep] = self

    def must_be_same(self, klass):
        if klass is Dir:
            return
        Base.must_be_same(self, klass)

    def _lookup_abs(self, p, klass, create=1):
        """
        Fast (?) lookup of a *normalized* absolute path.

        This method is intended for use by internal lookups with
        already-normalized path data.  For general-purpose lookups,
        use the FS.Entry(), FS.Dir() or FS.File() methods.

        The caller is responsible for making sure we're passed a
        normalized absolute path; we merely let Python's dictionary look
        up and return the One True Node.FS object for the path.

        If no Node for the specified "p" doesn't already exist, and
        "create" is specified, the Node may be created after recursive
        invocation to find or create the parent directory or directories.
        """
        k = _my_normcase(p)
        try:
            result = self._lookupDict[k]
        except KeyError:
            if not create:
                raise SCons.Errors.UserError
            # There is no Node for this path name, and we're allowed
            # to create it.
            dir_name, file_name = os.path.split(p)
            dir_node = self._lookup_abs(dir_name, Dir)
            result = klass(file_name, dir_node, self.fs)
            self._lookupDict[k] = result
            dir_node.entries[_my_normcase(file_name)] = result
            dir_node.implicit = None

            # Double-check on disk (as configured) that the Node we
            # created matches whatever is out there in the real world.
            result.diskcheck_match()
        else:
            # There is already a Node for this path name.  Allow it to
            # complain if we were looking for an inappropriate type.
            result.must_be_same(klass)
        return result

    def __str__(self):
        return self.abspath

    def entry_abspath(self, name):
        return self.abspath + name

    def entry_labspath(self, name):
        return self.labspath + name

    def entry_path(self, name):
        return self.path + name

    def entry_tpath(self, name):
        return self.tpath + name

    def is_under(self, dir):
        if self is dir:
            return 1
        else:
            return 0

    def up(self):
        return None

    def get_dir(self):
        return None

    def src_builder(self):
        return _null

class FileNodeInfo(SCons.Node.NodeInfoBase):
    current_version_id = 1

    field_list = ['csig', 'timestamp', 'size']

    # This should get reset by the FS initialization.
    top = None

    def str_to_node(self, s):
        top = self.top
        if os.path.isabs(s):
            n = top.fs._lookup(s, top, Entry)
        else:
            s = top.labspath + '/' + s
            n = top.root._lookup_abs(s, Entry)
        return n

class FileBuildInfo(SCons.Node.BuildInfoBase):
    current_version_id = 1

    def convert_to_sconsign(self):
        """
        Converts this FileBuildInfo object for writing to a .sconsign file

        This replaces each Node in our various dependency lists with its
        usual string representation: relative to the top-level SConstruct
        directory, or an absolute path if it's outside.
        """
        if os.sep == '/':
            node_to_str = str
        else:
            def node_to_str(n):
                try:
                    s = n.path
                except AttributeError:
                    s = str(n)
                else:
                    s = string.replace(s, os.sep, '/')
                return s
        for attr in ['bsources', 'bdepends', 'bimplicit']:
            try:
                val = getattr(self, attr)
            except AttributeError:
                pass
            else:
                setattr(self, attr, map(node_to_str, val))
    def convert_from_sconsign(self, dir, name):
        """
        Converts a newly-read FileBuildInfo object for in-SCons use

        For normal up-to-date checking, we don't have any conversion to
        perform--but we're leaving this method here to make that clear.
        """
        pass
    def prepare_dependencies(self):
        """
        Prepares a FileBuildInfo object for explaining what changed

        The bsources, bdepends and bimplicit lists have all been
        stored on disk as paths relative to the top-level SConstruct
        directory.  Convert the strings to actual Nodes (for use by the
        --debug=explain code and --implicit-cache).
        """
        attrs = [
            ('bsources', 'bsourcesigs'),
            ('bdepends', 'bdependsigs'),
            ('bimplicit', 'bimplicitsigs'),
        ]
        for (nattr, sattr) in attrs:
            try:
                strings = getattr(self, nattr)
                nodeinfos = getattr(self, sattr)
            except AttributeError:
                pass
            else:
                nodes = []
                for s, ni in zip(strings, nodeinfos):
                    if not isinstance(s, SCons.Node.Node):
                        s = ni.str_to_node(s)
                    nodes.append(s)
                setattr(self, nattr, nodes)
    def format(self, names=0):
        result = []
        bkids = self.bsources + self.bdepends + self.bimplicit
        bkidsigs = self.bsourcesigs + self.bdependsigs + self.bimplicitsigs
        for bkid, bkidsig in zip(bkids, bkidsigs):
            result.append(str(bkid) + ': ' +
                          string.join(bkidsig.format(names=names), ' '))
        result.append('%s [%s]' % (self.bactsig, self.bact))
        return string.join(result, '\n')

class File(Base):
    """A class for files in a file system.
    """

    memoizer_counters = []

    NodeInfo = FileNodeInfo
    BuildInfo = FileBuildInfo

    def diskcheck_match(self):
        diskcheck_match(self, self.isdir,
                        "Directory %s found where file expected.")

    def __init__(self, name, directory, fs):
        if __debug__: logInstanceCreation(self, 'Node.FS.File')
        Base.__init__(self, name, directory, fs)
        self._morph()

    def Entry(self, name):
        """Create an entry node named 'name' relative to
        the SConscript directory of this file."""
        return self.cwd.Entry(name)

    def Dir(self, name):
        """Create a directory node named 'name' relative to
        the SConscript directory of this file."""
        return self.cwd.Dir(name)

    def Dirs(self, pathlist):
        """Create a list of directories relative to the SConscript
        directory of this file."""
        return map(lambda p, s=self: s.Dir(p), pathlist)

    def File(self, name):
        """Create a file node named 'name' relative to
        the SConscript directory of this file."""
        return self.cwd.File(name)

    #def generate_build_dict(self):
    #    """Return an appropriate dictionary of values for building
    #    this File."""
    #    return {'Dir' : self.Dir,
    #            'File' : self.File,
    #            'RDirs' : self.RDirs}

    def _morph(self):
        """Turn a file system node into a File object."""
        self.scanner_paths = {}
        if not hasattr(self, '_local'):
            self._local = 0

        # If there was already a Builder set on this entry, then
        # we need to make sure we call the target-decider function,
        # not the source-decider.  Reaching in and doing this by hand
        # is a little bogus.  We'd prefer to handle this by adding
        # an Entry.builder_set() method that disambiguates like the
        # other methods, but that starts running into problems with the
        # fragile way we initialize Dir Nodes with their Mkdir builders,
        # yet still allow them to be overridden by the user.  Since it's
        # not clear right now how to fix that, stick with what works
        # until it becomes clear...
        if self.has_builder():
            self.changed_since_last_build = self.decide_target

    def scanner_key(self):
        return self.get_suffix()

    def get_contents(self):
        if not self.rexists():
            return ''
        fname = self.rfile().abspath
        try:
            r = open(fname, "rb").read()
        except EnvironmentError, e:
            if not e.filename:
                e.filename = fname
            raise
        return r

    memoizer_counters.append(SCons.Memoize.CountValue('get_size'))

    def get_size(self):
        try:
            return self._memo['get_size']
        except KeyError:
            pass

        if self.rexists():
            size = self.rfile().getsize()
        else:
            size = 0

        self._memo['get_size'] = size

        return size

    memoizer_counters.append(SCons.Memoize.CountValue('get_timestamp'))

    def get_timestamp(self):
        try:
            return self._memo['get_timestamp']
        except KeyError:
            pass

        if self.rexists():
            timestamp = self.rfile().getmtime()
        else:
            timestamp = 0

        self._memo['get_timestamp'] = timestamp

        return timestamp

    def store_info(self):
        # Merge our build information into the already-stored entry.
        # This accomodates "chained builds" where a file that's a target
        # in one build (SConstruct file) is a source in a different build.
        # See test/chained-build.py for the use case.
        self.dir.sconsign().store_info(self.name, self)

    convert_copy_attrs = [
        'bsources',
        'bimplicit',
        'bdepends',
        'bact',
        'bactsig',
        'ninfo',
    ]


    convert_sig_attrs = [
        'bsourcesigs',
        'bimplicitsigs',
        'bdependsigs',
    ]

    def convert_old_entry(self, old_entry):
        # Convert a .sconsign entry from before the Big Signature
        # Refactoring, doing what we can to convert its information
        # to the new .sconsign entry format.
        #
        # The old format looked essentially like this:
        #
        #   BuildInfo
        #       .ninfo (NodeInfo)
        #           .bsig
        #           .csig
        #           .timestamp
        #           .size
        #       .bsources
        #       .bsourcesigs ("signature" list)
        #       .bdepends
        #       .bdependsigs ("signature" list)
        #       .bimplicit
        #       .bimplicitsigs ("signature" list)
        #       .bact
        #       .bactsig
        #
        # The new format looks like this:
        #
        #   .ninfo (NodeInfo)
        #       .bsig
        #       .csig
        #       .timestamp
        #       .size
        #   .binfo (BuildInfo)
        #       .bsources
        #       .bsourcesigs (NodeInfo list)
        #           .bsig
        #           .csig
        #           .timestamp
        #           .size
        #       .bdepends
        #       .bdependsigs (NodeInfo list)
        #           .bsig
        #           .csig
        #           .timestamp
        #           .size
        #       .bimplicit
        #       .bimplicitsigs (NodeInfo list)
        #           .bsig
        #           .csig
        #           .timestamp
        #           .size
        #       .bact
        #       .bactsig
        #
        # The basic idea of the new structure is that a NodeInfo always
        # holds all available information about the state of a given Node
        # at a certain point in time.  The various .b*sigs lists can just
        # be a list of pointers to the .ninfo attributes of the different
        # dependent nodes, without any copying of information until it's
        # time to pickle it for writing out to a .sconsign file.
        #
        # The complicating issue is that the *old* format only stored one
        # "signature" per dependency, based on however the *last* build
        # was configured.  We don't know from just looking at it whether
        # it was a build signature, a content signature, or a timestamp
        # "signature".  Since we no longer use build signatures, the
        # best we can do is look at the length and if it's thirty two,
        # assume that it was (or might have been) a content signature.
        # If it was actually a build signature, then it will cause a
        # rebuild anyway when it doesn't match the new content signature,
        # but that's probably the best we can do.
        import SCons.SConsign
        new_entry = SCons.SConsign.SConsignEntry()
        new_entry.binfo = self.new_binfo()
        binfo = new_entry.binfo
        for attr in self.convert_copy_attrs:
            try:
                value = getattr(old_entry, attr)
            except AttributeError:
                pass
            else:
                setattr(binfo, attr, value)
                delattr(old_entry, attr)
        for attr in self.convert_sig_attrs:
            try:
                sig_list = getattr(old_entry, attr)
            except AttributeError:
                pass
            else:
                value = []
                for sig in sig_list:
                    ninfo = self.new_ninfo()
                    if len(sig) == 32:
                        ninfo.csig = sig
                    else:
                        ninfo.timestamp = sig
                    value.append(ninfo)
                setattr(binfo, attr, value)
                delattr(old_entry, attr)
        return new_entry

    memoizer_counters.append(SCons.Memoize.CountValue('get_stored_info'))

    def get_stored_info(self):
        try:
            return self._memo['get_stored_info']
        except KeyError:
            pass

        try:
            sconsign_entry = self.dir.sconsign().get_entry(self.name)
        except (KeyError, OSError):
            import SCons.SConsign
            sconsign_entry = SCons.SConsign.SConsignEntry()
            sconsign_entry.binfo = self.new_binfo()
            sconsign_entry.ninfo = self.new_ninfo()
        else:
            if isinstance(sconsign_entry, FileBuildInfo):
                # This is a .sconsign file from before the Big Signature
                # Refactoring; convert it as best we can.
                sconsign_entry = self.convert_old_entry(sconsign_entry)
            try:
                delattr(sconsign_entry.ninfo, 'bsig')
            except AttributeError:
                pass

        self._memo['get_stored_info'] = sconsign_entry

        return sconsign_entry

    def get_stored_implicit(self):
        binfo = self.get_stored_info().binfo
        binfo.prepare_dependencies()
        try: return binfo.bimplicit
        except AttributeError: return None

#    def rel_path(self, other):
#        return self.dir.rel_path(other)

    def _get_found_includes_key(self, env, scanner, path):
        return (id(env), id(scanner), path)

    memoizer_counters.append(SCons.Memoize.CountDict('get_found_includes', _get_found_includes_key))

    def get_found_includes(self, env, scanner, path):
        """Return the included implicit dependencies in this file.
        Cache results so we only scan the file once per path
        regardless of how many times this information is requested.
        """
        memo_key = (id(env), id(scanner), path)
        try:
            memo_dict = self._memo['get_found_includes']
        except KeyError:
            memo_dict = {}
            self._memo['get_found_includes'] = memo_dict
        else:
            try:
                return memo_dict[memo_key]
            except KeyError:
                pass

        if scanner:
            result = scanner(self, env, path)
            result = map(lambda N: N.disambiguate(), result)
        else:
            result = []

        memo_dict[memo_key] = result

        return result

    def _createDir(self):
        # ensure that the directories for this node are
        # created.
        self.dir._create()

    def retrieve_from_cache(self):
        """Try to retrieve the node's content from a cache

        This method is called from multiple threads in a parallel build,
        so only do thread safe stuff here. Do thread unsafe stuff in
        built().

        Returns true iff the node was successfully retrieved.
        """
        if self.nocache:
            return None
        if not self.is_derived():
            return None
        return self.get_build_env().get_CacheDir().retrieve(self)

    def built(self):
        """
        Called just after this node is successfully built.
        """
        # Push this file out to cache before the superclass Node.built()
        # method has a chance to clear the build signature, which it
        # will do if this file has a source scanner.
        #
        # We have to clear the memoized values *before* we push it to
        # cache so that the memoization of the self.exists() return
        # value doesn't interfere.
        self.clear_memoized_values()
        if self.exists():
            self.get_build_env().get_CacheDir().push(self)
        SCons.Node.Node.built(self)

    def visited(self):
        if self.exists():
            self.get_build_env().get_CacheDir().push_if_forced(self)

        ninfo = self.get_ninfo()
        old = self.get_stored_info()

        csig = None
        mtime = self.get_timestamp()
        size = self.get_size()

        max_drift = self.fs.max_drift
        if max_drift > 0:
            if (time.time() - mtime) > max_drift:
                try:
                    n = old.ninfo
                    if n.timestamp and n.csig and n.timestamp == mtime:
                        csig = n.csig
                except AttributeError:
                    pass
        elif max_drift == 0:
            try:
                csig = old.ninfo.csig
            except AttributeError:
                pass

        if csig:
            ninfo.csig = csig

        ninfo.timestamp = mtime
        ninfo.size = size

        if not self.has_builder():
            # This is a source file, but it might have been a target file
            # in another build that included more of the DAG.  Copy
            # any build information that's stored in the .sconsign file
            # into our binfo object so it doesn't get lost.
            self.get_binfo().__dict__.update(old.binfo.__dict__)

        self.store_info()

    def find_src_builder(self):
        if self.rexists():
            return None
        scb = self.dir.src_builder()
        if scb is _null:
            if diskcheck_sccs(self.dir, self.name):
                scb = get_DefaultSCCSBuilder()
            elif diskcheck_rcs(self.dir, self.name):
                scb = get_DefaultRCSBuilder()
            else:
                scb = None
        if scb is not None:
            try:
                b = self.builder
            except AttributeError:
                b = None
            if b is None:
                self.builder_set(scb)
        return scb

    def has_src_builder(self):
        """Return whether this Node has a source builder or not.

        If this Node doesn't have an explicit source code builder, this
        is where we figure out, on the fly, if there's a transparent
        source code builder for it.

        Note that if we found a source builder, we also set the
        self.builder attribute, so that all of the methods that actually
        *build* this file don't have to do anything different.
        """
        try:
            scb = self.sbuilder
        except AttributeError:
            scb = self.sbuilder = self.find_src_builder()
        return not scb is None

    def alter_targets(self):
        """Return any corresponding targets in a build directory.
        """
        if self.is_derived():
            return [], None
        return self.fs.build_dir_target_climb(self, self.dir, [self.name])

    def _rmv_existing(self):
        self.clear_memoized_values()
        Unlink(self, [], None)

    #
    # Taskmaster interface subsystem
    #

    def make_ready(self):
        self.has_src_builder()
        self.get_binfo()

    def prepare(self):
        """Prepare for this file to be created."""
        SCons.Node.Node.prepare(self)

        if self.get_state() != SCons.Node.up_to_date:
            if self.exists():
                if self.is_derived() and not self.precious:
                    self._rmv_existing()
            else:
                try:
                    self._createDir()
                except SCons.Errors.StopError, drive:
                    desc = "No drive `%s' for target `%s'." % (drive, self)
                    raise SCons.Errors.StopError, desc

    #
    #
    #

    def remove(self):
        """Remove this file."""
        if self.exists() or self.islink():
            self.fs.unlink(self.path)
            return 1
        return None

    def do_duplicate(self, src):
        self._createDir()
        try:
            Unlink(self, None, None)
        except SCons.Errors.BuildError:
            pass
        try:
            Link(self, src, None)
        except SCons.Errors.BuildError, e:
            desc = "Cannot duplicate `%s' in `%s': %s." % (src.path, self.dir.path, e.errstr)
            raise SCons.Errors.StopError, desc
        self.linked = 1
        # The Link() action may or may not have actually
        # created the file, depending on whether the -n
        # option was used or not.  Delete the _exists and
        # _rexists attributes so they can be reevaluated.
        self.clear()

    memoizer_counters.append(SCons.Memoize.CountValue('exists'))

    def exists(self):
        try:
            return self._memo['exists']
        except KeyError:
            pass
        # Duplicate from source path if we are set up to do this.
        if self.duplicate and not self.is_derived() and not self.linked:
            src = self.srcnode()
            if not src is self:
                # At this point, src is meant to be copied in a build directory.
                src = src.rfile()
                if src.abspath != self.abspath:
                    if src.exists():
                        self.do_duplicate(src)
                        # Can't return 1 here because the duplication might
                        # not actually occur if the -n option is being used.
                    else:
                        # The source file does not exist.  Make sure no old
                        # copy remains in the build directory.
                        if Base.exists(self) or self.islink():
                            self.fs.unlink(self.path)
                        # Return None explicitly because the Base.exists() call
                        # above will have cached its value if the file existed.
                        self._memo['exists'] = None
                        return None
        result = Base.exists(self)
        self._memo['exists'] = result
        return result

    #
    # SIGNATURE SUBSYSTEM
    #

    def get_csig(self):
        """
        Generate a node's content signature, the digested signature
        of its content.

        node - the node
        cache - alternate node to use for the signature cache
        returns - the content signature
        """
        ninfo = self.get_ninfo()
        try:
            return ninfo.csig
        except AttributeError:
            pass

        try:
            contents = self.get_contents()
        except IOError:
            # This can happen if there's actually a directory on-disk,
            # which can be the case if they've disabled disk checks,
            # or if an action with a File target actually happens to
            # create a same-named directory by mistake.
            csig = None
        else:
            csig = SCons.Util.MD5signature(contents)

        ninfo.csig = csig

        return csig

    #
    # DECISION SUBSYSTEM
    #

    def builder_set(self, builder):
        SCons.Node.Node.builder_set(self, builder)
        self.changed_since_last_build = self.decide_target

    def changed_content(self, target, prev_ni):
        cur_csig = self.get_csig()
        try:
            return cur_csig != prev_ni.csig
        except AttributeError:
            return 1

    def changed_state(self, target, prev_ni):
        return (self.state != SCons.Node.up_to_date)

    def changed_timestamp_then_content(self, target, prev_ni):
        if not self.changed_timestamp_match(target, prev_ni):
            try:
                self.get_ninfo().csig = prev_ni.csig
            except AttributeError:
                pass
            return False
        return self.changed_content(target, prev_ni)

    def changed_timestamp_newer(self, target, prev_ni):
        try:
            return self.get_timestamp() > target.get_timestamp()
        except AttributeError:
            return 1

    def changed_timestamp_match(self, target, prev_ni):
        try:
            return self.get_timestamp() != prev_ni.timestamp
        except AttributeError:
            return 1

    def decide_source(self, target, prev_ni):
        return target.get_build_env().decide_source(self, target, prev_ni)

    def decide_target(self, target, prev_ni):
        return target.get_build_env().decide_target(self, target, prev_ni)

    # Initialize this Node's decider function to decide_source() because
    # every file is a source file until it has a Builder attached...
    changed_since_last_build = decide_source

    def is_up_to_date(self):
        T = 0
        if T: Trace('is_up_to_date(%s):' % self)
        if not self.exists():
            if T: Trace(' not self.exists():')
            # The file doesn't exist locally...
            r = self.rfile()
            if r != self:
                # ...but there is one in a Repository...
                if not self.changed(r):
                    if T: Trace(' changed(%s):' % r)
                    # ...and it's even up-to-date...
                    if self._local:
                        # ...and they'd like a local copy.
                        LocalCopy(self, r, None)
                        self.store_info()
                    if T: Trace(' 1\n')
                    return 1
            self.changed()
            if T: Trace(' None\n')
            return None
        else:
            r = self.changed()
            if T: Trace(' self.exists():  %s\n' % r)
            return not r

    memoizer_counters.append(SCons.Memoize.CountValue('rfile'))

    def rfile(self):
        try:
            return self._memo['rfile']
        except KeyError:
            pass
        result = self
        if not self.exists():
            norm_name = _my_normcase(self.name)
            for dir in self.dir.get_all_rdirs():
                try: node = dir.entries[norm_name]
                except KeyError: node = dir.file_on_disk(self.name)
                if node and node.exists() and \
                   (isinstance(node, File) or isinstance(node, Entry) \
                    or not node.is_derived()):
                        result = node
                        break
        self._memo['rfile'] = result
        return result

    def rstr(self):
        return str(self.rfile())

    def get_cachedir_csig(self):
        """
        Fetch a Node's content signature for purposes of computing
        another Node's cachesig.

        This is a wrapper around the normal get_csig() method that handles
        the somewhat obscure case of using CacheDir with the -n option.
        Any files that don't exist would normally be "built" by fetching
        them from the cache, but the normal get_csig() method will try
        to open up the local file, which doesn't exist because the -n
        option meant we didn't actually pull the file from cachedir.
        But since the file *does* actually exist in the cachedir, we
        can use its contents for the csig.
        """
        try:
            return self.cachedir_csig
        except AttributeError:
            pass

        cachedir, cachefile = self.get_build_env().get_CacheDir().cachepath(self)
        if not self.exists() and cachefile and os.path.exists(cachefile):
            contents = open(cachefile, 'rb').read()
            self.cachedir_csig = SCons.Util.MD5signature(contents)
        else:
            self.cachedir_csig = self.get_csig()
        return self.cachedir_csig

    def get_cachedir_bsig(self):
        try:
            return self.cachesig
        except AttributeError:
            pass

        # Add the path to the cache signature, because multiple
        # targets built by the same action will all have the same
        # build signature, and we have to differentiate them somehow.
        children =  self.children()
        sigs = map(lambda n: n.get_cachedir_csig(), children)
        executor = self.get_executor()
        sigs.append(SCons.Util.MD5signature(executor.get_contents()))
        sigs.append(self.path)
        self.cachesig = SCons.Util.MD5collect(sigs)
        return self.cachesig

default_fs = None

def get_default_fs():
    global default_fs
    if not default_fs:
        default_fs = FS()
    return default_fs

class FileFinder:
    """
    """
    if SCons.Memoize.use_memoizer:
        __metaclass__ = SCons.Memoize.Memoized_Metaclass

    memoizer_counters = []

    def __init__(self):
        self._memo = {}

    def _find_file_key(self, filename, paths, verbose=None):
        return (filename, paths)
        
    memoizer_counters.append(SCons.Memoize.CountDict('find_file', _find_file_key))

    def find_file(self, filename, paths, verbose=None):
        """
        find_file(str, [Dir()]) -> [nodes]

        filename - a filename to find
        paths - a list of directory path *nodes* to search in.  Can be
                represented as a list, a tuple, or a callable that is
                called with no arguments and returns the list or tuple.

        returns - the node created from the found file.

        Find a node corresponding to either a derived file or a file
        that exists already.

        Only the first file found is returned, and none is returned
        if no file is found.
        """
        memo_key = self._find_file_key(filename, paths)
        try:
            memo_dict = self._memo['find_file']
        except KeyError:
            memo_dict = {}
            self._memo['find_file'] = memo_dict
        else:
            try:
                return memo_dict[memo_key]
            except KeyError:
                pass

        if verbose:
            if not SCons.Util.is_String(verbose):
                verbose = "find_file"
            if not callable(verbose):
                verbose = '  %s: ' % verbose
                verbose = lambda s, v=verbose: sys.stdout.write(v + s)
        else:
            verbose = lambda x: x

        filedir, filename = os.path.split(filename)
        if filedir:
            def filedir_lookup(p, fd=filedir):
                try:
                    return p.Dir(fd)
                except TypeError:
                    # We tried to look up a Dir, but it seems there's
                    # already a File (or something else) there.  No big.
                    return None
            paths = filter(None, map(filedir_lookup, paths))

        result = None
        for dir in paths:
            verbose("looking for '%s' in '%s' ...\n" % (filename, dir))
            node, d = dir.srcdir_find_file(filename)
            if node:
                verbose("... FOUND '%s' in '%s'\n" % (filename, d))
                result = node
                break

        memo_dict[memo_key] = result

        return result

find_file = FileFinder().find_file
