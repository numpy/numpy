# dblite.py module contributed by Ralf W. Grosse-Kunstleve.
# Extended for Unicode by Steven Knight.

import cPickle
import time
import shutil
import os
import os.path
import types
import __builtin__

_open = __builtin__.open # avoid name clash

keep_all_files = 00000
ignore_corrupt_dbfiles = 0

def corruption_warning(filename):
    print "Warning: Discarding corrupt database:", filename

if hasattr(types, 'UnicodeType'):
    def is_string(s):
        t = type(s)
        return t is types.StringType or t is types.UnicodeType
else:
    def is_string(s):
        return type(s) is types.StringType

try:
    unicode('a')
except NameError:
    def unicode(s): return s

dblite_suffix = '.dblite'
tmp_suffix = '.tmp'

class dblite:

  def __init__(self, file_base_name, flag, mode):
    assert flag in (None, "r", "w", "c", "n")
    if (flag is None): flag = "r"
    base, ext = os.path.splitext(file_base_name)
    if ext == dblite_suffix:
      # There's already a suffix on the file name, don't add one.
      self._file_name = file_base_name
      self._tmp_name = base + tmp_suffix
    else:
      self._file_name = file_base_name + dblite_suffix
      self._tmp_name = file_base_name + tmp_suffix
    self._flag = flag
    self._mode = mode
    self._dict = {}
    self._needs_sync = 00000
    if (self._flag == "n"):
      _open(self._file_name, "wb", self._mode)
    else:
      try:
        f = _open(self._file_name, "rb")
      except IOError, e:
        if (self._flag != "c"):
          raise e
        _open(self._file_name, "wb", self._mode)
      else:
        p = f.read()
        if (len(p) > 0):
          try:
            self._dict = cPickle.loads(p)
          except (cPickle.UnpicklingError, EOFError):
            if (ignore_corrupt_dbfiles == 0): raise
            if (ignore_corrupt_dbfiles == 1):
              corruption_warning(self._file_name)

  def __del__(self):
    if (self._needs_sync):
      self.sync()

  def sync(self):
    self._check_writable()
    f = _open(self._tmp_name, "wb", self._mode)
    cPickle.dump(self._dict, f, 1)
    f.close()
    # Windows doesn't allow renaming if the file exists, so unlink
    # it first, chmod'ing it to make sure we can do so.  On UNIX, we
    # may not be able to chmod the file if it's owned by someone else
    # (e.g. from a previous run as root).  We should still be able to
    # unlink() the file if the directory's writable, though, so ignore
    # any OSError exception  thrown by the chmod() call.
    try: os.chmod(self._file_name, 0777)
    except OSError: pass
    os.unlink(self._file_name)
    os.rename(self._tmp_name, self._file_name)
    self._needs_sync = 00000
    if (keep_all_files):
      shutil.copyfile(
        self._file_name,
        self._file_name + "_" + str(int(time.time())))

  def _check_writable(self):
    if (self._flag == "r"):
      raise IOError("Read-only database: %s" % self._file_name)

  def __getitem__(self, key):
    return self._dict[key]

  def __setitem__(self, key, value):
    self._check_writable()
    if (not is_string(key)):
      raise TypeError, "key `%s' must be a string but is %s" % (key, type(key))
    if (not is_string(value)):
      raise TypeError, "value `%s' must be a string but is %s" % (value, type(value))
    self._dict[key] = value
    self._needs_sync = 0001

  def keys(self):
    return self._dict.keys()

  def has_key(self, key):
    return key in self._dict

  def __contains__(self, key):
    return key in self._dict

  def iterkeys(self):
    return self._dict.iterkeys()

  __iter__ = iterkeys

  def __len__(self):
    return len(self._dict)

def open(file, flag=None, mode=0666):
  return dblite(file, flag, mode)

def _exercise():
  db = open("tmp", "n")
  assert len(db) == 0
  db["foo"] = "bar"
  assert db["foo"] == "bar"
  db[unicode("ufoo")] = unicode("ubar")
  assert db[unicode("ufoo")] == unicode("ubar")
  db.sync()
  db = open("tmp", "c")
  assert len(db) == 2, len(db)
  assert db["foo"] == "bar"
  db["bar"] = "foo"
  assert db["bar"] == "foo"
  db[unicode("ubar")] = unicode("ufoo")
  assert db[unicode("ubar")] == unicode("ufoo")
  db.sync()
  db = open("tmp", "r")
  assert len(db) == 4, len(db)
  assert db["foo"] == "bar"
  assert db["bar"] == "foo"
  assert db[unicode("ufoo")] == unicode("ubar")
  assert db[unicode("ubar")] == unicode("ufoo")
  try:
    db.sync()
  except IOError, e:
    assert str(e) == "Read-only database: tmp.dblite"
  else:
    raise RuntimeError, "IOError expected."
  db = open("tmp", "w")
  assert len(db) == 4
  db["ping"] = "pong"
  db.sync()
  try:
    db[(1,2)] = "tuple"
  except TypeError, e:
    assert str(e) == "key `(1, 2)' must be a string but is <type 'tuple'>", str(e)
  else:
    raise RuntimeError, "TypeError exception expected"
  try:
    db["list"] = [1,2]
  except TypeError, e:
    assert str(e) == "value `[1, 2]' must be a string but is <type 'list'>", str(e)
  else:
    raise RuntimeError, "TypeError exception expected"
  db = open("tmp", "r")
  assert len(db) == 5
  db = open("tmp", "n")
  assert len(db) == 0
  _open("tmp.dblite", "w")
  db = open("tmp", "r")
  _open("tmp.dblite", "w").write("x")
  try:
    db = open("tmp", "r")
  except cPickle.UnpicklingError:
    pass
  else:
    raise RuntimeError, "cPickle exception expected."
  global ignore_corrupt_dbfiles
  ignore_corrupt_dbfiles = 2
  db = open("tmp", "r")
  assert len(db) == 0
  os.unlink("tmp.dblite")
  try:
    db = open("tmp", "w")
  except IOError, e:
    assert str(e) == "[Errno 2] No such file or directory: 'tmp.dblite'", str(e)
  else:
    raise RuntimeError, "IOError expected."
  print "OK"

if (__name__ == "__main__"):
  _exercise()
