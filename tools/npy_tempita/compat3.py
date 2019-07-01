from __future__ import absolute_import, division, print_function

__all__ = ['PY3', 'b', 'basestring_', 'bytes', 'next', 'is_unicode',
           'iteritems']

PY3 = True

def b(s):
    if isinstance(s, str):
        return s.encode('latin1')
    return bytes(s)

def iteritems(d, **kw):
    return iter(d.items(**kw))

next = next
basestring_ = (bytes, str)
bytes = bytes
text = str


def is_unicode(obj):
    return isinstance(obj, str)


def coerce_text(v):
    if not isinstance(v, basestring_):
        attr = '__str__'
        if hasattr(v, attr):
            return str(v)
        else:
            return bytes(v)
    return v
