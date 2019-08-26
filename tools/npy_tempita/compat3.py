from __future__ import absolute_import, division, print_function

import sys

__all__ = ['PY3', 'b', 'basestring_', 'bytes', 'next', 'is_unicode',
           'iteritems']

PY3 = True if sys.version_info[0] >= 3 else False

if sys.version_info[0] < 3:

    def next(obj):
        return obj.next()

    def iteritems(d, **kw):
        return d.iteritems(**kw)

    b = bytes = str
    basestring_ = basestring

else:

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
    if sys.version_info[0] < 3:
        return isinstance(obj, unicode)
    else:
        return isinstance(obj, str)


def coerce_text(v):
    if not isinstance(v, basestring_):
        if sys.version_info[0] < 3:
            attr = '__unicode__'
        else:
            attr = '__str__'
        if hasattr(v, attr):
            return unicode(v)
        else:
            return bytes(v)
    return v
