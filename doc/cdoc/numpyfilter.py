#!/usr/bin/env python
"""
numpyfilter.py INPUTFILE

Interpret C comments as ReStructuredText, and replace them by the HTML output.
Also, add Doxygen /** and /**< syntax automatically where appropriate.

"""
from __future__ import division, absolute_import, print_function

import sys
import re
import os
import textwrap
import optparse

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle

CACHE_FILE = 'build/rst-cache.pck'

def main():
    p = optparse.OptionParser(usage=__doc__.strip())
    options, args = p.parse_args()

    if len(args) != 1:
        p.error("no input file given")

    comment_re = re.compile(r'(\n.*?)/\*(.*?)\*/', re.S)

    cache = load_cache()

    f = open(args[0], 'r')
    try:
        text = f.read()
        text = comment_re.sub(lambda m: process_match(m, cache), text)
        sys.stdout.write(text)
    finally:
        f.close()
        save_cache(cache)

def filter_comment(text):
    if text.startswith('NUMPY_API'):
        text = text[9:].strip()
    if text.startswith('UFUNC_API'):
        text = text[9:].strip()

    html = render_html(text)
    return html

def process_match(m, cache=None):
    pre, rawtext = m.groups()

    preline = pre.split("\n")[-1]

    if cache is not None and rawtext in cache:
        text = cache[rawtext]
    else:
        text = re.compile(r'^\s*\*', re.M).sub('', rawtext)
        text = textwrap.dedent(text)
        text = filter_comment(text)

        if cache is not None:
            cache[rawtext] = text

    if preline.strip():
        return pre + "/**< " + text + " */"
    else:
        return pre + "/** " + text + " */"

def load_cache():
    if os.path.exists(CACHE_FILE):
        f = open(CACHE_FILE, 'rb')
        try:
            cache = pickle.load(f)
        except:
            cache = {}
        finally:
            f.close()
    else:
        cache = {}
    return cache

def save_cache(cache):
    f = open(CACHE_FILE + '.new', 'wb')
    try:
        pickle.dump(cache, f)
    finally:
        f.close()
    os.rename(CACHE_FILE + '.new', CACHE_FILE)

def render_html(text):
    import docutils.parsers.rst
    import docutils.writers.html4css1
    import docutils.core

    docutils.parsers.rst.roles.DEFAULT_INTERPRETED_ROLE = 'title-reference'
    writer = docutils.writers.html4css1.Writer()
    parts = docutils.core.publish_parts(
        text,
        writer=writer,
        settings_overrides = dict(halt_level=5,
                                  traceback=True,
                                  default_reference_context='title-reference',
                                  stylesheet_path='',
                                  # security settings:
                                  raw_enabled=0,
                                  file_insertion_enabled=0,
                                  _disable_config=1,
                                  )
    )
    return parts['html_body'].encode('utf-8')

if __name__ == "__main__": main()
