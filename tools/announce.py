#!/usr/bin/env python
# -*- encoding:utf-8 -*-
"""
Script to generate contribor and pull request lists

This script generates contributor and pull request lists for release
announcements using Github v3 protocol. Use requires an authentication token in
order to have sufficient bandwidth, you can get one following the directions at
`<https://help.github.com/articles/creating-an-access-token-for-command-line-use/>_
Don't add any scope, as the default is read access to public information. The
token may be stored in an environment variable as you only get one chance to
see it.

Usage::

    $ ./tools/announce.py <token> <revision range>

The output is utf8 rst.

Dependencies
------------

- gitpython
- pygithub

Some code was copied from scipy `tools/gh_list.py` and `tools/authors.py`.

Examples
--------

From the bash command line with $GITHUB token.

    $ ./tools/announce $GITHUB v1.11.0..v1.11.1 > announce.rst

"""
from __future__ import print_function, division

import os
import sys
import re
import codecs
from git import Repo
from github import Github

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)
this_repo = Repo(os.path.join(os.path.dirname(__file__), ".."))

author_msg =\
u"""
A total of %d people contributed to this release.  People with a "+" by their
names contributed a patch for the first time.
"""


def get_authors(revision_range):
    pat = u'.*\\t(.*)\\n'
    lst_release, cur_release = [r.strip() for r in revision_range.split('..')]

    cur = set(re.findall(pat, this_repo.git.shortlog('-s', revision_range)))
    pre = set(re.findall(pat, this_repo.git.shortlog('-s', lst_release)))
    authors = [s + u' +' for s in cur - pre] + [s for s in cur & pre]
    authors.sort()
    return authors


def get_prs(repo, revision_range):
    # get pull request numbers from merges
    merges = this_repo.git.log(
        '--oneline', '--merges', revision_range)
    issues = re.findall(u"Merge pull request \#(\d*)", merges)
    merge_prnums = [int(s) for s in issues]

    # get pull request numbers from fast forward squash-merges
    commits = this_repo.git.log(
        '--oneline', '--no-merges', '--first-parent', revision_range)
    issues = re.findall(u'.*\(\#(\d+)\)\n', commits)
    squash_prnums = [int(s) for s in issues]

    # get PR data from github repo
    prs = [repo.get_pull(n) for n in sorted(merge_prnums + squash_prnums)]
    return prs


def main(token, revision_range):
    lst_release, cur_release = [r.strip() for r in revision_range.split('..')]

    github = Github(token)
    github_repo = github.get_repo('numpy/numpy')

    # document authors
    authors = get_authors(revision_range)
    heading = u"Contributors to {0}".format(cur_release)
    print()
    print(heading)
    print(u"-"*len(heading))
    print()
    for s in authors:
        print(u'- ' + s)
    print(author_msg % len(authors))

    # document pull requests
    heading = u"Pull requests merged for {0}".format(cur_release)
    print()
    print(heading)
    print(u"-"*len(heading))
    print()
    for pull in get_prs(github_repo, revision_range):
        pull_msg = u"- `#{0} <{1}>`__: {2}"
        title = re.sub(u"\s+", u" ", pull.title.strip())
        if len(title) > 60:
            remainder = re.sub(u"\s.*$", u"...", title[60:])
            if len(remainder) > 20:
                remainder = title[:80] + u"..."
            else:
                title = title[:60] + remainder
        print(pull_msg.format(pull.number, pull.html_url, title))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate author/pr lists for release")
    parser.add_argument('token', help='github access token')
    parser.add_argument('revision_range', help='<revision>..<revision>')
    args = parser.parse_args()
    main(args.token, args.revision_range)
