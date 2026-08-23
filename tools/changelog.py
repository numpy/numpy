#!/usr/bin/env python3
"""
Script to generate contributor and pull request lists

This script generates contributor and pull request lists for release
changelogs using Github v3 protocol. Use requires an authentication token in
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
- git >= 2.29.0

Some code was copied from scipy `tools/gh_list.py` and `tools/authors.py`.

Examples
--------

From the bash command line with $GITHUB token::

    $ ./tools/announce $GITHUB v1.13.0..v1.14.0 > 1.14.0-changelog.rst

"""
import os
import re

from git import Repo
from github import Github

this_repo = Repo(os.path.join(os.path.dirname(__file__), ".."))

author_msg =\
"""
A total of %d people contributed to this release.  People with a "+" by their
names contributed a patch for the first time.
"""

pull_request_msg =\
"""
A total of %d pull requests were merged for this release.
"""


def get_authors(revision_range):
    lst_release, cur_release = [r.strip() for r in revision_range.split('..')]
    authors_pat = r'^.*\t(.*)$'

    # authors and co-authors in current and previous releases.
    grp1 = '--group=author'
    grp2 = '--group=trailer:co-authored-by'
    cur = this_repo.git.shortlog('-s', grp1, grp2, revision_range)
    pre = this_repo.git.shortlog('-s', grp1, grp2, lst_release)
    authors_cur = set(re.findall(authors_pat, cur, re.M))
    authors_pre = set(re.findall(authors_pat, pre, re.M))

    # Ignore the bot Homu.
    authors_cur.discard('Homu')
    authors_pre.discard('Homu')

    # Ignore the bot dependabot-preview
    authors_cur.discard('dependabot-preview')
    authors_pre.discard('dependabot-preview')

    # Append '+' to new authors.
    authors_new = [s + ' +' for s in authors_cur - authors_pre]
    authors_old = list(authors_cur & authors_pre)
    authors = authors_new + authors_old
    authors.sort()
    return authors


def get_pull_requests(repo, revision_range):
    prnums = []

    # From regular merges
    merges = this_repo.git.log(
        '--oneline', '--merges', revision_range)
    issues = re.findall(r"Merge pull request \#(\d*)", merges)
    prnums.extend(int(s) for s in issues)

    # From Homu merges (Auto merges)
    issues = re. findall(r"Auto merge of \#(\d*)", merges)
    prnums.extend(int(s) for s in issues)

    # From fast forward squash-merges
    commits = this_repo.git.log(
        '--oneline', '--no-merges', '--first-parent', revision_range)
    issues = re.findall(r'^.*\((\#|gh-|gh-\#)(\d+)\)$', commits, re.M)
    prnums.extend(int(s[1]) for s in issues)

    # get PR data from github repo
    prnums.sort()
    prs = [repo.get_pull(n) for n in prnums]
    return prs


def main(token, revision_range):
    lst_release, cur_release = [r.strip() for r in revision_range.split('..')]

    github = Github(token)
    github_repo = github.get_repo('numpy/numpy')

    # document authors
    authors = get_authors(revision_range)
    heading = "Contributors"
    print()
    print(heading)
    print("=" * len(heading))
    print(author_msg % len(authors))

    for s in authors:
        print('* ' + s)

    # document pull requests
    pull_requests = get_pull_requests(github_repo, revision_range)
    heading = "Pull requests merged"
    pull_msg = "* `#{0} <{1}>`__: {2}"

    print()
    print(heading)
    print("=" * len(heading))
    print(pull_request_msg % len(pull_requests))

    def backtick_repl(matchobj):
        """repl to add an escaped space following a code block if needed"""
        if matchobj.group(2) != ' ':
            post = r'\ ' + matchobj.group(2)
        else:
            post = matchobj.group(2)
        return '``' + matchobj.group(1) + '``' + post

    for pull in pull_requests:
        # sanitize whitespace
        title = re.sub(r"\s+", " ", pull.title.strip())

        # substitute any single backtick not adjacent to a backtick
        # for a double backtick
        title = re.sub(
            r"(?P<pre>(?:^|(?<=[^`])))`(?P<post>(?=[^`]|$))",
            r"\g<pre>``\g<post>",
            title
        )
        # add an escaped space if code block is not followed by a space
        title = re.sub(r"``(.*?)``(.)", backtick_repl, title)

        # sanitize asterisks
        title = title.replace('*', '\\*')

        if len(title) > 60:
            remainder = re.sub(r"\s.*$", "...", title[60:])
            if len(remainder) > 20:
                # just use the first 80 characters, with ellipses.
                # note: this was previously bugged,
                # assigning to `remainder` rather than `title`
                title = title[:80] + "..."
            else:
                # use the first 60 characters and the next word
                title = title[:60] + remainder

            if title.count('`') % 4 != 0:
                # ellipses have cut in the middle of a code block,
                # so finish the code block before the ellipses
                title = title[:-3] + '``...'

        print(pull_msg.format(pull.number, pull.html_url, title))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate author/pr lists for release")
    parser.add_argument('token', help='github access token')
    parser.add_argument('revision_range', help='<revision>..<revision>')
    args = parser.parse_args()
    main(args.token, args.revision_range)
