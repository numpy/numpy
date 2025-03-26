import os
import re
import numpy as np
from git import Repo
from github import Github
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize repository
this_repo = Repo(os.path.join(os.path.dirname(__file__), ".."))

author_msg =\
"""
A total of %d people contributed to this release. People with a "+" by their
names contributed a patch for the first time.
"""

pull_request_msg =\
"""
A total of %d pull requests were merged for this release.
"""

def get_authors(revision_range):
    lst_release, cur_release = [r.strip() for r in revision_range.split('..')]
    authors_pat = r'^.*\t(.*)$'

    # authors and co-authors in current and previous releases
    grp1 = '--group=author'
    grp2 = '--group=trailer:co-authored-by'
    cur = this_repo.git.shortlog('-s', grp1, grp2, revision_range)
    pre = this_repo.git.shortlog('-s', grp1, grp2, lst_release)
    authors_cur = set(re.findall(authors_pat, cur, re.M))
    authors_pre = set(re.findall(authors_pat, pre, re.M))

    # Ignore bots
    authors_cur.discard('Homu')
    authors_pre.discard('Homu')
    authors_cur.discard('dependabot-preview')
    authors_pre.discard('dependabot-preview')

    # Append '+' to new authors
    authors_new = [s + ' +' for s in authors_cur - authors_pre]
    authors_old = list(authors_cur & authors_pre)
    authors = authors_new + authors_old
    authors.sort()
    return authors

def get_pull_requests(repo, revision_range):
    prnums = []

    # From regular merges
    merges = this_repo.git.log('--oneline', '--merges', revision_range)
    issues = re.findall(r"Merge pull request \#(\d*)", merges)
    prnums.extend(int(s) for s in issues)

    # From Homu merges (Auto merges)
    issues = re.findall(r"Auto merge of \#(\d*)", merges)
    prnums.extend(int(s) for s in issues)

    # From fast forward squash-merges
    commits = this_repo.git.log('--oneline', '--no-merges', '--first-parent', revision_range)
    issues = re.findall(r'^.*\((\#|gh-|gh-\#)(\d+)\)$', commits, re.M)
    prnums.extend(int(s[1]) for s in issues)


    prnums.sort()
    prs = [repo.get_pull(n) for n in prnums]
    return prs

def analyze_pull_requests(pulls):
    """
    Analyzes pull request titles to identify key topics and contributions using TF-IDF and cosine similarity.
    """
    titles = [pr.title for pr in pulls]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(titles)
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(X)
    return similarity_matrix

def main(token, revision_range):
    lst_release, cur_release = [r.strip() for r in revision_range.split('..')]

    github = Github(token)
    github_repo = github.get_repo('numpy/numpy')

    # Document authors
    authors = get_authors(revision_range)
    heading = "Contributors"
    print()
    print(heading)
    print("="*len(heading))
    print(author_msg % len(authors))

    for s in authors:
        print('* ' + s)

    # Document pull requests
    pull_requests = get_pull_requests(github_repo, revision_range)
    pull_similarities = analyze_pull_requests(pull_requests)

    heading = "Pull requests merged"
    pull_msg = "* `#{0} <{1}>`__: {2}"

    print()
    print(heading)
    print("="*len(heading))
    print(pull_request_msg % len(pull_requests))

    for i, pull in enumerate(pull_requests):
        title = re.sub(r"\s+", " ", pull.title.strip())
        if len(title) > 60:
            remainder = re.sub(r"\s.*$", "...", title[60:])
            if len(remainder) > 20:
                remainder = title[:80] + "..."
            else:
                title = title[:60] + remainder
        print(pull_msg.format(pull.number, pull.html_url, title))
        
    # Optionally print similarity matrix for debugging
    print("\nSimilarity matrix for PR titles:")
    print(pull_similarities)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate author/pr lists for release")
    parser.add_argument('token', help='GitHub access token')
    parser.add_argument('revision_range', help='<revision>..<revision>')
    args = parser.parse_args()
    main(args.token, args.revision_range)
