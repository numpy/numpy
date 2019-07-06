#!/usr/bin/env python

import argparse
import subprocess
import tempfile
import os
import sys
import shutil
import re


parser = argparse.ArgumentParser(
    description='Update static site in a remote repo, replacing existing content'
)
parser.add_argument('dir', help='directory of which content will be uploaded')
parser.add_argument('branch', help='agains which brach was the content built')
parser.add_argument('remote', help='remote to which content will be pushed')
parser.add_argument('--message', default='Commit bot upload',
                    help='commit message to use')
parser.add_argument('--committer', default='numpy-commit-bot',
                    help='Name of the git committer')
parser.add_argument('--email', default='numpy-commit-bot@nomail',
                    help='Email of the git committer')
parser.add_argument('--target', help='"neps" or empty')
args = parser.parse_args()
args.dir = os.path.abspath(args.dir)

if not os.path.exists(args.dir):
    print('Content directory does not exist')
    sys.exit(1)

m = re.search('maintenance/([\d.]*)\.x', args.branch)
if args.target:
    target = args.target
elif args.branch == 'master':
    target = 'devdocs'
elif m:
    target = m.groups()[0]
else:
    print('Only use this script to update master or a maintenance branch')
    sys.exit(1)

def run(cmd, stdout=True):
    pipe = None if stdout else subprocess.DEVNULL
    try:
        subprocess.check_call(cmd, stdout=pipe, stderr=pipe)
    except subprocess.CalledProcessError:
        print("\n! Error executing: `%s;` aborting" % ' '.join(cmd))
        sys.exit(1)


workdir = tempfile.mkdtemp()
os.chdir(workdir)

run(['git', 'clone', args.remote, 'built_doc'])
os.chdir('built_doc')
run(['git', 'config', '--local', 'user.name', args.committer])
run(['git', 'config', '--local', 'user.email', args.email])

print('- committing new content: "%s"' % args.message)
run(['git', 'rm', '--ignore-unmatch', target], stdout=False)
run(['cp', '-R', os.path.join(args.dir, '.'), target])
run(['git', 'add', target], stdout=False)
run(['git', 'commit', '--allow-empty', '-m', args.message], stdout=False)

print('- uploading as %s <%s>' % (args.committer, args.email))
run(['git', 'push', 'origin', 'master'])
shutil.rmtree(workdir)
