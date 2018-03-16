#!/usr/bin/env python

import argparse
import subprocess
import tempfile
import os
import sys
import shutil


parser = argparse.ArgumentParser(
    description='Upload files to a remote repo, replacing existing content'
)
parser.add_argument('dir', help='directory of which content will be uploaded')
parser.add_argument('remote', help='remote to which content will be pushed')
parser.add_argument('--message', default='Commit bot upload',
                    help='commit message to use')
parser.add_argument('--committer', default='numpy-commit-bot',
                    help='Name of the git committer')
parser.add_argument('--email', default='numpy-commit-bot@nomail',
                    help='Email of the git committer')

parser.add_argument(
    '--force', action='store_true',
    help='hereby acknowledge that remote repo content will be overwritten'
)
args = parser.parse_args()
args.dir = os.path.abspath(args.dir)

if not os.path.exists(args.dir):
    print('Content directory does not exist')
    sys.exit(1)


def run(cmd, stdout=True):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if stdout:
        for pipe in (p.stdout, p.stderr):
            for line in pipe:
                print(line.decode(), end='')
    p.communicate()

    if p.returncode:
        print("Error executing: '%s'" % ' '.join(cmd))


workdir = tempfile.mkdtemp()
os.chdir(workdir)

run(['git', 'init'])
run(['git', 'remote', 'add', 'origin',  args.remote])
run(['git', 'config', '--local', 'user.name', args.committer])
run(['git', 'config', '--local', 'user.email', args.email])

print('- committing new content: "%s"' % args.message)
run(['cp', '-R', os.path.join(args.dir, '.'), '.'])
run(['git', 'add', '.'], stdout=False)
run(['git', 'commit', '--allow-empty', '-m', args.message], stdout=False)

print('- uploading as %s <%s>' % (args.committer, args.email))
if args.force:
    run(['git', 'push', 'origin', 'master', '--force'])
else:
    print('\n!! No `--force` argument specified; aborting')
    print('!! Before enabling that flag, make sure you know what it does\n')
    sys.exit(1)

shutil.rmtree(workdir)
