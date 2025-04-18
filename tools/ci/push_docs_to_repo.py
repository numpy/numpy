#!/usr/bin/env python3

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
parser.add_argument('--count', default=1, type=int,
                    help="minimum number of expected files, defaults to 1")

parser.add_argument(
    '--force', action='store_true',
    help='hereby acknowledge that remote repo content will be overwritten'
)
args = parser.parse_args()
args.dir = os.path.abspath(args.dir)

if not os.path.exists(args.dir):
    print('Content directory does not exist')
    sys.exit(1)

count = len([name for name in os.listdir(args.dir)
             if os.path.isfile(os.path.join(args.dir, name))])

if count < args.count:
    print(f"Expected {args.count} top-directory files to upload, got {count}")
    sys.exit(1)

def run(cmd, stdout=True):
    pipe = None if stdout else subprocess.DEVNULL
    try:
        subprocess.check_call(cmd, stdout=pipe, stderr=pipe)
    except subprocess.CalledProcessError:
        print(f"\n! Error executing: `{' '.join(cmd)};` aborting")
        sys.exit(1)


workdir = tempfile.mkdtemp()
os.chdir(workdir)

run(['git', 'init'])
# ensure the working branch is called "main"
# (`--initial-branch=main` appeared to have failed on older git versions):
run(['git', 'checkout', '-b', 'main'])
run(['git', 'remote', 'add', 'origin',  args.remote])
run(['git', 'config', '--local', 'user.name', args.committer])
run(['git', 'config', '--local', 'user.email', args.email])

print(f'- committing new content: "{args.message}"')
run(['cp', '-R', os.path.join(args.dir, '.'), '.'])
run(['git', 'add', '.'], stdout=False)
run(['git', 'commit', '--allow-empty', '-m', args.message], stdout=False)

print(f'- uploading as {args.committer} <{args.email}>')
if args.force:
    run(['git', 'push', 'origin', 'main', '--force'])
else:
    print('\n!! No `--force` argument specified; aborting')
    print('!! Before enabling that flag, make sure you know what it does\n')
    sys.exit(1)

shutil.rmtree(workdir)
