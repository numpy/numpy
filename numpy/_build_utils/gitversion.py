import os


def init_version():
    init = os.path.join(os.path.dirname(__file__), '../__init__.py')
    data = open(init).readlines()

    version_line = next(line for line in data if line.startswith('__version__ ='))

    version = version_line.strip().split(' = ')[1].replace('"', '').replace("'", '')

    return version


def git_version(version):
    if 'dev' in version:
        # Append last commit date and hash to dev version information, if available

        import subprocess
        import os.path

        try:
            p = subprocess.Popen(
                ['git', 'log', '-1', '--format="%h %aI"'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(__file__),
            )
        except FileNotFoundError:
            pass
        else:
            out, err = p.communicate()
            if p.returncode == 0:
                git_hash, git_date = (
                    out.decode('utf-8')
                    .strip()
                    .replace('"', '')
                    .split('T')[0]
                    .replace('-', '')
                    .split()
                )

                version__ = '+'.join(
                    [tag for tag in version.split('+')
                     if not tag.startswith('git')]
                )
                version += f'+git{git_date}.{git_hash}'

        return version


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--write', help="Save version to this file")
    args = parser.parse_args()

    version = git_version(init_version())

    if args.write:
        with open(args.write, 'w') as f:
            f.write(f'version = "{version}"\n')

    print(version)
