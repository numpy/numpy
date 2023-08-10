import os
import textwrap


def init_version():
    init = os.path.join(os.path.dirname(__file__), '../../pyproject.toml')
    data = open(init).readlines()

    version_line = next(
        line for line in data if line.startswith('version =')
    )

    version = version_line.strip().split(' = ')[1]
    version = version.replace('"', '').replace("'", '')

    return version


def git_version(version):
    if 'dev' in version:
        # Append last commit date and hash to dev version information,
        # if available

        import subprocess
        import os.path

        try:
            p = subprocess.Popen(
                ['git', 'log', '-1', '--format="%H %aI"'],
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
                version += f'+git{git_date}.{git_hash[:7]}'
            else:
                git_hash = ''

        return version, git_hash


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--write', help="Save version to this file")
    args = parser.parse_args()

    version, git_hash = git_version(init_version())

    # For NumPy 2.0, this should only have one field: `version`
    template = textwrap.dedent(f'''
        version = "{version}"
        __version__ = version
        full_version = version

        git_revision = "{git_hash}"
        release = 'dev' not in version and '+' not in version
        short_version = version.split("+")[0]
    ''')

    if args.write:
        with open(args.write, 'w') as f:
            f.write(template)

    print(version)
