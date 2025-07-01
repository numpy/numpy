import os


def get_submodule_paths():
    '''
    Get paths to submodules so that we can exclude them from things like
    check_test_name.py, check_unicode.py, etc.
    '''
    root_directory = os.path.dirname(os.path.dirname(__file__))
    gitmodule_file = os.path.join(root_directory, '.gitmodules')
    with open(gitmodule_file) as gitmodules:
        data = gitmodules.read().split('\n')
        submodule_paths = [datum.split(' = ')[1] for datum in data if
                        datum.startswith('\tpath = ')]
        submodule_paths = [os.path.join(root_directory, path) for path in
                           submodule_paths]
    # vendored with a script rather than via gitmodules
    submodule_paths.append(os.path.join(root_directory, 'scipy/_lib/pyprima'))
    return submodule_paths


if __name__ == "__main__":
    print('\n'.join(get_submodule_paths()))
