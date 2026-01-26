import os
import sys

from . import common


def show_cpu_features():
    from numpy.lib._utils_impl import _opt_info
    info = _opt_info()
    info = "NumPy CPU features: " + (info or 'nothing enabled')
    # ASV wrapping stdout & stderr, so we assume having a tty here
    if 'SHELL' in os.environ and sys.platform != 'win32':
        # to avoid the red color that imposed by ASV
        print(f"\033[33m{info}\033[0m")
    else:
        print(info)

def dirty_lock(lock_name, lock_on_count=1):
    # this lock occurred before each round to avoid duplicate printing
    if not hasattr(os, "getppid"):
        return False
    ppid = os.getppid()
    if not ppid or ppid == os.getpid():
        # not sure if this gonna happen, but ASV run each round in
        # a separate process so the lock should be based on the parent
        # process id only
        return False
    lock_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "env", lock_name)
    )
    # ASV loads the 'benchmark_dir' to discover the available benchmarks
    # the issue here is ASV doesn't capture any strings from stdout or stderr
    # during this stage so we escape it and lock on the second increment
    try:
        with open(lock_path, 'a+') as f:
            f.seek(0)
            count, _ppid = (f.read().split() + [0, 0])[:2]
            count, _ppid = int(count), int(_ppid)
            if _ppid == ppid:
                if count >= lock_on_count:
                    return True
                count += 1
            else:
                count = 0
            f.seek(0)
            f.truncate()
            f.write(f"{count} {ppid}")
    except OSError:
        pass
    return False


# FIXME: there's no official way to provide extra information to the test log
if not dirty_lock("print_cpu_features.lock"):
    show_cpu_features()
