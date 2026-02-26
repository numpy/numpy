# NumPy Pixi packages

This directory contains definitions for [Pixi
packages](https://pixi.sh/latest/reference/pixi_manifest/#the-package-section) which can
be built from the NumPy source code.

Downstream developers can make use of these packages by adding them as Git dependencies
in a [Pixi workspace](https://pixi.sh/latest/first_workspace/).

This is particularly useful when developers need to build NumPy from source
(for example, for an ASan-instrumented build), as it does not require any manual
clone or build steps. Instead, Pixi will automatically handle both the build
and installation of the package.

See [scipy#24066](https://github.com/scipy/scipy/pull/24066) for a full example of
downstream use.

## Variants
Each package definition is contained in a subdirectory.
All package variants include debug symbols.

Currently defined variants:

### `default`
GIL-enabled build.

Usage:
```toml
[dependencies]
python = "*"
numpy.git = "https://github.com/numpy/numpy"
numpy.subdirectory = "pixi-packages/default"
```
See `default/pixi.toml` if you wish to use python git tip instead.

*Tip:* you may change fork and add `numpy.rev = "<branch or git hash>"` to test unmerged
PRs.

### `freethreading`
noGIL build.

Usage:
```toml
[dependencies]
python-freethreading = "*"
numpy.git = "https://github.com/numpy/numpy"
numpy.subdirectory = "pixi-packages/freethreading"
```
See `freethreading/pixi.toml` if you wish to use python git tip instead.

### `asan`
ASan-instrumented build with `-Db_sanitize=address`.

Usage:
```toml
[dependencies]
python.git = "https://github.com/python/cpython"
python.subdirectory = "Tools/pixi-packages/asan"
numpy.git = "https://github.com/numpy/numpy"
numpy.subdirectory = "pixi-packages/asan"
```

### `tsan-freethreading`
Freethreading TSan-instrumented build with `-Db_sanitize=thread`.

Usage:
```toml
[dependencies]
python.git = "https://github.com/python/cpython"
python.subdirectory = "Tools/pixi-packages/tsan-freethreading"
numpy.git = "https://github.com/numpy/numpy"
numpy.subdirectory = "pixi-packages/tsan-freethreading"
```

## Maintenance

- Keep host dependency requirements up to date

## Troubleshooting

TSan builds may crash on Linux with
```
FATAL: ThreadSanitizer: unexpected memory mapping 0x7977bd072000-0x7977bd500000
```
To fix it, try reducing `mmap_rnd_bits`:

```bash
$ sudo sysctl vm.mmap_rnd_bits
vm.mmap_rnd_bits = 32  # too high for TSan
$ sudo sysctl vm.mmap_rnd_bits=28  # reduce it
vm.mmap_rnd_bits = 28
```

## Opportunities for future improvement

- More package variants (such as UBSan)
- Support for Windows
- Using a single `pixi.toml` for all package variants is blocked on
  [pixi#2813](https://github.com/prefix-dev/pixi/issues/2813)
- Consider pinning dependency versions to guard against upstream breakages over time

## Known issues
- [numpy#30561](https://github.com/numpy/numpy/issues/30561): `default` and
  `freethreading` recipes must be manually tweaked to compile against cpython git tip;
  see `default/pixi.toml` and `freethreading/pixi.toml` for details.
- [pixi#5226](https://github.com/prefix-dev/pixi/issues/5226): lock file is invalidated
  on all `pixi` invocations
- [rattler-build#2094](https://github.com/prefix-dev/rattler-build/issues/2094): pixi
  0.63.0 introduces a regression regarding the license file; please skip it
