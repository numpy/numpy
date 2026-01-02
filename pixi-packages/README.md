# NumPy Pixi packages

This directory contains definitions for [Pixi packages](https://pixi.sh/latest/reference/pixi_manifest/#the-package-section)
which can be built from the NumPy source code.

Downstream developers can make use of these packages by adding them as Git dependencies in a
[Pixi workspace](https://pixi.sh/latest/first_workspace/), like:

```toml
[dependencies]
numpy = { git = "https://github.com/numpy/numpy", subdirectory = "pixi-packages/asan" }
```

This is particularly useful when developers need to build NumPy from source
(for example, for an ASan-instrumented build), as it does not require any manual
clone or build steps. Instead, Pixi will automatically handle both the build
and installation of the package.

See https://github.com/scipy/scipy/pull/24066 for a full example of downstream use.

Each package definition is contained in a subdirectory.
Currently defined package variants:

- `default`
- `asan`: ASan-instrumented build with `-Db_sanitize=address`

## Maintenance

- Keep host dependency requirements up to date
- For dependencies on upstream CPython Pixi packages, keep the git revision at a compatible version

## Opportunities for future improvement

- More package variants (such as TSan, UBSan)
- Support for Windows
- Using a single `pixi.toml` for all package variants is blocked on https://github.com/prefix-dev/pixi/issues/2813
- Consider pinning dependency versions to guard against upstream breakages over time
